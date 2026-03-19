import cv2
import numpy as np
import base64
import os
import time
import datetime
import threading
import atexit
from flask import Flask, request, jsonify
from pymongo import MongoClient

from config import Config
from modules.perception import PerceptionEngine
from modules.memory import MemoryManager
from modules.memory_vector import VectorMemory
from classifier import ObjectClassifier

from sentence_transformers import SentenceTransformer
from modules.manifold_engine   import ManifoldEngine
from modules.service_proposal  import ServiceProposalEngine

app    = Flask(__name__)
CONFIG = Config

# ──────────────────────────────────────────────────────
# GPU 設定
# ──────────────────────────────────────────────────────
sbert_model = SentenceTransformer('paraphrase-MiniLM-L6-v2', device='cuda')
print(" SBERT loaded on CUDA")

mongo_client = MongoClient(CONFIG.MONGO_URI)
db           = mongo_client[CONFIG.DB_NAME]

try:
    db.scene_snapshots.create_index([("pos", "2d")])
    print(" MongoDB 2D Index ready")
except Exception as e:
    print(f" Index notice: {e}")

perception = PerceptionEngine(
    ollama_url       = CONFIG.OLLAMA_URL,
    model_name       = CONFIG.VLM_MODEL,
    face_analyzer    = None,
    face_bank        = None,
    mongo_uri        = CONFIG.MONGO_URI,
    db_name          = CONFIG.DB_NAME,
    sbert_model_name = 'paraphrase-MiniLM-L6-v2',
)

memory        = MemoryManager(mongo_client, embedding_model=sbert_model)
vector_memory = VectorMemory(device='cuda')
classifier    = ObjectClassifier(db)
classifier.start()

vector_memory.sync_from_mongo(db.dynamic_objects)

atexit.register(perception.shutdown)

manifold_engine = ManifoldEngine(db=db, sbert_model=sbert_model)
proposal_engine = ServiceProposalEngine(
    db         = db,
    ollama_url = CONFIG.OLLAMA_URL,
    llm_model  = CONFIG.LLM_MODEL,
)

_ollama_lock = threading.Lock()

# ── Action 正規化表：VLM 自由描述第一個單字 → 分類標籤 ──────────
ACTION_NORMALIZE = {
    "drink":     "Drink",       "drinking":   "Drink",
    "sit":       "SittingIdle", "sitting":    "SittingIdle",
    "read":      "Reading",     "reading":    "Reading",
    "type":      "Typing",      "typing":     "Typing",
    "watch":     "Watching",    "watching":   "Watching",
    "sleep":     "Sleeping",    "sleeping":   "Sleeping",
    "eat":       "Eating",      "eating":     "Eating",
    "stand":     "Standing",    "standing":   "Standing",
    "exercise":  "Exercising",  "exercising": "Exercising",
    "cook":      "Cooking",     "cooking":    "Cooking",
    "walk":      "Walking",     "walking":    "Walking",
    "lying":     "Sleeping",    "lie":        "Sleeping",
}


def preview_images(image_list, source_nodes, hint_user_id, activity):
    save_dir = "debug_images"
    os.makedirs(save_dir, exist_ok=True)
    for i, img_b64 in enumerate(image_list):
        try:
            img_clean = img_b64.split(',')[1] if ',' in img_b64 else img_b64
            nparr     = np.frombuffer(base64.b64decode(img_clean), np.uint8)
            frame     = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if frame is None:
                continue
            ts        = datetime.datetime.now().strftime("%H%M%S")
            node_name = source_nodes[i] if i < len(source_nodes) else f"img_{i}"
            cv2.imwrite(f"{save_dir}/{ts}_{hint_user_id}_{activity}_{node_name}.jpg", frame)
            print(f" [Saved] {ts}_{hint_user_id}_{activity}_{node_name}.jpg")
        except Exception as e:
            print(f"⚠️ [Preview Skip] {e}")


def _wait_for_scene(max_wait: float = 12.0, poll: float = 1.0):
    import time as _time
    waited = 0.0
    while waited < max_wait:
        if db.scene_snapshots.count_documents({}) > 0:
            return
        print(f"    [WaitScene] scene_snapshots empty, waited {waited:.0f}s...")
        _time.sleep(poll)
        waited += poll
    print("   ⚠️  [WaitScene] Timeout, proceeding without scene data")


def log_eval(experiment, ground_truth, vlm_output, bound_label,
             user_id, room, vlm_ms, binding_results=None):
    doc = {
        "experiment":       experiment,
        "ground_truth":     ground_truth,
        "vlm_output":       vlm_output,
        "is_correct":       ground_truth.lower() == vlm_output.lower() if ground_truth else None,
        "bound_label":      bound_label,
        "user_id":          user_id,
        "room":             room,
        "vlm_inference_ms": vlm_ms,
        "timestamp":        datetime.datetime.now(),
    }
    if binding_results:
        doc["binding_results"] = binding_results
    db["eval_logs"].insert_one(doc)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data received"}), 400

        image_list   = data.get('image_list', [])
        hint_user_id = data.get('userID', 'Unknown_User')
        activity     = data.get('activity', '')
        user_pos_raw = data.get('user_pos')
        source_nodes = data.get('source_nodes', [])
        image_count  = data.get('image_count', len(image_list))
        room_name    = data.get('room_name', '')
        virtual_hour = data.get('virtual_hour')   # ← Exp4 時段

        if not room_name and source_nodes:
            first_node = source_nodes[0]
            room_name  = first_node.rsplit('_Cam', 1)[0] if '_Cam' in first_node else first_node
            print(f"[Room Fallback] '{first_node}' -> room='{room_name}'")

        if not image_list:
            return jsonify({"error": "image_list is empty"}), 400

        print(f"\n[Predict] user={hint_user_id} | activity={activity} | room={room_name} | "
              f"images={image_count} | nodes={source_nodes} | virtual_hour={virtual_hour}")

        preview_images(image_list, source_nodes, hint_user_id, activity)

        est_pos = None
        if user_pos_raw:
            est_pos = {
                "x": float(user_pos_raw.get("x", 0)),
                "z": float(user_pos_raw.get("z", 0))
            }

        if est_pos and hint_user_id:
            db.user_positions.update_one(
                {"user_id": hint_user_id},
                {"$set": {"user_id": hint_user_id, "x": est_pos["x"], "z": est_pos["z"],
                           "room": room_name, "updated_at": datetime.datetime.utcnow()}},
                upsert=True
            )

        data['room_name'] = room_name

        _wait_for_scene(max_wait=12.0)

        acquired = _ollama_lock.acquire(timeout=180)
        t0 = time.time()
        try:
            perception_res = perception.analyze_action_burst(data)
        finally:
            if acquired:
                _ollama_lock.release()
        vlm_ms = int((time.time() - t0) * 1000)

        user_id        = perception_res["user"]
        action         = perception_res["action"]          # VLM 原始輸出，保留給 log_eval
        detected_items = perception_res["items"]
        all_items      = perception_res["all_items"]
        spatial_rels   = perception_res["spatial"]
        vlm_desc       = perception_res["result"].get("context", "Observed behavior.")
        vlm_object     = perception_res["bound_instance"]

        # ── Action 正規化：VLM 自由描述 → 分類標籤 ──────────────
        _first_word  = action.lower().split()[0] if action else "none"
        action_label = ACTION_NORMALIZE.get(_first_word, action)
        print(f"[VLM] action='{action}' → label='{action_label}' | object={vlm_object} | {vlm_ms}ms")

        if action == "none":
            print("[Skip] VLM no valid action")
            return jsonify({
                "status":   "no_action",
                "user":     user_id,
                "action":   "none",
                "bound_to": "Unknown_Area",
                "reason":   "VLM returned no valid action"
            }), 200

        final_bound_label = memory.bind_and_update(
            user_id           = user_id,
            action            = action,           # 保留原始 action 給 memory
            est_pos           = est_pos,
            vlm_description   = vlm_desc,
            detected_items    = detected_items,
            all_items         = all_items,
            spatial_relations = spatial_rels,
            target_label      = vlm_object,
            room_name         = room_name,
        )
        print(f"[Bind] '{vlm_object}' -> '{final_bound_label}'")

        # log_eval 用原始 action（為了 Exp1 VLM 準確率計算）
        log_eval(
            experiment   = "exp1_exp2",
            ground_truth = activity,
            vlm_output   = action,
            bound_label  = final_bound_label,
            user_id      = user_id,
            room         = room_name,
            vlm_ms       = vlm_ms,
        )

        furniture_pos = None
        mongo_id      = None

        if final_bound_label and "Unknown" not in final_bound_label:
            furniture_doc = db.scene_snapshots.find_one({"label": final_bound_label})
            if furniture_doc:
                furniture_pos = furniture_doc.get('pos')
                mongo_id      = furniture_doc.get('_id')

            spatial_text = " ".join(
                [f"{r['subject']} {r['relation']} {r['object']}" for r in spatial_rels]
            ) if spatial_rels else ""

            vector_memory.add_memory(
                user_id           = user_id,
                action            = action,       # 保留原始 action
                furniture_label   = final_bound_label,
                vlm_description   = f"{vlm_desc} {spatial_text}".strip(),
                detected_items    = detected_items,
                all_items         = all_items,
                spatial_relations = spatial_rels,
                furniture_pos     = furniture_pos,
                mongo_id          = mongo_id
            )

            scene_items       = [i for i in all_items if i not in detected_items]
            all_dynamic_items = list(set(detected_items + scene_items))
            for item_label in all_dynamic_items:
                dyn_doc = db.dynamic_objects.find_one({"label": item_label.lower()})
                if dyn_doc:
                    vector_memory.upsert_dynamic_object(
                        label          = dyn_doc["label"],
                        room           = dyn_doc.get("room", room_name),
                        last_seen_on   = dyn_doc.get("last_seen_on", final_bound_label),
                        spatial_rel    = dyn_doc.get("spatial_rel", "near"),
                        furniture_pos  = dyn_doc.get("furniture_pos", furniture_pos),
                        seen_count     = dyn_doc.get("seen_count", 1),
                        interact_count = dyn_doc.get("interact_count", 0),
                        interacted_by  = dyn_doc.get("interacted_by", []),
                    )
        else:
            print("[Skip FAISS] bind failed")

        print(f"[Done] {user_id} @ {final_bound_label} -> {action_label} | {vlm_ms}ms")

        # ── Manifold：特徵向量 → 記錄 → 預判 ──────────────────────
        manifold_point_id = ""
        intent_prediction = {"trigger": False, "intent": "unknown", "confidence": 0.0}
        has_proposal      = False

        try:
            confidence_str = perception_res.get("result", {}).get("confidence", "unknown")

            # ── 取得前一個行為（序列上下文）──
            prev_doc = db.manifold_points.find_one(
                {"user_id": user_id},
                sort=[("timestamp", -1)]
            )
            prev_action_label = prev_doc.get("action", "unknown") if prev_doc else "unknown"
            print(f"[Manifold] prev_action='{prev_action_label}'")

            # 1. 建特徵向量（含時間編碼 + 序列上下文）
            feature_vec = manifold_engine.build_feature_vector(
                user_id        = user_id,
                action         = action_label,
                user_pos       = est_pos,
                room_name      = room_name,
                detected_items = detected_items,
                confidence     = confidence_str,
                virtual_hour   = virtual_hour,        # ← 時間編碼
                prev_action    = prev_action_label,   # ← 序列上下文
            )

            # 2. 記錄到 manifold_points
            manifold_point_id = manifold_engine.record_point(
                user_id      = user_id,
                feature_vec  = feature_vec,
                action       = action_label,
                bound_label  = final_bound_label,
                virtual_hour = virtual_hour,
                prev_action  = prev_action_label,     # ← 新增
            )

            # 3. 每 50 筆觸發非同步 refit
            manifold_engine.maybe_refit(user_id)

            # 4. 預判意圖
            intent_prediction = manifold_engine.predict_intent(
                user_id         = user_id,
                current_feature = feature_vec,
            )

            # 5. 觸發時請 ServiceProposalEngine 決策
            if intent_prediction.get("trigger"):
                dynamic_results = vector_memory.search_dynamic(
                    action_label, top_k=5, user_filter=user_id
                )
                proposal_result = proposal_engine.evaluate(
                    user_id           = user_id,
                    intent_prediction = intent_prediction,
                    manifold_point_id = manifold_point_id,
                    user_pos          = est_pos,
                    dynamic_results   = dynamic_results,
                )
                has_proposal = proposal_result.get("has_proposal", False)

        except Exception as manifold_err:
            print(f"[Manifold] non-critical error: {manifold_err}")

        return jsonify({
            "status":            "Success",
            "user":              user_id,
            "action":            action,
            "action_label":      action_label,
            "bound_to":          final_bound_label,
            "interacting_items": detected_items,
            "all_items":         all_items,
            "spatial_relations": spatial_rels,
            "description":       vlm_desc,
            "estimated_pos":     est_pos,
            "furniture_pos":     furniture_pos,
            "vlm_inference_ms":  vlm_ms,
            "manifold_point_id": manifold_point_id,
            "intent_prediction": intent_prediction,
            "has_proposal":      has_proposal,
        }), 200

    except Exception as e:
        import traceback
        print(f"[Predict Error] {e}\n{traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500


# ──────────────────────────────────────────────────────
# /set_virtual_hour
# ──────────────────────────────────────────────────────
@app.route('/set_virtual_hour', methods=['POST'])
def set_virtual_hour():
    try:
        data = request.get_json(force=True, silent=True) or {}
        hour = data.get('virtual_hour', -1)
        app.config['VIRTUAL_HOUR'] = float(hour)
        print(f"[VirtualHour] set to {hour}")
        return jsonify({"status": "ok", "virtual_hour": hour}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ──────────────────────────────────────────────────────
# /exp_checkpoint
# ──────────────────────────────────────────────────────
@app.route('/exp_checkpoint', methods=['GET'])
def exp_checkpoint():
    try:
        experiment = request.args.get('experiment', '')
        step       = request.args.get('step', 0, type=int)
        day        = request.args.get('day', 0, type=int)
        user       = request.args.get('user', 'User_Mom')
        action     = request.args.get('action', 'drinking')

        checkpoint_doc = {
            "experiment": experiment,
            "step":       step,
            "day":        day,
            "timestamp":  datetime.datetime.now(),
        }

        if experiment == "exp3a":
            obs = db.observation_logs.find_one(
                {"user": user, "action": action},
                sort=[("weight", -1)]
            )
            weight = obs["weight"] if obs else 0

            similarity = 0.0
            try:
                results    = vector_memory.search_habit(f"{user} {action}", user_id=user, top_k=1)
                similarity = float(results[0].get("similarity", 0.0)) if results else 0.0
            except Exception as fe:
                print(f"[Checkpoint] FAISS skipped: {fe}")

            checkpoint_doc.update({
                "user":       user,
                "action":     action,
                "weight":     weight,
                "similarity": round(similarity, 4),
            })
            print(f"[Checkpoint Exp3A] step={step} weight={weight} sim={similarity:.4f}")

        elif experiment == "exp3b":
            try:
                pipeline = [
                    {"$match": {"timestamp": {"$gte": datetime.datetime.now() - datetime.timedelta(hours=2)}}},
                    {"$group": {"_id": {"user": "$user", "action": "$action"}, "count": {"$sum": 1}}}
                ]
                today_obs_raw = list(db.observation_logs.aggregate(pipeline))
                today_obs = [{"user": r["_id"].get("user",""), "action": r["_id"].get("action",""), "count": r["count"]} for r in today_obs_raw]
                print(f"[Checkpoint Exp3B] day={day} obs={len(today_obs)}")
            except Exception as pe:
                print(f"[Checkpoint Exp3B] aggregate skipped: {pe}")
                checkpoint_doc["today_observations"] = []

        db["exp_checkpoints"].insert_one(checkpoint_doc)
        return jsonify({"status": "ok", "checkpoint": checkpoint_doc}), 200

    except Exception as e:
        print(f"[Checkpoint Error] {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/scene', methods=['POST'])
def handle_scene():
    try:
        data      = request.get_json()
        objects   = data.get('objects', [])

        if not objects:
            return jsonify({"status": "empty"}), 200

        now = datetime.datetime.utcnow()
        docs = []
        for obj in objects:
            label = obj.get('label', '').lower().strip()
            if not label:
                continue
            docs.append({
                "label":      label,
                "x":          obj.get('x', 0),
                "y":          obj.get('y', 0),
                "z":          obj.get('z', 0),
                "room":       obj.get('room', ''),
                "image":      obj.get('image', ''),
                "source":     obj.get('source', 'sensor'),
                "held_by":    obj.get('held_by', ''),
                "processed":  False,
                "received_at": now,
            })

        if docs:
            db.raw_objects.insert_many(docs)

        print(f"[Scene] 收到 {len(docs)} 個物件 → raw_objects（等待分類）")
        return jsonify({"status": "Success", "received": len(docs)}), 200

    except Exception as e:
        print(f"[Scene Error] {e}")
        return jsonify({"error": str(e)}), 500


# ──────────────────────────────────────────────────────
# /dynamic_sync
# ──────────────────────────────────────────────────────
@app.route('/dynamic_sync', methods=['POST'])
def dynamic_sync():
    try:
        data    = request.get_json()
        objects = data.get('objects', [])
        if not objects:
            return jsonify({"status": "empty"}), 200

        count = 0
        for obj in objects:
            label = obj.get('label', '').lower().strip()
            if not label:
                continue

            room         = obj.get('room', '')
            position     = obj.get('position', [0, 0])
            source       = obj.get('source', 'sensor')
            last_seen_on = obj.get('last_seen_on', 'unknown')
            spatial_rel  = obj.get('spatial_rel', 'near')

            db.dynamic_objects.update_one(
                {"label": label},
                {
                    "$set": {
                        "room":       room,
                        "sensor_pos": position,
                        "last_seen":  datetime.datetime.utcnow(),
                        "source":     source,
                    },
                    "$inc":         {"seen_count": 1},
                    "$setOnInsert": {
                        "first_seen":   datetime.datetime.utcnow(),
                        "last_seen_on": "unknown",
                        "spatial_rel":  "unknown",
                    },
                },
                upsert=True
            )

            dyn_doc = db.dynamic_objects.find_one({"label": label})
            if dyn_doc:
                vector_memory.upsert_dynamic_object(
                    label          = label,
                    room           = room,
                    last_seen_on   = last_seen_on,
                    spatial_rel    = spatial_rel,
                    furniture_pos  = position,
                    seen_count     = dyn_doc.get("seen_count", 1),
                    interact_count = dyn_doc.get("interact_count", 0),
                    interacted_by  = dyn_doc.get("interacted_by", []),
                )
            count += 1

        print(f"[DynamicSync] {count} 個物件已更新（source=sensor）")
        return jsonify({"status": "Success", "updated": count}), 200

    except Exception as e:
        import traceback
        print(f"[DynamicSync Error] {e}\n{traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500


@app.route('/query', methods=['POST'])
def query_habit():
    try:
        data       = request.get_json()
        user_query = data.get('query', '')
        user_id    = data.get('userID', None)

        if not user_query:
            return jsonify({"error": "Empty query"}), 400

        print(f"\n[Query] '{user_query}' | user={user_id}")

        results = vector_memory.search_habit(user_query, user_id=user_id, top_k=5)

        for r in results:
            if r.get('instance') and r['instance'] != 'Unknown_Area':
                fresh = db.scene_snapshots.find_one({"label": r['instance']})
                if fresh:
                    r['furniture_pos']     = fresh.get('pos')
                    r['all_items']         = fresh.get('items', [])
                    r['spatial_relations'] = fresh.get('spatial_relations', [])

        top_habit  = vector_memory.get_top_habit(user_query, user_id=user_id, top_k=1)
        nav_target = None
        answer     = "I don't remember."

        if top_habit:
            fresh_doc = db.scene_snapshots.find_one({"label": top_habit['instance']})
            if fresh_doc:
                nav_target                     = fresh_doc.get('pos')
                top_habit['all_items']         = fresh_doc.get('items', [])
                top_habit['spatial_relations'] = fresh_doc.get('spatial_relations', [])
            else:
                nav_target = top_habit.get('furniture_pos')

            interact_str = ", ".join(top_habit.get('interacting_items', [])) or "nothing specific"
            answer = (
                f"Based on observations, {user_id or 'the user'} usually "
                f"{top_habit['actions'][0] if top_habit.get('actions') else 'does something'} "
                f"near {top_habit['instance']} "
                f"(seen {top_habit['count']} times), "
                f"typically interacting with: {interact_str}."
            )
        elif results:
            best       = results[0]
            nav_target = best.get('furniture_pos')
            answer     = f"I remember {best['user']} {best['action']} near {best['instance']}."

        print(f"[Answer] {answer}")
        print(f"[Nav]    {nav_target}")

        return jsonify({
            "status":           "Success",
            "answer":           answer,
            "nav_target":       nav_target,
            "top_habit":        top_habit,
            "semantic_results": results[:3]
        }), 200

    except Exception as e:
        import traceback
        print(f"[Query Error] {e}\n{traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500


@app.route('/log_navigation', methods=['POST'])
def log_navigation():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data"}), 400

        db["navigation_logs"].insert_one({
            "user_id":        data.get("user_id"),
            "intent":         data.get("intent"),
            "start_pos":      data.get("start_pos"),
            "goal_pos":       data.get("goal_pos"),
            "waypoints":      data.get("waypoints", []),
            "waypoint_count": data.get("waypoint_count", 0),
            "success":        data.get("success", False),
            "fail_reason":    data.get("fail_reason", ""),
            "total_time":     data.get("total_time", 0),
            "total_distance": data.get("total_distance", 0),
            "timestamp":      datetime.datetime.utcnow()
        })

        print(f"[NavLog] {data.get('user_id')} | intent={data.get('intent')} | "
              f"waypoints={data.get('waypoint_count')} | success={data.get('success')}")

        return jsonify({"status": "Success"}), 200

    except Exception as e:
        print(f"[NavLog Error] {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/export_training', methods=['POST'])
def export_training():
    try:
        data           = request.get_json() or {}
        user_id_filter = data.get('userID', None)
        export_type    = data.get('type', 'all')

        if export_type == 'all':
            stats = training_exporter.export_all(user_id_filter)
        elif export_type == 'perception':
            stats = {"perception": training_exporter.export_perception(user_id_filter)}
        elif export_type == 'dialogue':
            stats = {"dialogue": training_exporter.export_dialogue(user_id_filter)}
        elif export_type == 'navigation':
            stats = {"navigation": training_exporter.export_navigation(user_id_filter)}
        elif export_type == 'scene':
            stats = {"scene_graph": training_exporter.export_scene_graph()}
        elif export_type == 'habit':
            stats = {"habit_sequence": training_exporter.export_habit_sequence(user_id_filter)}
        else:
            return jsonify({"error": f"Unknown type: {export_type}"}), 400

        total = sum(stats.values())
        return jsonify({
            "status":     "Success",
            "stats":      stats,
            "total":      total,
            "output_dir": training_exporter.output_dir,
        }), 200

    except Exception as e:
        import traceback
        print(f"[Export Error] {e}\n{traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500


@app.route('/interact', methods=['POST'])
def interact():
    try:
        data      = request.get_json()
        query     = data.get('query', '')
        user_id   = data.get('userID', 'Unknown')
        robot_pos = data.get('robot_pos')
        user_pos  = data.get('user_pos')
        room      = data.get('room', '')

        if not query:
            return jsonify({"error": "Empty query"}), 400

        print(f"\n[Interact] user={user_id} | query='{query}' | room={room}")

        result = interaction_engine.process(
            query     = query,
            user_id   = user_id,
            robot_pos = robot_pos,
            user_pos  = user_pos,
            room      = room
        )
        return jsonify(result), 200

    except Exception as e:
        import traceback
        print(f"[Interact Error] {e}\n{traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500


@app.route('/service_proposal', methods=['GET'])
def service_proposal():
    try:
        proposal = proposal_engine.get_next_proposal()
        if proposal:
            print(f"[Proposal] 發出提案: {proposal.get('user_id')} → {proposal.get('intent')}")
            return jsonify(proposal), 200
        return jsonify({"status": "no_proposal"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/service_response', methods=['POST'])
def service_response():
    try:
        data        = request.get_json()
        proposal_id = data.get("proposal_id", "")
        user_id     = data.get("user_id", "Unknown")
        result      = data.get("result", "ignored")

        if not proposal_id:
            return jsonify({"error": "proposal_id required"}), 400

        response = proposal_engine.handle_response(
            proposal_id     = proposal_id,
            user_id         = user_id,
            result          = result,
            manifold_engine = manifold_engine,
        )
        return jsonify(response), 200
    except Exception as e:
        import traceback
        print(f"[ServiceResponse Error] {e}\n{traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500


@app.route('/service_history', methods=['GET'])
def service_history():
    try:
        user_id   = request.args.get("user_id")
        query     = {"user_id": user_id} if user_id else {}
        proposals = list(
            db.service_proposals.find(query, {"_id": 0})
            .sort("created_at", -1).limit(50)
        )
        for p in proposals:
            for k in ["created_at", "responded_at"]:
                if k in p and hasattr(p[k], "isoformat"):
                    p[k] = p[k].isoformat()

        stats = list(db.intent_stats.find(query, {"_id": 0}))
        return jsonify({
            "proposals":    proposals,
            "intent_stats": stats,
            "total":        len(proposals),
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/interact/confirm', methods=['POST'])
def interact_confirm():
    try:
        data       = request.get_json()
        choice     = int(data.get('choice', 3))
        nav_target = data.get('nav_target')
        nav_label  = data.get('nav_label', '')
        user_id    = data.get('userID', 'Unknown')
        query      = data.get('query', '')

        result = interaction_engine.confirm(
            choice     = choice,
            nav_target = nav_target,
            nav_label  = nav_label,
            user_id    = user_id,
            query      = query,
        )
        return jsonify(result), 200

    except Exception as e:
        import traceback
        print(f"[Confirm Error] {e}\n{traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500


from modules.interaction import InteractionEngine
from modules.training_exporter import TrainingExporter

interaction_engine = InteractionEngine(
    mongo_client  = mongo_client,
    vector_memory = vector_memory,
    ollama_url    = CONFIG.OLLAMA_URL,
    model_name    = CONFIG.LLM_MODEL,
)
training_exporter = TrainingExporter(mongo_client)


if __name__ == "__main__":
    host = getattr(CONFIG, 'FLASK_HOST', '0.0.0.0')
    port = int(getattr(CONFIG, 'FLASK_PORT', 5000))
    print(f"\n Robot Brain Server on {host}:{port}")
    print(f"   SBERT device : cuda")
    print(f"   VLM model    : {CONFIG.VLM_MODEL}   ← perception")
    print(f"   LLM model    : {CONFIG.LLM_MODEL}  ← interaction/RAG")
    app.run(host=host, port=port, debug=False)