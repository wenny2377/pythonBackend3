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

app    = Flask(__name__)
CONFIG = Config

# ──────────────────────────────────────────────────────
# GPU 設定
# ──────────────────────────────────────────────────────
sbert_model = SentenceTransformer('paraphrase-MiniLM-L6-v2', device='cuda')
print("✅ SBERT loaded on CUDA")

mongo_client = MongoClient(CONFIG.MONGO_URI)
db           = mongo_client[CONFIG.DB_NAME]

try:
    db.scene_snapshots.create_index([("pos", "2d")])
    print("✅ MongoDB 2D Index ready")
except Exception as e:
    print(f"ℹ️ Index notice: {e}")

# ── FIX: 移除 spatial_module=None（perception v4 已移除此參數）──
perception = PerceptionEngine(
    ollama_url       = CONFIG.OLLAMA_URL,
    model_name       = CONFIG.VLM_MODEL,   # llava-phi3：視覺辨識
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

# 啟動時把 MongoDB dynamic_objects 同步進 FAISS
# 支援同事的 sensor 資料 + 自己的 VLM 資料
vector_memory.sync_from_mongo(db.dynamic_objects)

# 程式結束時確保 BulkWriteBuffer 全部寫入
atexit.register(perception.shutdown)

# Ollama 單執行緒鎖（GPU 模式仍建議保留，避免顯存衝突）
_ollama_lock = threading.Lock()


# ──────────────────────────────────────────────────────
# 工具函式
# ──────────────────────────────────────────────────────
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
            print(f"📸 [Saved] {ts}_{hint_user_id}_{activity}_{node_name}.jpg")
        except Exception as e:
            print(f"⚠️ [Preview Skip] {e}")


def _wait_for_scene(max_wait: float = 12.0, poll: float = 1.0):
    """scene_snapshots 還空時最多等 max_wait 秒，等 classifier 跑完"""
    import time as _time
    waited = 0.0
    while waited < max_wait:
        if db.scene_snapshots.count_documents({}) > 0:
            return
        print(f"   ⏳ [WaitScene] scene_snapshots empty, waited {waited:.0f}s...")
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


# ──────────────────────────────────────────────────────
# /predict  主感知路由
# ──────────────────────────────────────────────────────
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

        if not room_name and source_nodes:
            first_node = source_nodes[0]
            room_name  = first_node.rsplit('_Cam', 1)[0] if '_Cam' in first_node else first_node
            print(f"[Room Fallback] '{first_node}' -> room='{room_name}'")

        if not image_list:
            return jsonify({"error": "image_list is empty"}), 400

        print(f"\n[Predict] user={hint_user_id} | activity={activity} | room={room_name} | "
              f"images={image_count} | nodes={source_nodes}")

        preview_images(image_list, source_nodes, hint_user_id, activity)

        est_pos = None
        if user_pos_raw:
            est_pos = {
                "x": float(user_pos_raw.get("x", 0)),
                "z": float(user_pos_raw.get("z", 0))
            }
        # user_pos 存進 DB，讓 classifier 知道人在哪（held_by 判斷用）
        if est_pos and hint_user_id:
            db.user_positions.update_one(
                {"user_id": hint_user_id},
                {"$set": {"user_id": hint_user_id, "x": est_pos["x"], "z": est_pos["z"],
                           "room": room_name, "updated_at": datetime.datetime.utcnow()}},
                upsert=True
            )


        data['room_name'] = room_name
        # scene_snapshots 尚未建立時等待（/scene 和 /predict 幾乎同時到）
        _wait_for_scene(max_wait=12.0)


        # ── VLM 推理（計時供 eval_logs）──
        acquired = _ollama_lock.acquire(timeout=180)
        t0 = time.time()
        try:
            # perception v4：內部已處理 dynamic_objects 更新
            perception_res = perception.analyze_action_burst(data)
        finally:
            if acquired:
                _ollama_lock.release()
        vlm_ms = int((time.time() - t0) * 1000)

        user_id        = perception_res["user"]
        action         = perception_res["action"]
        detected_items = perception_res["items"]
        all_items      = perception_res["all_items"]
        spatial_rels   = perception_res["spatial"]
        vlm_desc       = perception_res["result"].get("context", "Observed behavior.")
        vlm_object     = perception_res["bound_instance"]

        print(f"[VLM] action={action} | object={vlm_object} | {vlm_ms}ms")

        if action == "none":
            print("[Skip] VLM no valid action")
            return jsonify({
                "status":   "no_action",
                "user":     user_id,
                "action":   "none",
                "bound_to": "Unknown_Area",
                "reason":   "VLM returned no valid action"
            }), 200

        # ── memory.bind_and_update：只更新 observation_logs + activity_sequences
        # ── dynamic_objects 已由 perception v4 處理，不重複更新
        final_bound_label = memory.bind_and_update(
            user_id           = user_id,
            action            = action,
            est_pos           = est_pos,
            vlm_description   = vlm_desc,
            detected_items    = detected_items,
            all_items         = all_items,
            spatial_relations = spatial_rels,
            target_label      = vlm_object,
            room_name         = room_name,
        )
        print(f"[Bind] '{vlm_object}' -> '{final_bound_label}'")

        # ── 實驗一、二自動記錄 ──
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
                user_id         = user_id,
                action          = action,
                furniture_label = final_bound_label,
                vlm_description = f"{vlm_desc} {spatial_text}".strip(),
                detected_items  = detected_items,
                all_items       = all_items,
                spatial_relations = spatial_rels,
                furniture_pos   = furniture_pos,
                mongo_id        = mongo_id
            )

            # ── FAISS 同步最新 dynamic_objects（從 DB 讀取 perception 剛寫入的結果）──
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

        print(f"[Done] {user_id} @ {final_bound_label} -> {action} | {vlm_ms}ms")

        return jsonify({
            "status":            "Success",
            "user":              user_id,
            "action":            action,
            "bound_to":          final_bound_label,
            "interacting_items": detected_items,
            "all_items":         all_items,
            "spatial_relations": spatial_rels,
            "description":       vlm_desc,
            "estimated_pos":     est_pos,
            "furniture_pos":     furniture_pos,
            "vlm_inference_ms":  vlm_ms,
        }), 200

    except Exception as e:
        import traceback
        print(f"[Predict Error] {e}\n{traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500


# ──────────────────────────────────────────────────────
# /exp_checkpoint  實驗三A/B checkpoint 記錄
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


# ──────────────────────────────────────────────────────
# 其餘 routes
# ──────────────────────────────────────────────────────
@app.route('/scene', methods=['POST'])
def handle_scene():
    """
    收到 Unity / 同事 sensor 的物件資料
    全部存進 raw_objects（不在這裡分類）
    classifier.py 背景執行緒每 5 秒讀取 raw_objects → 分類
    """
    try:
        data      = request.get_json()
        objects   = data.get('objects', [])
        timestamp = data.get('timestamp', '')

        if not objects:
            return jsonify({"status": "empty"}), 200

        now = datetime.datetime.utcnow()
        docs = []
        for obj in objects:
            label = obj.get('label', '').lower().strip()
            if not label:
                continue
            docs.append({
                "label":     label,
                "x":         obj.get('x', 0),
                "y":         obj.get('y', 0),
                "z":         obj.get('z', 0),
                "room":      obj.get('room', ''),
                "image":     obj.get('image', ''),
                "source":    obj.get('source', 'sensor'),
                "held_by":   obj.get('held_by', ''),  # Unity 送來的持有者
                "processed": False,
                "received_at": now,
            })

        if docs:
            db.raw_objects.insert_many(docs)

        print(f"[Scene] 收到 {len(docs)} 個物件 → raw_objects（等待分類）")
        return jsonify({"status": "Success", "received": len(docs)}), 200

    except Exception as e:
        print(f"[Scene Error] {e}")
        return jsonify({"error": str(e)}), 500



@app.route('/dynamic_sync', methods=['POST'])
def dynamic_sync():
    """
    接收 Unity DynamicObjectSync 傳來的動態物件位置更新。
    source: "sensor"（同事 or Unity 模擬）
    直接寫進 dynamic_objects，並同步 FAISS。
    """
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
            # last_seen_on / spatial_rel 由 VLM perception.py 補充
            # sensor 端只有 label / room / position
            last_seen_on = obj.get('last_seen_on', 'unknown')
            spatial_rel  = obj.get('spatial_rel', 'near')

            # 寫進 MongoDB
            # sensor_pos = 物件真實世界座標（sensor 提供）
            # furniture_pos = 最近家具座標（VLM binding 後補充）
            db.dynamic_objects.update_one(
                {"label": label},
                {
                    "$set": {
                        "room":        room,
                        "sensor_pos":  position,   # 真實座標，只有 sensor 寫
                        "last_seen":   datetime.datetime.utcnow(),
                        "source":      source,
                    },
                    "$inc":         {"seen_count": 1},
                    "$setOnInsert": {
                        "first_seen":  datetime.datetime.utcnow(),
                        "last_seen_on": "unknown",
                        "spatial_rel":  "unknown",
                    },
                },
                upsert=True
            )

            # 同步 FAISS
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


# ──────────────────────────────────────────────────────
# 延遲載入（避免 circular import）
# ──────────────────────────────────────────────────────
from modules.interaction import InteractionEngine
from modules.training_exporter import TrainingExporter

interaction_engine = InteractionEngine(
    mongo_client  = mongo_client,
    vector_memory = vector_memory,
    ollama_url    = CONFIG.OLLAMA_URL,
    model_name    = CONFIG.LLM_MODEL,      # gemma3:4b：語言推理
)
training_exporter = TrainingExporter(mongo_client)



if __name__ == "__main__":
    host = getattr(CONFIG, 'FLASK_HOST', '0.0.0.0')
    port = int(getattr(CONFIG, 'FLASK_PORT', 5000))
    print(f"\n🚀 Robot Brain Server on {host}:{port}")
    print(f"   SBERT device : cuda")
    print(f"   VLM model    : {CONFIG.VLM_MODEL}   ← perception")
    print(f"   LLM model    : {CONFIG.LLM_MODEL}  ← interaction/RAG")
    app.run(host=host, port=port, debug=False)