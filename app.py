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

# ── SBERT-based action normalization ─────────────────────────────────────────
# Prototype descriptions for each behavioral label.
# SBERT encodes the VLM free-text output and finds the closest prototype.
# No keyword rules — purely semantic similarity.
BEHAVIOR_PROTOTYPES = {
    "Drink":       "a person drinking, holding a bottle or cup, sipping a beverage",
    "SittingIdle": "a person sitting still, resting, idle on a sofa or chair",
    "Reading":     "a person reading a book, holding a book, looking at pages",
    "Typing":      "a person typing on a keyboard, working at a computer or laptop",
    "Watching":    "a person watching television, looking at a screen or monitor",
    "Standing":    "a person standing still, not doing anything specific",
    "Walking":     "a person walking, moving across the room",
    "Sleeping":    "a person sleeping, lying down, resting on a bed",
    "Eating":      "a person eating food, having a meal",
}

_proto_vecs   = None   # cached on first call
_proto_labels = list(BEHAVIOR_PROTOTYPES.keys())

def _get_proto_vecs():
    """Encode prototype descriptions once and cache."""
    global _proto_vecs
    if _proto_vecs is None:
        import torch
        _proto_vecs = sbert_model.encode(
            list(BEHAVIOR_PROTOTYPES.values()),
            normalize_embeddings=True,
            convert_to_tensor=True,
        )
        print(f"[SBERT Norm] Prototype vectors built ({len(_proto_labels)} classes)")
    return _proto_vecs

def normalize_action_sbert(raw: str, threshold: float = 0.35) -> str:
    """
    Map VLM free-text output to a canonical behavior label using SBERT.

    - Computes cosine similarity between the VLM description and each
      BEHAVIOR_PROTOTYPES entry.
    - Returns the label of the closest prototype if sim >= threshold.
    - Falls back to the raw string if no prototype is close enough
      (handles unknown behaviors gracefully).
    - threshold=0.35 is conservative; typical sim for correct matches
      is 0.45–0.75.
    """
    if not raw or raw.strip() in ("", "none"):
        return "unknown"

    import torch
    q_vec      = sbert_model.encode(
        [raw], normalize_embeddings=True, convert_to_tensor=True
    )
    sims       = torch.nn.functional.cosine_similarity(q_vec, _get_proto_vecs())
    best_idx   = int(sims.argmax())
    best_sim   = float(sims[best_idx])
    best_label = _proto_labels[best_idx]

    if best_sim >= threshold:
        print(f"[SBERT Norm] '{raw[:60]}' → '{best_label}' (sim={best_sim:.3f})")
        return best_label
    else:
        print(f"[SBERT Norm] '{raw[:60]}' → fallback "
              f"(best={best_label} sim={best_sim:.3f} < {threshold})")
        return raw


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
            print(f"[Preview Skip] {e}")


def _wait_for_scene(max_wait: float = 12.0, poll: float = 1.0):
    import time as _time
    waited = 0.0
    while waited < max_wait:
        if db.scene_snapshots.count_documents({}) > 0:
            return
        print(f"    [WaitScene] scene_snapshots empty, waited {waited:.0f}s...")
        _time.sleep(poll)
        waited += poll
    print("   [WaitScene] Timeout, proceeding without scene data")


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
        virtual_hour = data.get('virtual_hour')

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
        action         = perception_res["action"]   # raw VLM output, kept for log_eval
        detected_items = perception_res["items"]
        all_items      = perception_res["all_items"]
        spatial_rels   = perception_res["spatial"]
        vlm_desc       = perception_res["result"].get("context", "Observed behavior.")
        vlm_object     = perception_res["bound_instance"]

        # ── Action normalization: SBERT semantic matching ─────────────────
        # Replaces the old first-word lookup table (ACTION_NORMALIZE).
        # normalize_action_sbert handles full VLM sentences such as
        # "man sitting at a desk working on his laptop" → "Typing"
        action_label = normalize_action_sbert(action)
        print(f"[VLM] action='{action}' → label='{action_label}' "
              f"| object={vlm_object} | {vlm_ms}ms")

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
            action            = action_label,   # normalized label, not raw VLM output
            est_pos           = est_pos,
            vlm_description   = vlm_desc,
            detected_items    = detected_items,
            all_items         = all_items,
            spatial_relations = spatial_rels,
            target_label      = vlm_object,
            room_name         = room_name,
        )
        print(f"[Bind] '{vlm_object}' -> '{final_bound_label}'")

        # log_eval: keep raw VLM output as vlm_output for Exp1 F1 calculation
        # action_label is stored separately so Exp2 graph reads clean labels
        log_eval(
            experiment   = "exp1_exp2",
            ground_truth = activity,
            vlm_output   = action,              # raw output for F1 comparison
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
                action            = action_label,   # normalized label stored in semantic_memories
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

        # ── Manifold: feature vector → record → intent prediction ─────────
        manifold_point_id = ""
        intent_prediction = {"trigger": False, "intent": "unknown", "confidence": 0.0}
        has_proposal      = False

        try:
            confidence_str = perception_res.get("result", {}).get("confidence", "unknown")

            prev_doc = db.manifold_points.find_one(
                {"user_id": user_id},
                sort=[("timestamp", -1)]
            )
            prev_action_label = prev_doc.get("action", "unknown") if prev_doc else "unknown"
            print(f"[Manifold] prev_action='{prev_action_label}'")

            feature_vec = manifold_engine.build_feature_vector(
                user_id        = user_id,
                action         = action_label,
                user_pos       = est_pos,
                room_name      = room_name,
                detected_items = detected_items,
                confidence     = confidence_str,
                virtual_hour   = virtual_hour,
                prev_action    = prev_action_label,
            )

            manifold_point_id = manifold_engine.record_point(
                user_id      = user_id,
                feature_vec  = feature_vec,
                action       = action_label,
                bound_label  = final_bound_label,
                virtual_hour = virtual_hour,
                prev_action  = prev_action_label,
            )

            manifold_engine.maybe_refit(user_id)

            intent_prediction = manifold_engine.predict_intent(
                user_id         = user_id,
                current_feature = feature_vec,
            )

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


@app.route('/exp_checkpoint', methods=['GET', 'POST'])
def exp_checkpoint():
    try:
        if request.method == 'POST':
            body       = request.get_json(force=True, silent=True) or {}
            episode    = body.get('episode', 0)
            user_id    = body.get('user_id', 'User_Mom')
            action     = body.get('action', 'Drink')
            experiment = 'experiment2'
        else:
            experiment = request.args.get('experiment', 'experiment2')
            episode    = request.args.get('step', 0, type=int)
            user_id    = request.args.get('user', 'User_Mom')
            action     = request.args.get('action', 'Drink')

        obs = db.observation_logs.find_one(
            {"user": user_id, "action": {"$regex": action, "$options": "i"}},
            sort=[("weight", -1)]
        )
        weight = obs["weight"] if obs else 0

        similarity = 0.0
        try:
            query   = f"{user_id} {action}"
            results = vector_memory.search_habit(query, user_id=user_id, top_k=1)
            if results:
                similarity = float(results[0].get("similarity", 0.0))
        except Exception as fe:
            print(f"[Checkpoint] FAISS query skipped: {fe}")

        checkpoint_doc = {
            "experiment": experiment,
            "episode":    episode,
            "user_id":    user_id,
            "action":     action,
            "weight":     weight,
            "similarity": round(similarity, 4),
            "timestamp":  datetime.datetime.utcnow(),
        }
        db["exp_checkpoint_logs"].insert_one(checkpoint_doc)

        print(f"[Checkpoint] {experiment} ep={episode} "
              f"user={user_id} action={action} "
              f"weight={weight} sim={similarity:.4f}")

        return jsonify({
            "status":     "ok",
            "experiment": experiment,
            "episode":    episode,
            "user_id":    user_id,
            "action":     action,
            "weight":     weight,
            "similarity": round(similarity, 4),
        }), 200

    except Exception as e:
        print(f"[Checkpoint Error] {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/scene', methods=['POST'])
def handle_scene():
    try:
        data    = request.get_json()
        objects = data.get('objects', [])

        if not objects:
            return jsonify({"status": "empty"}), 200

        now  = datetime.datetime.utcnow()
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

        print(f"[Scene] received {len(docs)} objects → raw_objects")
        return jsonify({"status": "Success", "received": len(docs)}), 200

    except Exception as e:
        print(f"[Scene Error] {e}")
        return jsonify({"error": str(e)}), 500


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

        print(f"[DynamicSync] {count} objects updated (source=sensor)")
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
            print(f"[Proposal] {proposal.get('user_id')} → {proposal.get('intent')}")
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
    print(f"   VLM model    : {CONFIG.VLM_MODEL}")
    print(f"   LLM model    : {CONFIG.LLM_MODEL}")
    print(f"   Action norm  : SBERT semantic matching ({len(BEHAVIOR_PROTOTYPES)} classes)")
    app.run(host=host, port=port, debug=False)