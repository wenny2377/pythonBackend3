import cv2
import numpy as np
import base64
import os
import datetime
from flask import Flask, request, jsonify
from pymongo import MongoClient

from config import Config
from modules.perception import PerceptionEngine
from modules.memory import MemoryManager
from modules.memory_vector import VectorMemory

from sentence_transformers import SentenceTransformer

app = Flask(__name__)
CONFIG = Config

sbert_model  = SentenceTransformer('paraphrase-MiniLM-L6-v2', device='cpu')
mongo_client = MongoClient(CONFIG.MONGO_URI)

db = mongo_client[CONFIG.DB_NAME]
try:
    db.scene_snapshots.create_index([("pos", "2d")])
    print("✅ MongoDB 2D Index ready")
except Exception as e:
    print(f"ℹ️ Index notice: {e}")

perception = PerceptionEngine(
    ollama_url=CONFIG.OLLAMA_URL,
    model_name=CONFIG.OLLAMA_MODEL,
    face_analyzer=None,
    face_bank=None,
    spatial_module=None
)

memory        = MemoryManager(mongo_client, embedding_model=sbert_model)
vector_memory = VectorMemory()


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

        data['room_name'] = room_name

        perception_res = perception.analyze_action_burst(data)
        user_id        = perception_res["user"]
        action         = perception_res["action"]
        detected_items = perception_res["items"]
        all_items      = perception_res["all_items"]
        spatial_rels   = perception_res["spatial"]
        vlm_desc       = perception_res["result"].get("context", "Observed behavior.")
        vlm_object     = perception_res["bound_instance"]

        print(f"[VLM] action={action} | object={vlm_object}")
        print(f"[VLM] interacting={detected_items} | scene={all_items}")
        print(f"[VLM] spatial_relations={spatial_rels}")

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
            user_id=user_id,
            action=action,
            est_pos=est_pos,
            vlm_description=vlm_desc,
            detected_items=detected_items,
            all_items=all_items,
            spatial_relations=spatial_rels,
            target_label=vlm_object,
            room_name=room_name,
        )
        print(f"[Bind] '{vlm_object}' -> '{final_bound_label}'")

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
                user_id=user_id,
                action=action,
                furniture_label=final_bound_label,
                vlm_description=f"{vlm_desc} {spatial_text}".strip(),
                detected_items=detected_items,
                all_items=all_items,
                spatial_relations=spatial_rels,
                furniture_pos=furniture_pos,
                mongo_id=mongo_id
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

        print(f"[Done] {user_id} @ {final_bound_label} -> {action}")

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
            "furniture_pos":     furniture_pos
        }), 200

    except Exception as e:
        import traceback
        print(f"[Predict Error] {e}\n{traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500


@app.route('/scene', methods=['POST'])
def handle_scene():
    try:
        data    = request.get_json()
        objects = data.get('objects', [])
        count   = memory.sync_scene(objects)
        print(f"[Scene Sync] {count}")
        return jsonify({"status": "Success", "synced_count": count}), 200
    except Exception as e:
        print(f"[Scene Error] {e}")
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
        print(f"[Export] {stats} | total={total}")

        return jsonify({
            "status":     "Success",
            "stats":      stats,
            "total":      total,
            "output_dir": training_exporter.output_dir,
            "files": [
                "perception_data.jsonl",
                "dialogue_data.jsonl",
                "navigation_data.jsonl",
                "scene_graph_data.jsonl",
                "habit_sequence_data.jsonl"
            ]
        }), 200

    except Exception as e:
        import traceback
        print(f"[Export Error] {e}\n{traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500


@app.route('/cleanup', methods=['POST'])
def manual_cleanup():
    try:
        stats = cleanup_manager.run_all(auto=False)
        return jsonify({"status": "Success", "cleaned": stats}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/cleanup/status', methods=['GET'])
def cleanup_status():
    try:
        return jsonify(cleanup_manager.status()), 200
    except Exception as e:
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
            query=query,
            user_id=user_id,
            robot_pos=robot_pos,
            user_pos=user_pos,
            room=room
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
            choice=choice,
            nav_target=nav_target,
            nav_label=nav_label,
            user_id=user_id,
            query=query,
        )
        return jsonify(result), 200

    except Exception as e:
        import traceback
        print(f"[Confirm Error] {e}\n{traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    host = getattr(CONFIG, 'FLASK_HOST', '0.0.0.0')
    port = int(getattr(CONFIG, 'FLASK_PORT', 5000))
    print(f"\nRobot Brain Server Running on {host}:{port}")
    app.run(host=host, port=port, debug=False)


from modules.interaction import InteractionEngine
from modules.training_exporter import TrainingExporter
from cleanup import CleanupManager

interaction_engine = InteractionEngine(
    mongo_client=mongo_client,
    vector_memory=vector_memory,
    ollama_url=CONFIG.OLLAMA_URL,
    model_name=CONFIG.OLLAMA_MODEL
)
training_exporter = TrainingExporter(mongo_client)
cleanup_manager   = CleanupManager(mongo_client)

cleanup_manager.start_scheduler(interval_hours=24)