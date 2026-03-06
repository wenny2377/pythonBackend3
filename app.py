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

        if not image_list:
            return jsonify({"error": "image_list is empty"}), 400

        print(f"\n[Predict] user={hint_user_id} | activity={activity} | "
              f"images={image_count} | nodes={source_nodes}")

        preview_images(image_list, source_nodes, hint_user_id, activity)

        est_pos = None
        if user_pos_raw:
            est_pos = {
                "x": float(user_pos_raw.get("x", 0)),
                "z": float(user_pos_raw.get("z", 0))
            }

        # ── Stage 1：VLM 感知（含空間關係）──
        perception_res  = perception.analyze_action_burst(data)
        user_id         = perception_res["user"]
        action          = perception_res["action"]
        detected_items  = perception_res["items"]        # 互動物品
        all_items       = perception_res["all_items"]    # 畫面全部物品
        spatial_rels    = perception_res["spatial"]      # 空間介係詞關係
        vlm_desc        = perception_res["result"].get("context", "Observed behavior.")
        vlm_object      = perception_res["bound_instance"]

        print(f"[VLM] action={action} | object={vlm_object}")
        print(f"[VLM] interacting={detected_items} | scene={all_items}")
        print(f"[VLM] spatial_relations={spatial_rels}")

        # ── Stage 2：家具綁定 + 物品雙軌 + 空間關係存入 MongoDB ──
        final_bound_label = memory.bind_and_update(
            user_id=user_id,
            action=action,
            est_pos=est_pos,
            vlm_description=vlm_desc,
            detected_items=detected_items,   # 人直接使用的物品
            all_items=all_items,             # 畫面中所有物品
            spatial_relations=spatial_rels,  # 空間關係
            target_label=vlm_object
        )
        print(f"[Bind] '{vlm_object}' → '{final_bound_label}'")

        # ── Stage 3：FAISS 向量化（含空間關係文字）──
        furniture_pos = None
        mongo_id      = None

        if final_bound_label and "Unknown" not in final_bound_label:
            furniture_doc = db.scene_snapshots.find_one({"label": final_bound_label})
            if furniture_doc:
                furniture_pos = furniture_doc.get('pos')
                mongo_id      = furniture_doc.get('_id')

            # 把空間關係也加進向量化文字，提升查詢召回率
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
        else:
            print("⚠️ [Skip FAISS] 家具綁定失敗")

        print(f"[Done] {user_id} @ {final_bound_label} → {action}")

        return jsonify({
            "status":             "Success",
            "user":               user_id,
            "action":             action,
            "bound_to":           final_bound_label,
            "interacting_items":  detected_items,
            "all_items":          all_items,
            "spatial_relations":  spatial_rels,
            "description":        vlm_desc,
            "estimated_pos":      est_pos,
            "furniture_pos":      furniture_pos
        }), 200

    except Exception as e:
        import traceback
        print(f"❌ [Predict Error] {e}\n{traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500


@app.route('/scene', methods=['POST'])
def handle_scene():
    try:
        data    = request.get_json()
        objects = data.get('objects', [])
        count   = memory.sync_scene(objects)
        print(f"[Scene Sync] 同步 {count} 個家具")
        return jsonify({"status": "Success", "synced_count": count}), 200
    except Exception as e:
        print(f"❌ [Scene Error] {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/query', methods=['POST'])
def query_habit():
    """
    RAG 查詢：FAISS 模糊搜尋 → MongoDB 取最新座標與空間關係 → 組合回答

    支援查詢類型：
    - 「媽媽通常在哪裡喝水？」→ nav_target 座標
    - 「刀子通常放在哪裡？」→ spatial_relations 搜尋
    - 「流理台上通常有什麼？」→ scene_snapshots.items
    """
    try:
        data       = request.get_json()
        user_query = data.get('query', '')
        user_id    = data.get('userID', None)

        if not user_query:
            return jsonify({"error": "Empty query"}), 400

        print(f"\n[Query] '{user_query}' | user={user_id}")

        # Step 1：FAISS 模糊搜尋
        results = vector_memory.search_habit(user_query, user_id=user_id, top_k=5)

        # Step 2：用 instance label 從 MongoDB 補最新資料
        for r in results:
            if r.get('instance') and r['instance'] != 'Unknown_Area':
                fresh = db.scene_snapshots.find_one({"label": r['instance']})
                if fresh:
                    r['furniture_pos']      = fresh.get('pos')
                    r['all_items']          = fresh.get('items', [])
                    r['spatial_relations']  = fresh.get('spatial_relations', [])

        # Step 3：最高頻習慣
        top_habit  = vector_memory.get_top_habit(user_query, user_id=user_id, top_k=1)
        nav_target = None
        answer     = "I don't remember."

        if top_habit:
            fresh_doc = db.scene_snapshots.find_one({"label": top_habit['instance']})
            if fresh_doc:
                nav_target                   = fresh_doc.get('pos')
                top_habit['all_items']       = fresh_doc.get('items', [])
                top_habit['spatial_relations'] = fresh_doc.get('spatial_relations', [])
            else:
                nav_target = top_habit.get('furniture_pos')

            interact_str = ", ".join(top_habit.get('interacting_items', [])) or "nothing specific"
            answer = (f"Based on observations, {user_id or 'the user'} usually "
                      f"{top_habit['action']} near {top_habit['instance']} "
                      f"(seen {top_habit['count']} times), "
                      f"typically interacting with: {interact_str}.")

        elif results:
            best       = results[0]
            nav_target = best.get('furniture_pos')
            answer     = (f"I remember {best['user']} {best['action']} "
                          f"near {best['instance']}.")

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
        print(f"❌ [Query Error] {e}\n{traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    host = getattr(CONFIG, 'FLASK_HOST', '0.0.0.0')
    port = int(getattr(CONFIG, 'FLASK_PORT', 5000))
    print(f"\nRobot Brain Server Running on {host}:{port}")
    app.run(host=host, port=port, debug=False)


# ─────────────────────────────────────────────
# 延遲初始化（避免循環 import）
# ─────────────────────────────────────────────
from modules.interaction import InteractionEngine
from modules.training_exporter import TrainingExporter
from modules.cleanup import CleanupManager

interaction_engine  = InteractionEngine(
    mongo_client=mongo_client,
    vector_memory=vector_memory,
    ollama_url=CONFIG.OLLAMA_URL,
    model_name=CONFIG.OLLAMA_MODEL
)
training_exporter = TrainingExporter(mongo_client)
cleanup_manager   = CleanupManager(mongo_client)


# 啟動定時清理（每 24 小時執行一次）
cleanup_manager.start_scheduler(interval_hours=24)


# ─────────────────────────────────────────────
# /interact — 人機交互主路由
# ─────────────────────────────────────────────
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
        print(f"❌ [Interact Error] {e}\n{traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500


# ─────────────────────────────────────────────
# /log_navigation — 接收 Unity 導航路徑訓練資料
# ─────────────────────────────────────────────
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
        print(f"❌ [NavLog Error] {e}")
        return jsonify({"error": str(e)}), 500


# ─────────────────────────────────────────────
# /export_training — 匯出所有訓練資料 JSONL
# ─────────────────────────────────────────────
@app.route('/export_training', methods=['POST'])
def export_training():
    try:
        data            = request.get_json() or {}
        user_id_filter  = data.get('userID', None)
        export_type     = data.get('type', 'all')  # all / perception / dialogue / navigation / scene / habit

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
        print(f"[Export] 完成：{stats} | 共 {total} 筆")

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
        print(f"❌ [Export Error] {e}\n{traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500


# ─────────────────────────────────────────────
# /cleanup — 手動觸發清理
# ─────────────────────────────────────────────
@app.route('/cleanup', methods=['POST'])
def manual_cleanup():
    try:
        stats = cleanup_manager.run_all(auto=False)
        return jsonify({"status": "Success", "cleaned": stats}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ─────────────────────────────────────────────
# /cleanup/status — 查詢各 collection 筆數
# ─────────────────────────────────────────────
@app.route('/cleanup/status', methods=['GET'])
def cleanup_status():
    try:
        return jsonify(cleanup_manager.status()), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500