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


# ─────────────────────────────────────────────
# 影像預覽輔助函式 (修改版：改為存檔而非彈出視窗)
# ─────────────────────────────────────────────
def preview_images(image_list, source_nodes, hint_user_id, activity):
    """
    將影像儲存到本地 debug_images 資料夾，避免彈出視窗導致 OpenCV 報錯
    """
    save_dir = "debug_images"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for i, img_b64 in enumerate(image_list):
        try:
            # 處理 Base64 前綴
            img_clean = img_b64.split(',')[1] if ',' in img_b64 else img_b64
            img_data = base64.b64decode(img_clean)
            nparr = np.frombuffer(img_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if frame is None:
                continue

            # 檔案命名：時間_用戶_動作_節點.jpg
            timestamp = datetime.datetime.now().strftime("%H%M%S")
            node_name = source_nodes[i] if i < len(source_nodes) else f"img_{i}"
            filename = f"{save_dir}/{timestamp}_{hint_user_id}_{activity}_{node_name}.jpg"
            
            # 直接存檔，不呼叫 cv2.imshow
            cv2.imwrite(filename, frame)
            print(f"📸 [Saved] 影像已存至: {filename}")

        except Exception as e:
            print(f"⚠️ [Preview Skip] 無法處理影像 {i}: {e}")

    # 移除 cv2.waitKey(1)，因為現在不需要處理視窗刷新

# ─────────────────────────────────────────────
# /predict
# ─────────────────────────────────────────────
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
        node_scores  = data.get('node_scores', [])
        image_count  = data.get('image_count', len(image_list))

        if not image_list:
            return jsonify({"error": "image_list is empty"}), 400

        print(f"\n[Predict] user={hint_user_id} | activity={activity} | "
              f"images={image_count} | nodes={source_nodes}")

        # ── 預覽：收到影像立刻跳出視窗 ──
        preview_images(image_list, source_nodes, hint_user_id, activity)

        # ── est_pos ──
        est_pos = None
        if user_pos_raw:
            est_pos = {
                "x": float(user_pos_raw.get("x", 0)),
                "z": float(user_pos_raw.get("z", 0))
            }
            print(f"[Position] est_pos = {est_pos}")

        # ── Stage 1：VLM 感知 ──
        perception_res = perception.analyze_action_burst(data)

        user_id        = perception_res["user"]
        action         = perception_res["action"]
        detected_items = perception_res["items"]
        vlm_desc       = perception_res["result"].get("context", "Observed behavior.")
        voted_bound_id = perception_res["bound_instance"]

        # ── Stage 2：語義綁定 ──
        final_bound_label = voted_bound_id

        if voted_bound_id != "Unknown_Area":
            if detected_items:
                db.scene_snapshots.update_one(
                    {"label": voted_bound_id},
                    {"$addToSet": {"items": {"$each": detected_items}}}
                )
                print(f"[Inventory] {voted_bound_id} → {detected_items}")

            memory.bind_and_update(
                user_id=user_id,
                action=action,
                est_pos=est_pos,
                vlm_description=vlm_desc,
                detected_items=detected_items,
                target_label=voted_bound_id
            )
        else:
            if est_pos:
                final_bound_label = memory.bind_and_update(
                    user_id=user_id,
                    action=action,
                    est_pos=est_pos,
                    vlm_description=vlm_desc,
                    detected_items=detected_items
                )
            else:
                print("⚠️ [Skip bind] est_pos 為空且投票失敗")

        # ── Stage 3：FAISS 向量記憶 ──
        if final_bound_label and "Unknown" not in final_bound_label:
            vector_memory.add_memory(user_id, action, final_bound_label, vlm_desc)
        else:
            print("⚠️ [Skip FAISS] 位置不確定")

        print(f"[Done] {user_id} @ {final_bound_label} → {action} | items: {detected_items}")

        return jsonify({
            "status":        "Success",
            "user":          user_id,
            "action":        action,
            "bound_to":      final_bound_label,
            "items":         detected_items,
            "description":   vlm_desc,
            "estimated_pos": est_pos
        }), 200

    except Exception as e:
        import traceback
        print(f"❌ [Predict Error] {e}\n{traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500


# ─────────────────────────────────────────────
# /scene
# ─────────────────────────────────────────────
@app.route('/scene', methods=['POST'])
def handle_scene():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data received"}), 400

        objects = data.get('objects', [])
        count   = memory.sync_scene(objects)

        print(f"[Scene Sync] 同步 {count} 個家具到 MongoDB")
        return jsonify({"status": "Success", "synced_count": count}), 200
    except Exception as e:
        print(f"❌ [Scene Error] {e}")
        return jsonify({"error": str(e)}), 500


# ─────────────────────────────────────────────
# /query
# ─────────────────────────────────────────────
@app.route('/query', methods=['POST'])
def query_habit():
    try:
        data       = request.get_json()
        user_query = data.get('query', '')
        if not user_query:
            return jsonify({"error": "Empty query"}), 400

        results = vector_memory.search_habit(user_query, top_k=3)

        answer = "I don't remember."
        if results:
            best   = results[0]
            answer = f"I remember seeing something related to '{user_query}' near {best['instance']}."

        return jsonify({
            "status":           "Success",
            "semantic_results": results,
            "answer":           answer
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    host = getattr(CONFIG, 'FLASK_HOST', '0.0.0.0')
    port = int(getattr(CONFIG, 'FLASK_PORT', 5000))
    print(f"\nRobot Brain Server Running on {host}:{port}")
    app.run(host=host, port=port, debug=False)