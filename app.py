import cv2
import numpy as np
import base64
import os
import datetime
from flask import Flask, request, jsonify
from pymongo import MongoClient

# --- 1. Configuration and Core Module Loading ---
from config import Config
from modules.perception import PerceptionEngine
from modules.spatial import SpatialReasoning
from modules.memory import MemoryManager
from modules.memory_vector import VectorMemory

# Initialize embedding model and database
from sentence_transformers import SentenceTransformer
from insightface.app import FaceAnalysis

app = Flask(__name__)
CONFIG = Config

# SBERT for semantic object alignment
sbert_model = SentenceTransformer('paraphrase-MiniLM-L6-v2', device='cpu')
mongo_client = MongoClient(CONFIG.MONGO_URI)

# --- 🚀 Automatically Create MongoDB 2D Index ---
db = mongo_client[CONFIG.DB_NAME]
try:
    db.scene_snapshots.create_index([("pos", "2d")])
    print("✅ MongoDB geospatial index (2D Index) is ready")
except Exception as e:
    print(f"ℹ️ Index creation notice: {e}")

# --- 2. Initialize InsightFace ---
face_app = FaceAnalysis(name='buffalo_l', root='.', providers=['CPUExecutionProvider'])
face_app.prepare(ctx_id=-1, det_size=(640, 640))

def load_face_bank(path="./faces"):
    face_bank = {}
    if not os.path.exists(path):
        os.makedirs(path)
        return face_bank
    for fn in os.listdir(path):
        if fn.endswith((".jpg", ".png", ".jpeg")):
            img = cv2.imread(os.path.join(path, fn))
            faces = face_app.get(img)
            if faces:
                face = sorted(
                    faces,
                    key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]),
                    reverse=True
                )[0]
                face_bank[os.path.splitext(fn)[0]] = face.normed_embedding
    print(f"✅ Face feature bank loaded: {list(face_bank.keys())}")
    return face_bank

# Spatial reasoning module
spatial_reasoning = SpatialReasoning()

# Inject components into perception engine (pass spatial module for internal voting use)
perception = PerceptionEngine(
    CONFIG.OLLAMA_URL,
    CONFIG.OLLAMA_MODEL,
    face_analyzer=face_app,
    face_bank=load_face_bank(),
    spatial_module=spatial_reasoning
)

memory = MemoryManager(mongo_client, embedding_model=sbert_model)
vector_memory = VectorMemory()

# --- 3. Route Implementations ---

@app.route('/predict', methods=['POST'])
def predict():
    """
    Visual perception -> spatial voting -> furniture & small object binding -> semantic memory storage
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data received"}), 400

        image_list = data.get('image_list', [])
        robot_pos = data.get('robot_pos')
        robot_yaw = data.get('robot_rotation_y')
        camera_fov = data.get('camera_fov')

        # --- Stage 1: Multimodal perception and spatial voting ---
        # Internally completed:
        # 1. VLM JSON parsing
        # 2. Multi-frame action voting
        # 3. Multi-frame spatial coordinate alignment voting
        perception_res = perception.analyze_action_burst(
            image_list,
            robot_pos=robot_pos,
            robot_yaw=robot_yaw,
            camera_fov=camera_fov
        )

        user_id = perception_res["user"]
        action = perception_res["action"]
        detected_items = perception_res["items"]  # This is a list: ['apple', 'book']
        vlm_desc = perception_res["result"].get("context", "Observed behavior.")

        # Most stable furniture ID selected by voting across three frames
        voted_bound_id = perception_res["bound_instance"]

        # --- Stage 2: Spatial coordinate estimation (use middle frame as representative) ---
        mid_idx = len(image_list) // 2
        img_b64 = image_list[mid_idx]
        encoded_data = img_b64.split(',')[1] if ',' in img_b64 else img_b64
        nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
        mid_frame_cv = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        est_pos = spatial_reasoning.estimate_coordinate(
            mid_frame_cv, robot_pos, robot_yaw, camera_fov
        )

        # --- Stage 3: Semantic binding and database update ---

        # A. Update furniture inventory in MongoDB (dynamic small object binding)
        # If voting result is not Unknown, force update that furniture's contents
        final_bound_label = voted_bound_id

        if voted_bound_id != "Unknown_Area":
            # 1. Update furniture item list (Inventory)
            if detected_items:
                db.scene_snapshots.update_one(
                    {"label": voted_bound_id},
                    {"$addToSet": {"items": {"$each": detected_items}}}
                )
                print(f"[Inventory Sync] {voted_bound_id} updated items: {detected_items}")

            # 2. Record user's semantic behavior (Semantic Memory)
            memory.bind_and_update(
                user_id=user_id,
                action=action,
                est_pos=est_pos,
                vlm_description=vlm_desc,
                detected_items=detected_items,
                target_label=voted_bound_id  # Pass voted ID
            )
        else:
            # If voting fails, attempt one final distance-based lookup using est_pos
            final_bound_label = memory.bind_and_update(
                user_id=user_id,
                action=action,
                est_pos=est_pos,
                vlm_description=vlm_desc,
                detected_items=detected_items
            )

        # B. FAISS vector storage (for queries like "Where are sweet things in the house?")
        if final_bound_label and "Unknown" not in final_bound_label:
            vector_memory.add_memory(user_id, action, final_bound_label, vlm_desc)
        else:
            print("⚠️ [Skip FAISS] Location uncertain, skipping vectorization.")

        # --- Final response ---
        print(f"[Final Success] {user_id} @ {final_bound_label} -> {action} (Items: {detected_items})")

        return jsonify({
            "status": "Success",
            "user": user_id,
            "action": action,
            "bound_to": final_bound_label,
            "items": detected_items,
            "description": vlm_desc,
            "estimated_pos": est_pos
        }), 200

    except Exception as e:
        print(f"❌ [Error] Predict API: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/scene', methods=['POST'])
def handle_scene():
    """Receive furniture coordinate data from Unity and sync to MongoDB"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data received"}), 400

        objects = data.get('objects', [])

        # Call MemoryManager internal sync_scene
        count = memory.sync_scene(objects)

        print(f"[Scene Sync] Successfully synchronized {count} furniture entities to MongoDB.")
        return jsonify({"status": "Success", "synced_count": count}), 200
    except Exception as e:
        print(f"❌ [Scene Error] Sync failed: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/query', methods=['POST'])
def query_habit():
    """Long-term memory query: search historical behavior via natural language using FAISS"""
    try:
        data = request.get_json()
        user_query = data.get('query', "")
        if not user_query:
            return jsonify({"error": "Empty query"}), 400

        # 1. Use FAISS to find the top 3 most relevant memories
        results = vector_memory.search_habit(user_query, top_k=3)

        # Generate simple response
        answer = "I don't remember."
        if results:
            best = results[0]
            answer = f"I remember seeing something related to '{user_query}' near {best['instance']}."

        return jsonify({
            "status": "Success",
            "semantic_results": results,
            "answer": answer
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    # Start server
    host = getattr(CONFIG, 'FLASK_HOST', '0.0.0.0')
    port = int(getattr(CONFIG, 'FLASK_PORT', 5000))
    print(f"\nRobot Brain Server Running on {host}:{port}")
    app.run(host=host, port=port, debug=False)