import cv2
import numpy as np
import base64
import os
import time
import datetime
import threading
import math
import json
import re
import atexit
import queue as _queue

from flask import Flask, request, jsonify, Response, stream_with_context
from pymongo import MongoClient

from config import Config
from modules.perception import PerceptionEngine
from modules.memory import MemoryManager
from modules.memory_vector import VectorMemory
from modules.classifier import ObjectClassifier, BASE_FURNITURE_KEYWORDS, OBJECT_CATEGORIES
from modules.manifold_engine import ManifoldEngine, build_x
from modules.service_proposal import ServiceProposalEngine

from sentence_transformers import SentenceTransformer

app    = Flask(__name__)
CONFIG = Config

sbert_model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda')
print("SBERT loaded on CUDA")

mongo_client = MongoClient(CONFIG.MONGO_URI)
db           = mongo_client[CONFIG.DB_NAME]

try:
    db.scene_snapshots.create_index([("pos", "2d")])
    db.observation_logs.create_index(
        [("last_seen", 1)],
        expireAfterSeconds=14 * 86400,
        name="observation_ttl_14d"
    )
    print("[MongoDB] indexes ready")
except Exception as e:
    print(f"[MongoDB] index notice: {e}")

manifold_engine = ManifoldEngine(db=db, sbert_model=sbert_model)

perception = PerceptionEngine(
    ollama_url       = CONFIG.OLLAMA_URL,
    model_name       = CONFIG.VLM_MODEL,
    face_analyzer    = None,
    face_bank        = None,
    mongo_uri        = CONFIG.MONGO_URI,
    db_name          = CONFIG.DB_NAME,
    sbert_model_name = 'all-MiniLM-L6-v2',
    manifold_engine  = manifold_engine,
)

memory        = MemoryManager(mongo_client, embedding_model=sbert_model)
vector_memory = VectorMemory(device='cuda')
classifier    = ObjectClassifier(db)
classifier.start()

vector_memory.sync_from_mongo(db.dynamic_objects)

atexit.register(perception.shutdown)

proposal_engine = ServiceProposalEngine(
    db         = db,
    ollama_url = CONFIG.OLLAMA_URL,
    llm_model  = CONFIG.LLM_MODEL,
)

_vlm_lock = threading.Lock()

_llm_task_queue   = _queue.Queue()
_llm_task_results = {}
_llm_task_lock    = threading.Lock()


def _llm_worker():
    while True:
        try:
            task_id, fn, args, kwargs = _llm_task_queue.get(timeout=1)
            try:
                result = fn(*args, **kwargs)
            except Exception as e:
                result = None
                print(f"[LLM Worker] task {task_id} failed: {e}")
            with _llm_task_lock:
                _llm_task_results[task_id] = result
            _llm_task_queue.task_done()
        except _queue.Empty:
            continue


_llm_worker_thread = threading.Thread(target=_llm_worker, daemon=True)
_llm_worker_thread.start()
print("[LLM Queue] worker started")


def submit_llm_task(fn, *args, **kwargs):
    import uuid
    task_id = str(uuid.uuid4())
    _llm_task_queue.put((task_id, fn, args, kwargs))
    _llm_task_queue.join()
    with _llm_task_lock:
        return _llm_task_results.pop(task_id, None)


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
            cv2.imwrite(
                f"{save_dir}/{ts}_{hint_user_id}_{activity}_{node_name}.jpg",
                frame)
        except Exception as e:
            print(f"[Preview Skip] {e}")


def _wait_for_scene(max_wait: float = 12.0, poll: float = 1.0):
    import time as _time
    waited = 0.0
    while waited < max_wait:
        if db.scene_snapshots.count_documents({}) > 0:
            return
        _time.sleep(poll)
        waited += poll


def _find_nearest_furniture(x: float, z: float,
                             room: str, max_dist: float = 1.5) -> str:
    query = {}
    if room:
        query["room"] = {"$regex": room, "$options": "i"}
    furniture_docs = list(
        db.scene_snapshots.find(query, {"label": 1, "pos": 1}))
    if not furniture_docs:
        furniture_docs = list(
            db.scene_snapshots.find({}, {"label": 1, "pos": 1}))

    best_label = "floor"
    best_dist  = float("inf")

    for doc in furniture_docs:
        pos = doc.get("pos")
        if not isinstance(pos, list) or len(pos) < 2:
            continue
        dist = math.sqrt((x - pos[0]) ** 2 + (z - pos[1]) ** 2)
        if dist < best_dist:
            best_dist  = dist
            best_label = doc["label"]

    return best_label if best_dist <= max_dist else "floor"


def _get_category_for_label(label: str) -> str:
    label_l = label.lower().strip()
    for cat, keywords in OBJECT_CATEGORIES.items():
        if label_l in keywords or any(kw in label_l for kw in keywords):
            return cat
    return "other"


def nightly_maintenance():
    print("[Maintenance] running nightly tasks...")
    try:
        db.observation_logs.update_many(
            {},
            {"$mul": {"weight": getattr(CONFIG, 'HABIT_DECAY_FACTOR', 0.95)}}
        )
        db.observation_logs.delete_many(
            {"weight": {"$lt": getattr(CONFIG, 'HABIT_MIN_WEIGHT', 1.0)}}
        )
        print("[Maintenance] habit decay done")

        if hasattr(interaction_engine, '_has_skill_manager') and \
                interaction_engine._has_skill_manager:
            sm = interaction_engine.skill_manager
            for doc in db.user_skills.find({}, {"user_id": 1}):
                try:
                    submit_llm_task(sm.nightly_refactor, doc["user_id"])
                except Exception as e:
                    print(f"[Maintenance] refactor failed for "
                          f"{doc['user_id']}: {e}")

        print("[Maintenance] done")
    except Exception as e:
        print(f"[Maintenance] error: {e}")

    threading.Timer(86400, nightly_maintenance).start()


def _stream_ollama(system: str, user_prompt: str):
    import requests as _req
    try:
        resp = _req.post(
            f"{CONFIG.OLLAMA_URL}/api/chat",
            json={
                "model":    CONFIG.LLM_MODEL,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user_prompt},
                ],
                "stream":  True,
                "options": {
                    "temperature": CONFIG.LLM_TEMPERATURE,
                    "num_predict": CONFIG.LLM_MAX_TOKENS,
                },
            },
            stream=True,
            timeout=90,
        )
        resp.raise_for_status()
        for line in resp.iter_lines():
            if not line:
                continue
            try:
                chunk = json.loads(line)
                token = chunk.get("message", {}).get("content", "")
                if token:
                    yield (f"data: "
                           f"{json.dumps({'type': 'token', 'content': token})}"
                           f"\n\n")
                if chunk.get("done"):
                    break
            except json.JSONDecodeError:
                continue
    except Exception as e:
        print(f"[Stream Ollama] error: {e}")
        yield (f"data: "
               f"{json.dumps({'type': 'token', 'content': 'Sorry, an error occurred.'})}"
               f"\n\n")


_robot_state = {
    "nav_target":  None,
    "nav_label":   "",
    "last_answer": "",
    "highlight":   "",
}


@app.route('/nav_target', methods=['GET'])
def get_nav_target():
    return jsonify({
        "nav_target": _robot_state["nav_target"],
        "nav_label":  _robot_state["nav_label"],
    })


@app.route('/highlight', methods=['GET'])
def get_highlight():
    return jsonify({"label": _robot_state["highlight"]})


@app.route('/last_answer', methods=['GET'])
def get_last_answer():
    return jsonify({"answer": _robot_state["last_answer"]})


@app.route('/interact/stream', methods=['POST'])
def interact_stream():
    data    = request.get_json()
    query   = data.get('query', '')
    user_id = data.get('userID', 'Unknown')
    room    = data.get('room', '')

    if not query:
        return jsonify({"error": "Empty query"}), 400

    def generate():
        intent        = interaction_engine._classify_intent(query)
        need_category = interaction_engine._extract_need_category(query)

        yield (f"data: "
               f"{json.dumps({'type': 'intent', 'intent': intent})}"
               f"\n\n")

        if intent == "interrupt":
            answer = "Understood, stopping now."
            _robot_state["last_answer"] = answer
            _robot_state["nav_target"]  = None
            _robot_state["nav_label"]   = ""
            _robot_state["highlight"]   = ""
            yield (f"data: "
                   f"{json.dumps({'type': 'token', 'content': answer})}"
                   f"\n\n")
            yield (f"data: "
                   f"{json.dumps({'type': 'done', 'nav_target': None, 'nav_label': None, 'confidence': 1.0, 'intent_type': 'interrupt', 'options': [], 'is_personalized': False})}"
                   f"\n\n")
            return

        if intent == "chat":
            system = (
                "You are a friendly home robot companion. "
                "Reply warmly and briefly in English. "
                "1-2 sentences max. Do NOT wrap in quotes. "
                "Do NOT suggest navigation."
            )
            chat_buf = ""
            for ev in _stream_ollama(
                    system, f'{user_id} said: "{query}"'):
                yield ev
                try:
                    parsed = json.loads(
                        ev.replace("data: ", "").strip())
                    if parsed.get("type") == "token":
                        chat_buf += parsed.get("content", "")
                except Exception:
                    pass
            chat_answer = chat_buf.strip().strip('"').strip("'")
            _robot_state["last_answer"] = chat_answer
            _robot_state["nav_target"]  = None
            _robot_state["nav_label"]   = ""
            _robot_state["highlight"]   = ""
            interaction_engine._schedule_skill_update(
                user_id=user_id, query=query,
                answer=chat_answer, env_snapshot="", rec_items=[],
            )
            yield (f"data: "
                   f"{json.dumps({'type': 'done', 'nav_target': None, 'nav_label': None, 'confidence': 1.0, 'intent_type': 'chat', 'options': [], 'is_personalized': False})}"
                   f"\n\n")
            return

        if intent == "query":
            result = interaction_engine._query_response(
                query, user_id, room)
            answer = result.get("answer", "")
            for char in answer:
                yield (f"data: "
                       f"{json.dumps({'type': 'token', 'content': char})}"
                       f"\n\n")
            _robot_state["last_answer"] = answer
            _robot_state["nav_target"]  = result.get("nav_target")
            _robot_state["nav_label"]   = result.get("nav_label", "")
            _robot_state["highlight"]   = result.get("nav_label", "")
            yield (f"data: "
                   f"{json.dumps({'type': 'done', 'nav_target': result.get('nav_target'), 'nav_label': result.get('nav_label'), 'confidence': result.get('confidence', 0.85), 'intent_type': 'query', 'options': result.get('options', []), 'is_personalized': False})}"
                   f"\n\n")
            return

        env_snapshot = interaction_engine._build_env_snapshot(need_category)

        skill_md = ""
        if interaction_engine._has_skill_manager:
            sm       = interaction_engine.skill_manager
            skill_md = (sm.get_skill_chunks(user_id, query)
                        or sm.get_skill(user_id) or "")

        from modules.interaction import ONE_SHOT_SYSTEM
        system = ONE_SHOT_SYSTEM.format(
            env_snapshot=env_snapshot,
            skill_md=skill_md or "(No skill profile yet.)",
        )

        furniture_labels = [
            d["label"]
            for d in db.scene_snapshots.find({}, {"label": 1})
            if "label" in d
        ]
        user_prompt = (
            f"User ID: {user_id}\n"
            f"User said: \"{query}\"\n"
            f"Robot current room: {room}\n"
            f"Known furniture: "
            f"{', '.join(furniture_labels) or 'unknown'}\n"
            f"Time: {datetime.datetime.now().strftime('%H:%M')}\n\n"
            f"Reply with the JSON format specified."
        )

        buffer = ""
        for event_line in _stream_ollama(system, user_prompt):
            yield event_line
            try:
                ev = json.loads(
                    event_line.replace("data: ", "").strip())
                if ev.get("type") == "token":
                    buffer += ev.get("content", "")
            except Exception:
                pass

        nav_target      = None
        nav_label       = "unknown"
        plain_answer    = ""
        options         = []
        is_personalized = bool(
            skill_md and "(No skill profile" not in skill_md)

        try:
            m = re.search(r'\{.*\}', buffer, re.DOTALL)
            if m:
                rj           = json.loads(m.group(0))
                plain_answer = (rj.get("answer", "")
                                .strip().strip('"').strip("'"))
                raw_nav      = rj.get("nav_target", "unknown")
                nav_label    = rj.get("nav_label", raw_nav)
                nav_target   = (
                    interaction_engine._resolve_pos(raw_nav)
                    if raw_nav and raw_nav != "unknown" else None
                )
                options = interaction_engine._build_options(
                    nav_target, nav_label, query)
        except Exception as e:
            print(f"[Stream] JSON parse error: {e}")

        _robot_state["last_answer"] = plain_answer or buffer.strip()
        _robot_state["nav_target"]  = nav_target
        _robot_state["nav_label"]   = (
            nav_label if nav_label != "unknown" else "")
        _robot_state["highlight"]   = (
            nav_label if nav_label != "unknown" else "")

        interaction_engine._schedule_skill_update(
            user_id=user_id, query=query,
            answer=buffer, env_snapshot=env_snapshot, rec_items=[],
        )

        yield (f"data: "
               f"{json.dumps({'type': 'done', 'nav_target': nav_target, 'nav_label': nav_label, 'confidence': 0.85, 'intent_type': 'oneshot', 'options': options, 'is_personalized': is_personalized})}"
               f"\n\n")

    return Response(
        stream_with_context(generate()),
        mimetype='text/event-stream',
        headers={'Cache-Control': 'no-cache',
                 'X-Accel-Buffering': 'no'},
    )


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
        user_fwd_raw = data.get('user_forward')
        source_nodes = data.get('source_nodes', [])
        room_name    = data.get('room_name', '')
        virtual_hour = data.get('virtual_hour')

        if not room_name and source_nodes:
            first_node = source_nodes[0]
            room_name  = (first_node.rsplit('_Cam', 1)[0]
                          if '_Cam' in first_node else first_node)

        if not image_list:
            return jsonify({"error": "image_list is empty"}), 400

        preview_images(
            image_list, source_nodes, hint_user_id, activity)

        est_pos = None
        if user_pos_raw:
            est_pos = {
                "x": float(user_pos_raw.get("x", 0)),
                "z": float(user_pos_raw.get("z", 0)),
            }

        est_forward = None
        if user_fwd_raw:
            est_forward = {
                "x": float(user_fwd_raw.get("x", 0)),
                "y": float(user_fwd_raw.get("y", 0)),
                "z": float(user_fwd_raw.get("z", 0)),
            }

        if est_pos and hint_user_id:
            db.user_positions.update_one(
                {"user_id": hint_user_id},
                {"$set": {
                    "user_id":    hint_user_id,
                    "x":          est_pos["x"],
                    "z":          est_pos["z"],
                    "room":       room_name,
                    "forward":    est_forward,
                    "updated_at": datetime.datetime.utcnow(),
                }},
                upsert=True,
            )

        data['room_name']    = room_name
        data['user_forward'] = est_forward
        _wait_for_scene(max_wait=12.0)

        acquired = _vlm_lock.acquire(timeout=180)
        t0 = time.time()
        try:
            perception_res = perception.analyze_action_burst(data)
        finally:
            if acquired:
                _vlm_lock.release()
        vlm_ms = int((time.time() - t0) * 1000)

        user_id         = perception_res["user"]
        action          = perception_res["action"]
        spatial_action  = perception_res.get("spatial_action", action)
        upgrade_reason  = perception_res.get("upgrade_reason", "")
        zone_label      = perception_res.get("zone_label", "")
        detected_items  = perception_res["items"]
        all_items       = perception_res["all_items"]
        spatial_rels    = perception_res["spatial"]
        vlm_desc        = perception_res["result"].get(
            "context", "Observed behavior.")
        vlm_object      = perception_res["bound_instance"]

        print(f"[VLM] vlm={action} spatial={spatial_action} "
              f"reason='{upgrade_reason}' | {vlm_ms}ms")

        if action == "none":
            return jsonify({
                "status":   "no_action",
                "user":     user_id,
                "action":   "none",
                "bound_to": "Unknown_Area",
                "reason":   "VLM returned no valid action",
            }), 200

        final_bound_label = memory.bind_and_update(
            user_id           = user_id,
            action            = spatial_action,
            est_pos           = est_pos,
            vlm_description   = vlm_desc,
            detected_items    = detected_items,
            all_items         = all_items,
            spatial_relations = spatial_rels,
            target_label      = vlm_object,
            room_name         = room_name,
        )
        print(f"[Bind] '{vlm_object}' -> '{final_bound_label}'")

        furniture_pos = None
        mongo_id      = None

        if final_bound_label and "Unknown" not in final_bound_label:
            furniture_doc = db.scene_snapshots.find_one(
                {"label": final_bound_label})
            if furniture_doc:
                furniture_pos = furniture_doc.get('pos')
                mongo_id      = furniture_doc.get('_id')

            spatial_text = " ".join(
                [f"{r['subject']} {r['relation']} {r['object']}"
                 for r in spatial_rels]
            ) if spatial_rels else ""

            vector_memory.add_memory(
                user_id           = user_id,
                action            = spatial_action,
                furniture_label   = final_bound_label,
                vlm_description   = (
                    f"{vlm_desc} {spatial_text}").strip(),
                detected_items    = detected_items,
                all_items         = all_items,
                spatial_relations = spatial_rels,
                furniture_pos     = furniture_pos,
                mongo_id          = mongo_id,
            )

            scene_items       = [
                i for i in all_items if i not in detected_items]
            all_dynamic_items = list(set(detected_items + scene_items))
            for item_label in all_dynamic_items:
                dyn_doc = db.dynamic_objects.find_one(
                    {"label": item_label.lower()})
                if dyn_doc:
                    vector_memory.upsert_dynamic_object(
                        label          = dyn_doc["label"],
                        room           = dyn_doc.get("room", room_name),
                        last_seen_on   = dyn_doc.get(
                            "last_seen_on", final_bound_label),
                        spatial_rel    = dyn_doc.get("spatial_rel", "near"),
                        furniture_pos  = dyn_doc.get(
                            "furniture_pos", furniture_pos),
                        seen_count     = dyn_doc.get("seen_count", 1),
                        interact_count = dyn_doc.get("interact_count", 0),
                        interacted_by  = dyn_doc.get("interacted_by", []),
                    )
        else:
            print("[Skip FAISS] bind failed")

        intent_prediction = {
            "trigger": False, "intent": "unknown", "confidence": 0.0}
        has_proposal = False

        try:
            prev_doc = db.activity_sequences.find_one(
                {"user": user_id}, sort=[("timestamp", -1)])
            prev_action_label = "Standing"
            if prev_doc and prev_doc.get("sequence"):
                seq = prev_doc["sequence"]
                if len(seq) >= 2:
                    prev_action_label = seq[-2].get("action", "Standing")

            intent_prediction = manifold_engine.predict_intent(
                user_id      = user_id,
                virtual_hour = virtual_hour,
                user_pos     = est_pos,
                prev_action  = prev_action_label,
            )

            if intent_prediction.get("trigger"):
                dynamic_results = vector_memory.search_dynamic(
                    spatial_action, top_k=5, user_filter=user_id)
                proposal_result = proposal_engine.evaluate(
                    user_id           = user_id,
                    intent_prediction = intent_prediction,
                    manifold_point_id = "",
                    user_pos          = est_pos,
                    dynamic_results   = dynamic_results,
                )
                has_proposal = proposal_result.get("has_proposal", False)

            habit_learner.check_and_update(user_id)

        except Exception as manifold_err:
            print(f"[Manifold] non-critical error: {manifold_err}")

        return jsonify({
            "status":            "Success",
            "user":              user_id,
            "action":            action,
            "spatial_action":    spatial_action,
            "upgrade_reason":    upgrade_reason,
            "zone_label":        zone_label,
            "bound_to":          final_bound_label,
            "interacting_items": detected_items,
            "all_items":         all_items,
            "spatial_relations": spatial_rels,
            "description":       vlm_desc,
            "estimated_pos":     est_pos,
            "furniture_pos":     furniture_pos,
            "vlm_inference_ms":  vlm_ms,
            "intent_prediction": intent_prediction,
            "has_proposal":      has_proposal,
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
        if not objects:
            return jsonify({"status": "empty"}), 200

        now  = datetime.datetime.utcnow()
        docs = []

        for obj in objects:
            label = obj.get('label', '').lower().strip()
            if not label:
                continue

            is_furniture = any(
                kw in label for kw in BASE_FURNITURE_KEYWORDS)
            if is_furniture:
                db.scene_snapshots.update_one(
                    {"label": label},
                    {"$set": {
                        "label":        label,
                        "pos":          [obj.get('x', 0), obj.get('z', 0)],
                        "x":            obj.get('x', 0),
                        "y":            obj.get('y', 0),
                        "z":            obj.get('z', 0),
                        "room":         obj.get('room', ''),
                        "source":       obj.get('source', 'sensor'),
                        "last_updated": now,
                        "is_static":    True,
                    }},
                    upsert=True,
                )

            docs.append({
                "label":       label,
                "x":           obj.get('x', 0),
                "y":           obj.get('y', 0),
                "z":           obj.get('z', 0),
                "room":        obj.get('room', ''),
                "image":       obj.get('image', ''),
                "source":      obj.get('source', 'sensor'),
                "held_by":     obj.get('held_by', ''),
                "processed":   False,
                "received_at": now,
            })

        if docs:
            db.raw_objects.insert_many(docs)

        print(f"[Scene] received {len(docs)} objects")
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

        now   = datetime.datetime.utcnow()
        count = 0

        for obj in objects:
            label  = obj.get('label', '').lower().strip()
            source = obj.get('source', 'sensor')
            if not label:
                continue

            if source == "unity_user":
                position = obj.get('position', [0, 0])
                forward  = obj.get('forward', [0, 0, 0])
                db.user_positions.update_one(
                    {"user_id": label},
                    {"$set": {
                        "user_id":    label,
                        "x":          float(position[0]),
                        "z":          float(position[1]),
                        "forward":    forward,
                        "activity":   obj.get('activity', ''),
                        "room":       obj.get('room', ''),
                        "updated_at": now,
                    }},
                    upsert=True,
                )
                count += 1
                continue

            room     = obj.get('room', '')
            position = obj.get('position', [0, 0])
            x        = float(position[0])
            z        = float(position[1])

            last_seen_on = _find_nearest_furniture(x, z, room)
            category     = _get_category_for_label(label)

            db.dynamic_objects.update_one(
                {"label": label},
                {
                    "$set": {
                        "room":         room,
                        "sensor_pos":   position,
                        "last_seen":    now,
                        "last_seen_on": last_seen_on,
                        "source":       source,
                        "category":     category,
                    },
                    "$inc":         {"seen_count": 1},
                    "$setOnInsert": {
                        "first_seen":  now,
                        "spatial_rel": "on",
                    },
                },
                upsert=True,
            )

            dyn_doc = db.dynamic_objects.find_one({"label": label})
            if dyn_doc:
                vector_memory.upsert_dynamic_object(
                    label          = label,
                    room           = room,
                    last_seen_on   = last_seen_on,
                    spatial_rel    = "on",
                    furniture_pos  = position,
                    seen_count     = dyn_doc.get("seen_count", 1),
                    interact_count = dyn_doc.get("interact_count", 0),
                    interacted_by  = dyn_doc.get("interacted_by", []),
                )
            count += 1

        unity_labels = [
            obj.get('label', '').lower().strip()
            for obj in objects
            if obj.get('source') == 'unity'
            and obj.get('label', '').strip()
        ]
        if unity_labels:
            stale = db.dynamic_objects.delete_many({
                "source": "unity",
                "label":  {"$nin": unity_labels},
            })
            if stale.deleted_count > 0:
                print(f"[DynamicSync] Removed "
                      f"{stale.deleted_count} stale objects")

        print(f"[DynamicSync] {count} objects updated")
        return jsonify({"status": "Success", "updated": count}), 200

    except Exception as e:
        import traceback
        print(f"[DynamicSync Error] {e}\n{traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500


@app.route('/set_virtual_hour', methods=['POST'])
def set_virtual_hour():
    try:
        data = request.get_json(force=True, silent=True) or {}
        hour = data.get('virtual_hour', -1)
        app.config['VIRTUAL_HOUR'] = float(hour)
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
            {"user": user_id,
             "action": {"$regex": action, "$options": "i"}},
            sort=[("weight", -1)],
        )
        weight = obs["weight"] if obs else 0

        similarity = 0.0
        try:
            results = vector_memory.search_habit(
                f"{user_id} {action}", user_id=user_id, top_k=1)
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
        return jsonify({"error": str(e)}), 500


@app.route('/query', methods=['POST'])
def query_habit():
    try:
        data       = request.get_json()
        user_query = data.get('query', '')
        user_id    = data.get('userID', None)

        if not user_query:
            return jsonify({"error": "Empty query"}), 400

        results = vector_memory.search_habit(
            user_query, user_id=user_id, top_k=5)

        for r in results:
            if r.get('instance') and r['instance'] != 'Unknown_Area':
                fresh = db.scene_snapshots.find_one(
                    {"label": r['instance']})
                if fresh:
                    r['furniture_pos']     = fresh.get('pos')
                    r['all_items']         = fresh.get('items', [])
                    r['spatial_relations'] = fresh.get(
                        'spatial_relations', [])

        top_habit  = vector_memory.get_top_habit(
            user_query, user_id=user_id, top_k=1)
        nav_target = None
        answer     = "I don't remember."

        if top_habit:
            fresh_doc = db.scene_snapshots.find_one(
                {"label": top_habit['instance']})
            if fresh_doc:
                nav_target = fresh_doc.get('pos')
            else:
                nav_target = top_habit.get('furniture_pos')

            interact_str = (
                ", ".join(top_habit.get('interacting_items', []))
                or "nothing specific")
            answer = (
                f"Based on observations, "
                f"{user_id or 'the user'} usually "
                f"{top_habit['actions'][0] if top_habit.get('actions') else 'does something'} "
                f"near {top_habit['instance']} "
                f"(seen {top_habit['count']} times), "
                f"typically interacting with: {interact_str}."
            )
        elif results:
            best       = results[0]
            nav_target = best.get('furniture_pos')
            answer = (f"I remember {best['user']} {best['action']} "
                      f"near {best['instance']}.")

        return jsonify({
            "status":           "Success",
            "answer":           answer,
            "nav_target":       nav_target,
            "top_habit":        top_habit,
            "semantic_results": results[:3],
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
            "timestamp":      datetime.datetime.utcnow(),
        })
        return jsonify({"status": "Success"}), 200
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

        result = interaction_engine.process(
            query     = query,
            user_id   = user_id,
            robot_pos = robot_pos,
            user_pos  = user_pos,
            room      = room,
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


@app.route('/track_position', methods=['POST'])
def track_position():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data"}), 400

        user_id   = data.get("userID", "Unknown")
        x         = float(data.get("x", 0))
        z         = float(data.get("z", 0))
        room_name = data.get("room_name", "")

        virtual_hour = app.config.get("VIRTUAL_HOUR", None)
        if virtual_hour is not None and float(virtual_hour) < 0:
            virtual_hour = None

        est_pos = {"x": x, "z": z}

        db.user_positions.update_one(
            {"user_id": user_id},
            {"$set": {
                "user_id":    user_id,
                "x":          x,
                "z":          z,
                "room":       room_name,
                "updated_at": datetime.datetime.utcnow(),
            }},
            upsert=True,
        )

        return jsonify({"status": "ok"}), 200

    except Exception as e:
        import traceback
        print(f"[TrackPosition Error] {e}\n{traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500


@app.route('/manifold_status', methods=['GET'])
def manifold_status():
    try:
        users  = ["User_Mom", "User_Dad"]
        status = {}
        for uid in users:
            n_samples = manifold_engine.get_training_count(uid)
            has_model = manifold_engine._get_model(uid) is not None
            status[uid] = {
                "training_samples": n_samples,
                "model_ready":      has_model,
            }
        return jsonify({"status": "ok", "manifold": status}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/manifold_train', methods=['POST'])
def manifold_train():
    try:
        data    = request.get_json(force=True, silent=True) or {}
        user_id = data.get("user_id", "")
        if not user_id:
            for uid in ["User_Mom", "User_Dad"]:
                threading.Thread(
                    target=manifold_engine.train_model,
                    args=(uid,), daemon=True).start()
            return jsonify({"status": "training started for all users"}), 200
        threading.Thread(
            target=manifold_engine.train_model,
            args=(user_id,), daemon=True).start()
        return jsonify({"status": "training started", "user_id": user_id}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


from modules.interaction import InteractionEngine
from modules.training_exporter import TrainingExporter
from modules.habit_learner import HabitLearner

interaction_engine = InteractionEngine(
    mongo_client  = mongo_client,
    vector_memory = vector_memory,
    ollama_url    = CONFIG.OLLAMA_URL,
    model_name    = CONFIG.LLM_MODEL,
)
training_exporter = TrainingExporter(mongo_client)

habit_learner = HabitLearner(
    db_client     = mongo_client,
    skill_manager = interaction_engine.skill_manager,
)


@app.route('/habit_feedback', methods=['POST'])
def habit_feedback():
    try:
        data    = request.get_json()
        user_id = data.get("user_id", "")
        result  = data.get("result", "")
        intent  = data.get("intent", "")
        item    = data.get("item", "")

        if result == "rejected":
            habit_learner.handle_rejection(user_id, intent, item)
        elif result == "accepted":
            habit_learner.handle_acceptance(user_id, intent, item)

        return jsonify({"status": "ok"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/habit_check', methods=['POST'])
def habit_check():
    try:
        data    = request.get_json()
        user_id = data.get("user_id", "")
        if not user_id:
            return jsonify({"error": "user_id required"}), 400
        habit_learner.check_and_update(user_id)
        return jsonify({"status": "ok", "user_id": user_id}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    host = getattr(CONFIG, 'FLASK_HOST', '0.0.0.0')
    port = int(getattr(CONFIG, 'FLASK_PORT', 5000))
    print(f"\nRobot Brain Server on {host}:{port}")
    print(f"  VLM model : {CONFIG.VLM_MODEL}")
    print(f"  LLM model : {CONFIG.LLM_MODEL}")
    threading.Timer(86400, nightly_maintenance).start()
    app.run(host=host, port=port, debug=False)