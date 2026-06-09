import re
import logging
import os
import json
import math
import uuid
import atexit
import base64
import datetime
import threading
import queue as _queue

import numpy as np
import cv2
from flask import Flask, request, jsonify, Response, stream_with_context
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer

from config import Config

from modules.scene_engine      import SceneEngine
from modules.perception_engine import PerceptionEngine
from modules.habit_engine      import HabitEngine
from modules.manifold_engine   import ManifoldEngine
from modules.saycan_engine     import SayCanEngine
from modules.service_proposal  import ServiceProposalEngine
from modules.skill_manager     import SkillManager
from modules.memory_vector     import VectorMemory
from modules.interaction       import InteractionEngine
from modules.entropy_monitor   import EntropyMonitor
from modules.classifier        import (ObjectClassifier,
                                        BASE_FURNITURE_KEYWORDS,
                                        OBJECT_CATEGORIES)

try:
    import yaml as _yaml
    _YAML_OK = True
except ImportError:
    _YAML_OK = False


def _load_yaml(path: str) -> dict:
    if _YAML_OK and os.path.exists(path):
        with open(path) as f:
            return _yaml.safe_load(f) or {}
    return {}


app = Flask(__name__)
CONFIG = Config

werkzeug_logger = logging.getLogger("werkzeug")
werkzeug_logger.setLevel(logging.WARNING)
print(f"System init: device={CONFIG.DEVICE} "
      f"| LLM={CONFIG.LLM_MODEL} | VLM={CONFIG.VLM_MODEL}")

sbert_model = SentenceTransformer('all-MiniLM-L6-v2', device=CONFIG.DEVICE)
print("SBERT loaded on", CONFIG.DEVICE)

mongo_client = MongoClient(CONFIG.MONGO_URI)
db = mongo_client[CONFIG.DB_NAME]

try:
    db.scene_snapshots.create_index([("pos", "2d")])
    db.observation_logs.create_index(
        [("last_seen", 1)],
        expireAfterSeconds=14 * 86400,
        name="observation_ttl_14d")
    print("[MongoDB] indexes ready")
except Exception as e:
    print(f"[MongoDB] index notice: {e}")

_ontology = _load_yaml("config/robot_ontology.yaml")
_beh_cfg = _load_yaml("config/behavior_config.yaml")
_sys_cfg = _load_yaml("config/system_config.yaml")

behavior_labels = _beh_cfg.get("behavior_labels", [
    "Drinking", "SittingDrink", "Sitting", "Eating", "Cooking", "Opening",
    "Laying", "Watching", "Reading", "Cleaning", "PhoneUse",
    "Typing", "StandUp", "PickingUp", "PuttingDown", "Standing", "Walking",
])

scene_engine = SceneEngine(
    db=db,
    ollama_url=CONFIG.OLLAMA_URL,
    sbert_model=sbert_model,
    ontology=_ontology,
    system_cfg=_sys_cfg,
    behavior_labels=behavior_labels,
)

manifold_engine = ManifoldEngine(db=db, sbert_model=sbert_model)

vector_memory = VectorMemory(device=CONFIG.DEVICE)
vector_memory.sync_from_mongo(db.dynamic_objects)

perception = PerceptionEngine(
    db=db,
    ollama_url=CONFIG.OLLAMA_URL,
    vlm_model=CONFIG.VLM_MODEL,
    sbert_model=sbert_model,
    scene_engine=scene_engine,
)
perception.manifold_engine = manifold_engine

skill_manager = SkillManager(mongo_client, CONFIG.DB_NAME)

habit_engine = HabitEngine(
    db=db,
    skill_manager=skill_manager,
    manifold_engine=manifold_engine,
    vector_memory=vector_memory,
)

saycan_engine = SayCanEngine(
    db=db,
    manifold_engine=manifold_engine,
    ollama_url=CONFIG.OLLAMA_URL,
    llm_model=CONFIG.LLM_MODEL,
    sbert_model=sbert_model,
    vector_memory=vector_memory,
)

proposal_engine = ServiceProposalEngine(
    db=db,
    ollama_url=CONFIG.OLLAMA_URL,
    llm_model=CONFIG.LLM_MODEL,
)

interaction_engine = InteractionEngine(
    mongo_client=mongo_client,
    vector_memory=vector_memory,
    ollama_url=CONFIG.OLLAMA_URL,
    model_name=CONFIG.LLM_MODEL,
    saycan_engine=saycan_engine,
)

entropy_monitor = EntropyMonitor()

classifier = ObjectClassifier(db)
classifier.start()

atexit.register(perception.shutdown)

_vlm_lock = threading.Lock()
_predict_queue = _queue.Queue()

_gt_cache = {}
_gt_cache_lock = threading.Lock()

_robot_state = {
    "nav_target": None,
    "nav_label": "",
    "last_answer": "",
    "highlight": "",
}


def preview_images(image_list, source_nodes, hint_user_id, activity):
    save_dir = "debug_images"
    os.makedirs(save_dir, exist_ok=True)
    for i, img_b64 in enumerate(image_list):
        try:
            img_clean = img_b64.split(',')[1] if ',' in img_b64 else img_b64
            nparr = np.frombuffer(base64.b64decode(img_clean), np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if frame is None:
                continue
            ts = datetime.datetime.now().strftime("%H%M%S")
            node_name = source_nodes[i] if i < len(source_nodes) else f"img_{i}"
            cv2.imwrite(f"{save_dir}/{ts}_{hint_user_id}_{activity}_{node_name}.jpg", frame)
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


_FURNITURE_BLACKLIST = {
    "floor", "ceiling", "wall", "ground", "wooden floor",
    "tile floor", "carpet", "concrete floor", "baseboard",
    "white wall", "window", "door",
}


def _find_nearest_furniture(x: float, z: float, room: str, max_dist: float = 3.0) -> str:
    query = {}
    if room:
        query["room"] = {"$regex": room, "$options": "i"}
    furniture_docs = list(db.scene_snapshots.find(query, {"label": 1, "pos": 1}))
    if not furniture_docs:
        furniture_docs = list(db.scene_snapshots.find({}, {"label": 1, "pos": 1}))

    best_label = "Unknown_Area"
    best_dist = float("inf")

    for doc in furniture_docs:
        label = doc.get("label", "").lower().strip()
        if label in _FURNITURE_BLACKLIST:
            continue
        pos = doc.get("pos")
        if not isinstance(pos, list) or len(pos) < 2:
            continue
        dist = math.sqrt((x - pos[0]) ** 2 + (z - pos[1]) ** 2)
        if dist < best_dist:
            best_dist = dist
            best_label = doc["label"]

    return best_label if best_dist <= max_dist else "Unknown_Area"


def _get_category_for_label(label: str) -> str:
    label_l = label.lower().strip()
    for cat, keywords in OBJECT_CATEGORIES.items():
        if label_l in keywords or any(kw in label_l for kw in keywords):
            return cat
    return "other"


def _stream_ollama(system: str, user_prompt: str):
    import requests as _req
    try:
        resp = _req.post(
            f"{CONFIG.OLLAMA_URL}/api/chat",
            json={
                "model": CONFIG.LLM_MODEL,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_prompt},
                ],
                "stream": True,
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
                    yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"
                if chunk.get("done"):
                    break
            except json.JSONDecodeError:
                continue
    except Exception as e:
        print(f"[Stream Ollama] error: {e}")
        yield f"data: {json.dumps({'type': 'token', 'content': 'Sorry, an error occurred.'})}\n\n"


def nightly_maintenance():
    print("[Maintenance] running nightly tasks...")
    try:
        db.observation_logs.update_many(
            {}, {"$mul": {"weight": getattr(CONFIG, 'HABIT_DECAY_FACTOR', 0.95)}})
        db.observation_logs.delete_many(
            {"weight": {"$lt": getattr(CONFIG, 'HABIT_MIN_WEIGHT', 1.0)}})
        print("[Maintenance] habit decay done")

        if interaction_engine._has_skill_manager:
            sm = interaction_engine.skill_manager
            for doc in db.user_skills.find({}, {"user_id": 1}):
                try:
                    sm.nightly_refactor(doc["user_id"])
                except Exception as e:
                    print(f"[Maintenance] refactor failed: {e}")

        print("[Maintenance] done")
    except Exception as e:
        print(f"[Maintenance] error: {e}")

    threading.Timer(86400, nightly_maintenance).start()


@app.route("/ready", methods=["GET"])
def ready():
    status = scene_engine.status()
    return jsonify({
        "ready": status["ready"],
        "zone_count": status["zone_count"],
        "affinity_count": status["affinity_count"],
        "zones": status["zones"],
    }), 200


@app.route("/experiment_done", methods=["POST"])
def experiment_done():
    def _final_train():
        for uid in ["User_Mom", "User_Dad"]:
            n = db.manifold_training_data.count_documents({"user_id": uid})
            if n >= 20:
                print(f"[ExperimentDone] Final MLP train: {uid} ({n} samples)")
                manifold_engine.train_model(uid)
    threading.Thread(target=_final_train, daemon=True).start()
    return jsonify({"status": "ok", "message": "Final training triggered"}), 200


@app.route('/nav_target', methods=['GET'])
def get_nav_target():
    return jsonify({
        "nav_target": _robot_state["nav_target"],
        "nav_label": _robot_state["nav_label"],
    })


@app.route('/highlight', methods=['GET'])
def get_highlight():
    return jsonify({"label": _robot_state["highlight"]})


@app.route('/last_answer', methods=['GET'])
def get_last_answer():
    return jsonify({"answer": _robot_state["last_answer"]})


@app.route('/interact/stream', methods=['POST'])
def interact_stream():
    data = request.get_json()
    query = data.get('query', '')
    user_id = data.get('userID', 'Unknown')
    room = data.get('room', '')

    if not query:
        return jsonify({"error": "Empty query"}), 400

    def generate():
        intent = interaction_engine._classify_intent(query)
        yield f"data: {json.dumps({'type': 'intent', 'intent': intent})}\n\n"

        if intent == "interrupt":
            answer = "Understood, stopping now."
            _robot_state["last_answer"] = answer
            _robot_state["nav_target"] = None
            _robot_state["nav_label"] = ""
            _robot_state["highlight"] = ""
            yield f"data: {json.dumps({'type': 'token', 'content': answer})}\n\n"
            yield f"data: {json.dumps({'type': 'done', 'nav_target': None, 'nav_label': None, 'confidence': 1.0, 'intent_type': 'interrupt', 'options': [], 'is_personalized': False})}\n\n"
            return

        if intent == "chat":
            from modules.interaction import CHAT_SYSTEM
            chat_buf = ""
            for ev in _stream_ollama(CHAT_SYSTEM, f'{user_id} said: "{query}"'):
                yield ev
                try:
                    parsed = json.loads(ev.replace("data: ", "").strip())
                    if parsed.get("type") == "token":
                        chat_buf += parsed.get("content", "")
                except Exception:
                    pass
            chat_answer = chat_buf.strip().strip('"').strip("'")
            _robot_state["last_answer"] = chat_answer
            _robot_state["nav_target"] = None
            _robot_state["nav_label"] = ""
            _robot_state["highlight"] = ""
            interaction_engine._schedule_skill_update(
                user_id=user_id, query=query,
                answer=chat_answer, env_snapshot="", rec_items=[])
            yield f"data: {json.dumps({'type': 'done', 'nav_target': None, 'nav_label': None, 'confidence': 1.0, 'intent_type': 'chat', 'options': [], 'is_personalized': False})}\n\n"
            return

        if intent == "locate":
            result = interaction_engine._query_response(query, user_id, room)
            answer = result.get("answer", "")
            for char in answer:
                yield f"data: {json.dumps({'type': 'token', 'content': char})}\n\n"
            _robot_state["last_answer"] = answer
            _robot_state["nav_target"] = result.get("nav_target")
            _robot_state["nav_label"] = result.get("nav_label", "")
            _robot_state["highlight"] = result.get("nav_label", "")
            yield f"data: {json.dumps({'type': 'done', 'nav_target': result.get('nav_target'), 'nav_label': result.get('nav_label'), 'confidence': result.get('confidence', 0.85), 'intent_type': 'query', 'options': result.get('options', []), 'is_personalized': False})}\n\n"
            return

        result = interaction_engine.process(query=query, user_id=user_id, room=room)

        answer = result.get("answer", "")
        nav_target = result.get("nav_target")
        nav_label = result.get("nav_label", "unknown")

        for char in answer:
            yield f"data: {json.dumps({'type': 'token', 'content': char})}\n\n"

        _robot_state["last_answer"] = answer
        _robot_state["nav_target"] = nav_target
        _robot_state["nav_label"] = nav_label if nav_label != "unknown" else ""
        _robot_state["highlight"] = nav_label if nav_label != "unknown" else ""

        yield f"data: {json.dumps({'type': 'done', 'nav_target': nav_target, 'nav_label': nav_label, 'confidence': result.get('confidence', 0.85), 'intent_type': result.get('intent_type', 'need'), 'options': result.get('options', []), 'is_personalized': result.get('is_personalized', False)})}\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype='text/event-stream',
        headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'},
    )


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data received"}), 400

    episode_id = data.get("episode_id") or str(uuid.uuid4())

    existing = db.eval_logs.find_one({"episode_id": episode_id})
    if existing:
        return jsonify({"status": "ok", "episode_id": episode_id, "cached": True}), 200

    t_capture = data.get("t_capture", "")
    ground_truth = data.get("activity", "")
    if t_capture and ground_truth:
        with _gt_cache_lock:
            _gt_cache[t_capture] = ground_truth

    _predict_queue.put((episode_id, data))
    return jsonify({"status": "queued", "episode_id": episode_id}), 200


def _predict_worker():
    while True:
        try:
            episode_id, data = _predict_queue.get(timeout=1)
            try:
                _process_predict(episode_id, data)
            except Exception as e:
                import traceback
                print(f"[PredictWorker] {e}\n{traceback.format_exc()}")
            finally:
                _predict_queue.task_done()
        except _queue.Empty:
            continue


def _process_predict(episode_id, data):
    import time
    try:
        image_list = data.get('image_list', [])
        hint_user_id = data.get('userID', 'Unknown_User')
        activity = data.get('activity', '')
        user_pos_raw = data.get('user_pos')
        user_fwd_raw = data.get('user_forward')
        source_nodes = data.get('source_nodes', [])
        room_name = data.get('room_name', '')
        virtual_hour = data.get('virtual_hour')
        virtual_day = data.get('virtual_day', '')
        t_capture = data.get('t_capture', '')

        if not room_name and source_nodes:
            first_node = source_nodes[0].split('_b')[0]
            camIdx = first_node.rfind('_Cam')
            room_name = first_node[:camIdx] if camIdx > 0 else first_node

        if not image_list:
            return

        preview_images(image_list, source_nodes, hint_user_id, activity)

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

        if est_forward is None and hint_user_id:
            _pos_doc = db.user_positions.find_one({"user_id": hint_user_id})
            if _pos_doc and _pos_doc.get("forward"):
                _fwd = _pos_doc["forward"]
                if isinstance(_fwd, list) and len(_fwd) >= 3:
                    est_forward = {"x": _fwd[0], "y": 0.0, "z": _fwd[2]}
                elif isinstance(_fwd, dict):
                    est_forward = {"x": float(_fwd.get("x", 0)),
                                   "y": 0.0,
                                   "z": float(_fwd.get("z", 0))}

        if est_pos and hint_user_id:
            db.user_positions.update_one(
                {"user_id": hint_user_id},
                {"$set": {
                    "user_id": hint_user_id,
                    "x": est_pos["x"],
                    "z": est_pos["z"],
                    "room": room_name,
                    "forward": est_forward,
                    "updated_at": datetime.datetime.utcnow(),
                }},
                upsert=True,
            )

        data['room_name'] = room_name
        data['user_forward'] = est_forward
       
        _wait_for_scene(max_wait=12.0)

        with _vlm_lock:
            t0 = time.time()
            result = perception.analyze_action_burst(data)
            vlm_ms = int((time.time() - t0) * 1000)

        _user_id = result.get("user", "") or result.get("user_id", "")
        _user_pos = result.get("user_pos") or {}
        _exp_mode = result.get("experiment_mode", "habit")

        _activity_votes = [result.get("action", "")] if result.get("action") else []
        _body_votes = [result["result"].get("_body_position", "")]
        _held_votes = [result["result"].get("_held_object", "none")]
        _entropy_info = entropy_monitor.analyze(
            _user_id, _activity_votes, _body_votes, _held_votes)

        _raw_action = result.get("spatial_action") or result.get("action", "Unknown")
        _action = _raw_action
        _zone = result.get("zone_label") or result.get("zone_name") or ""

        if _action and _action not in ("none", "", "Unknown") and _zone:
            habit_engine.record(
                user_id=_user_id,
                action=_action,
                zone_name=_zone,
                pos=[_user_pos.get("x", 0), _user_pos.get("z", 0)],
                virtual_hour=result.get("virtual_hour", 12.0),
                time_slot=result.get("time_slot", ""),
                interacting_items=result.get("items", []),
                raw_desc=str(result.get("result", "")),
                room=result.get("room", ""),
                instance=_zone,
                spatial_relations=result.get("spatial_relations", {}),
                experiment_mode=_exp_mode,
            )

            if _zone and _action not in ("none", "", "Unknown"):
                scene_engine.update_user_affinity(
                    user_id=_user_id,
                    zone_name=_zone,
                    action=_action,
                    room=result.get("room", ""),
                    virtual_day=virtual_day,
                )

        user_id = result["user"]
        action = result["action"]
        spatial_action = _action
        upgrade_reason = result.get("upgrade_reason", "")
        zone_label = result.get("zone_label", "")
        detected_items = result.get("items", [])
        all_items = result.get("all_items", [])
        spatial_rels = result.get("spatial_relations", [])
        vlm_desc = result["result"].get("context", "Observed behavior.")
        vlm_object = result["bound_instance"]

        print(f"[VLM] vlm={action} spatial={spatial_action} "
              f"entropy={_entropy_info['overall_entropy']:.2f} | {vlm_ms}ms")

        if action == "none":
            return

        final_bound_label = vlm_object if (
            vlm_object and "Unknown" not in str(vlm_object)
        ) else ""

        if not final_bound_label and est_pos:
            import math as _math
            best_doc = None
            best_dist = float("inf")
            query_r = {"room": {"$regex": room_name, "$options": "i"}} if room_name else {}
            for doc in db.scene_snapshots.find(query_r, {"label": 1, "pos": 1}):
                pos = doc.get("pos")
                if not isinstance(pos, list) or len(pos) < 2:
                    continue
                dist = _math.sqrt(
                    (est_pos["x"] - pos[0]) ** 2 +
                    (est_pos["z"] - pos[1]) ** 2
                )
                if dist < best_dist:
                    best_dist = dist
                    best_doc = doc
            if best_doc and best_dist <= 5.0:
                final_bound_label = best_doc["label"]

        if not final_bound_label:
            final_bound_label = "Unknown_Area"

        try:
            today = datetime.datetime.utcnow().strftime("%Y-%m-%d")
            db.activity_sequences.update_one(
                {"user": user_id, "date": today},
                {
                    "$push": {
                        "sequence": {
                            "action": spatial_action,
                            "instance": final_bound_label,
                            "timestamp": datetime.datetime.utcnow(),
                        }
                    },
                    "$setOnInsert": {"user": user_id, "date": today},
                },
                upsert=True,
            )
        except Exception as seq_e:
            print(f"[Sequence] {seq_e}")

        print(f"[Bind] '{vlm_object}' -> '{final_bound_label}'")

        furniture_pos = None
        mongo_id = None

        if final_bound_label and "Unknown" not in final_bound_label:
            furniture_doc = db.scene_snapshots.find_one({"label": final_bound_label})
            if furniture_doc:
                furniture_pos = furniture_doc.get('pos')
                mongo_id = furniture_doc.get('_id')

            spatial_text = " ".join(
                [f"{r['subject']} {r['relation']} {r['object']}"
                 for r in spatial_rels]) if spatial_rels else ""

            vector_memory.add_memory(
                user_id=user_id,
                action=spatial_action,
                furniture_label=final_bound_label,
                vlm_description=f"{vlm_desc} {spatial_text}".strip(),
                detected_items=detected_items,
                all_items=all_items,
                spatial_relations=spatial_rels,
                furniture_pos=furniture_pos,
                mongo_id=mongo_id,
            )

            for item_label in list(set(detected_items + all_items)):
                dyn_doc = db.dynamic_objects.find_one({"label": item_label.lower()})
                if dyn_doc:
                    vector_memory.upsert_dynamic_object(
                        label=dyn_doc["label"],
                        room=dyn_doc.get("room", room_name),
                        last_seen_on=dyn_doc.get("last_seen_on", final_bound_label),
                        spatial_rel=dyn_doc.get("spatial_rel", "near"),
                        furniture_pos=dyn_doc.get("furniture_pos", furniture_pos),
                        seen_count=dyn_doc.get("seen_count", 1),
                        interact_count=dyn_doc.get("interact_count", 0),
                        interacted_by=dyn_doc.get("interacted_by", []),
                    )

        try:
            _prev_seq_doc = db.activity_sequences.find_one(
                {"user": user_id}, sort=[("date", -1)])
            _real_prev = "Standing"
            if _prev_seq_doc and len(_prev_seq_doc.get("sequence", [])) >= 2:
                _real_prev = _prev_seq_doc["sequence"][-2].get("action", "Standing")

            intent_prediction = manifold_engine.predict_intent(
                user_id=user_id,
                virtual_hour=virtual_hour,
                user_pos=est_pos,
                prev_action=_real_prev,
            )

            if intent_prediction.get("trigger"):
                dynamic_results = vector_memory.search_dynamic(
                    spatial_action, top_k=5, user_filter=user_id)
                proposal_engine.evaluate(
                    user_id=user_id,
                    intent_prediction=intent_prediction,
                    manifold_point_id="",
                    user_pos=est_pos,
                    dynamic_results=dynamic_results,
                )

            habit_engine._check_and_update_skill(user_id)

            NO_WEIGHT_ACTIONS = set(_beh_cfg.get("no_weight_actions",
                ["PickingUp", "PuttingDown", "Walking", "Standing"]))
            _record_action = spatial_action if spatial_action not in ("Unknown", "none", "") \
                             else activity or ""

            if (_record_action and
                    _record_action not in NO_WEIGHT_ACTIONS and
                    _record_action != "Unknown" and
                    manifold_engine is not None and
                    _exp_mode != "recognition" and
                    result["result"].get("_vlm_confidence", 0) >= 0.20):

                _prev_for_mlp = "Standing"
                _seq_doc_mlp = db.activity_sequences.find_one(
                    {"user": user_id}, sort=[("date", -1)])
                if _seq_doc_mlp and len(_seq_doc_mlp.get("sequence", [])) >= 2:
                    _prev_for_mlp = _seq_doc_mlp["sequence"][-2].get("action", "Standing")

                try:
                    manifold_engine.record_training_sample(
                        user_id=user_id,
                        virtual_hour=virtual_hour,
                        user_pos=est_pos,
                        prev_action=_prev_for_mlp,
                        current_action=_record_action,
                    )
                except Exception:
                    pass

        except Exception as e:
            print(f"[Manifold] non-critical error: {e}")

        db.eval_logs.update_one(
            {"episode_id": episode_id},
            {"$set": {
                "status": "done",
                "entropy": _entropy_info["overall_entropy"],
                "forward_x": est_forward.get("x", 0) if est_forward else None,
                "forward_z": est_forward.get("z", 0) if est_forward else None,
                "vlm_ms": vlm_ms,
            }},
        )

        print(f"[Predict] done | {user_id} | {action} -> {spatial_action} | {vlm_ms}ms")

    except Exception as e:
        import traceback
        print(f"[PredictWorker] {e}\n{traceback.format_exc()}")


@app.route('/scene', methods=['POST'])
def handle_scene():
    try:
        data = request.get_json()
        objects = data.get('objects', [])
        if not objects:
            return jsonify({"status": "empty"}), 200

        now = datetime.datetime.utcnow()
        docs = []

        for obj in objects:
            label = obj.get('label', '').lower().strip()
            if not label:
                continue

            if any(kw in label for kw in BASE_FURNITURE_KEYWORDS):
                db.scene_snapshots.update_one(
                    {"label": label},
                    {"$set": {
                        "label": label,
                        "pos": [obj.get('x', 0), obj.get('z', 0)],
                        "x": obj.get('x', 0),
                        "y": obj.get('y', 0),
                        "z": obj.get('z', 0),
                        "room": obj.get('room', ''),
                        "source": obj.get('source', 'sensor'),
                        "last_updated": now,
                        "is_static": True,
                    }},
                    upsert=True,
                )

            docs.append({
                "label": label,
                "x": obj.get('x', 0),
                "y": obj.get('y', 0),
                "z": obj.get('z', 0),
                "room": obj.get('room', ''),
                "source": obj.get('source', 'sensor'),
                "held_by": obj.get('held_by', ''),
                "processed": False,
                "received_at": now,
            })

        if docs:
            db.raw_objects.insert_many(docs)

        scene_engine.build()
        print(f"[Scene] received {len(docs)} objects")
        return jsonify({"status": "Success", "received": len(docs)}), 200

    except Exception as e:
        print(f"[Scene Error] {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/dynamic_sync', methods=['POST'])
def dynamic_sync():
    try:
        data = request.get_json()
        objects = data.get('objects', [])
        if not objects:
            return jsonify({"status": "empty"}), 200

        timestamp_str = data.get('timestamp', '')
        if timestamp_str:
            try:
                now = datetime.datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            except:
                now = datetime.datetime.utcnow()
        else:
            now = datetime.datetime.utcnow()

        count = 0

        for obj in objects:
            label = obj.get('label', '').lower().strip()
            source = obj.get('source', 'sensor')
            if not label:
                continue

            if source == "unity_user":
                position = obj.get('position', [0, 0])
                forward = obj.get('forward', [0, 0, 0])
                db.user_positions.update_one(
                    {"user_id": label},
                    {"$set": {
                        "user_id": label,
                        "x": float(position[0]),
                        "z": float(position[1]),
                        "forward": forward,
                        "activity": obj.get('activity', ''),
                        "room": obj.get('room', ''),
                        "updated_at": now,
                    }},
                    upsert=True,
                )
                count += 1
                continue

            room = obj.get('room', '')
            position = obj.get('position', [0, 0])
            x = float(position[0])
            z = float(position[1])

            held_by = obj.get('held_by', '')
            last_seen_on = _find_nearest_furniture(x, z, room)
            category = _get_category_for_label(label)

            set_fields = {
                "room": room,
                "sensor_pos": position,
                "last_seen": now,
                "status": "active",
                "status_since": now,
                "source": source,
                "category": category,
            }

            if held_by:
                set_fields["held_by"] = held_by
               
                existing = db.dynamic_objects.find_one(
                    {"label": label}, {"held_by": 1})
                if not existing or existing.get("held_by") != held_by:
                    set_fields["held_since"] = now
            else:
                set_fields["last_seen_on"] = last_seen_on

            db.dynamic_objects.update_one(
                {"label": label},
                {
                    "$set": set_fields,
                    "$inc": {"seen_count": 1},
                    "$setOnInsert": {
                        "first_seen": now,
                        "spatial_rel": "on",
                    },
                },
                upsert=True,
            )

            dyn_doc = db.dynamic_objects.find_one({"label": label})
            if dyn_doc:
                vector_memory.upsert_dynamic_object(
                    label=label,
                    room=room,
                    last_seen_on=last_seen_on,
                    spatial_rel="on",
                    furniture_pos=position,
                    seen_count=dyn_doc.get("seen_count", 1),
                    interact_count=dyn_doc.get("interact_count", 0),
                    interacted_by=dyn_doc.get("interacted_by", []),
                )
            count += 1

        unity_labels = [
            obj.get('label', '').lower().strip()
            for obj in objects
            if obj.get('source') == 'unity' and obj.get('label', '').strip()
        ]
        if unity_labels:
            stale = db.dynamic_objects.delete_many({
                "source": "unity",
                "label": {"$nin": unity_labels},
            })
            if stale.deleted_count > 0:
                print(f"[DynamicSync] Removed {stale.deleted_count} stale objects")

        
        return jsonify({"status": "Success", "updated": count}), 200

    except Exception as e:
        import traceback
        print(f"[DynamicSync Error] {e}\n{traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500


@app.route('/set_device_state', methods=['POST'])
def set_device_state():
    data = request.get_json()
    label = data.get('label', '')
    state = data.get('state', 'off')
    timestamp_str = data.get('timestamp', '')
    
    if timestamp_str:
        try:
            now = datetime.datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        except:
            now = datetime.datetime.utcnow()
    else:
        now = datetime.datetime.utcnow()
    
    if label:
        db.device_states.update_one(
            {'label': label},
            {'$set': {'state': state,
                      'updated_at': now}},
            upsert=True
        )
    return jsonify({'status': 'ok'}), 200


@app.route('/set_held_object', methods=['POST'])
def set_held_object():
    data = request.get_json()
    user_id = data.get('user_id', '')
    held_object = data.get('held_object', 'none')
    timestamp_str = data.get('timestamp', '')
    
    if timestamp_str:
        try:
            now = datetime.datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        except:
            now = datetime.datetime.utcnow()
    else:
        now = datetime.datetime.utcnow()
    
    if not user_id:
        return jsonify({'status': 'error'}), 400
    
    if held_object == 'none':
        db.dynamic_objects.update_many(
            {"held_by": user_id},
            {"$set": {"held_by": "", "held_since": None, "last_seen": now}}
        )
        print(f"[SetHeld] {user_id} cleared all held objects at {now}")
    else:
        existing = db.dynamic_objects.find_one({"label": held_object.lower()})
        existing_since = existing.get("held_since") if existing else None
        
        if existing_since is None or now > existing_since:
            db.dynamic_objects.update_one(
                {"label": held_object.lower()},
                {"$set": {"held_by": user_id, "held_since": now, "last_seen": now}},
                upsert=True
            )
            print(f"[SetHeld] {user_id} picked up {held_object} at {now}")
        else:
            print(f"[SetHeld] Ignored {held_object} at {now}, existing_since={existing_since}")
    
    return jsonify({'status': 'ok'}), 200

@app.route('/set_virtual_hour', methods=['POST'])
def set_virtual_hour():
    try:
        data = request.get_json(force=True, silent=True) or {}
        hour = data.get('virtual_hour', -1)
        app.config['VIRTUAL_HOUR'] = float(hour)
        db.system_config.update_one(
            {"key": "virtual_hour"},
            {"$set": {"value": float(hour)}},
            upsert=True)
        return jsonify({"status": "ok", "virtual_hour": hour}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/exp_checkpoint', methods=['GET', 'POST'])
def exp_checkpoint():
    try:
        if request.method == 'POST':
            body = request.get_json(force=True, silent=True) or {}
            episode = body.get('episode', 0)
            user_id = body.get('user_id', 'User_Mom')
            action = body.get('action', 'Drink')
            experiment = 'experiment2'
        else:
            experiment = request.args.get('experiment', 'experiment2')
            episode = request.args.get('step', 0, type=int)
            user_id = request.args.get('user', 'User_Mom')
            action = request.args.get('action', 'Drink')

        obs = db.observation_logs.find_one(
            {"user": user_id, "action": {"$regex": action, "$options": "i"}},
            sort=[("weight", -1)])
        weight = obs["weight"] if obs else 0

        similarity = 0.0
        try:
            results = vector_memory.search_habit(
                f"{user_id} {action}", user_id=user_id, top_k=1)
            if results:
                similarity = float(results[0].get("similarity", 0.0))
        except Exception as fe:
            print(f"[Checkpoint] FAISS query skipped: {fe}")

        db["exp_checkpoint_logs"].insert_one({
            "experiment": experiment,
            "episode": episode,
            "user_id": user_id,
            "action": action,
            "weight": weight,
            "similarity": round(similarity, 4),
            "timestamp": datetime.datetime.utcnow(),
        })

        return jsonify({
            "status": "ok",
            "experiment": experiment,
            "episode": episode,
            "user_id": user_id,
            "action": action,
            "weight": weight,
            "similarity": round(similarity, 4),
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/query', methods=['POST'])
def query_habit():
    try:
        data = request.get_json()
        user_query = data.get('query', '')
        user_id = data.get('userID', None)

        if not user_query:
            return jsonify({"error": "Empty query"}), 400

        results = vector_memory.search_habit(user_query, user_id=user_id, top_k=5)
        top_habit = vector_memory.get_top_habit(user_query, user_id=user_id, top_k=1)

        nav_target = None
        answer = "I don't remember."

        if top_habit:
            fresh_doc = db.scene_snapshots.find_one({"label": top_habit['instance']})
            nav_target = fresh_doc.get('pos') if fresh_doc else top_habit.get('furniture_pos')
            interact_str = ", ".join(top_habit.get('interacting_items', [])) or "nothing specific"
            answer = (
                f"Based on observations, "
                f"{user_id or 'the user'} usually "
                f"{top_habit['actions'][0] if top_habit.get('actions') else 'does something'} "
                f"near {top_habit['instance']} "
                f"(seen {top_habit['count']} times), "
                f"typically interacting with: {interact_str}."
            )
        elif results:
            best = results[0]
            nav_target = best.get('furniture_pos')
            answer = f"I remember {best['user']} {best['action']} near {best['instance']}."

        return jsonify({
            "status": "Success",
            "answer": answer,
            "nav_target": nav_target,
            "top_habit": top_habit,
            "semantic_results": results[:3],
        }), 200

    except Exception as e:
        import traceback
        print(f"[Query Error] {e}\n{traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500


@app.route('/interact', methods=['POST'])
def interact():
    try:
        data = request.get_json()
        query = data.get('query', '')
        user_id = data.get('userID', 'Unknown')
        robot_pos = data.get('robot_pos')
        user_pos = data.get('user_pos')
        room = data.get('room', '')

        if not query:
            return jsonify({"error": "Empty query"}), 400

        result = interaction_engine.process(
            query=query, user_id=user_id,
            robot_pos=robot_pos, user_pos=user_pos, room=room)
        return jsonify(result), 200

    except Exception as e:
        import traceback
        print(f"[Interact Error] {e}\n{traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500


@app.route('/interact/confirm', methods=['POST'])
def interact_confirm():
    try:
        data = request.get_json()
        choice = int(data.get('choice', 3))
        nav_target = data.get('nav_target')
        nav_label = data.get('nav_label', '')
        user_id = data.get('userID', 'Unknown')
        query = data.get('query', '')
        result = interaction_engine.confirm(
            choice=choice, nav_target=nav_target,
            nav_label=nav_label, user_id=user_id, query=query)
        return jsonify(result), 200
    except Exception as e:
        import traceback
        print(f"[Confirm Error] {e}\n{traceback.format_exc()}")
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
        data = request.get_json()
        proposal_id = data.get("proposal_id", "")
        user_id = data.get("user_id", "Unknown")
        result = data.get("result", "ignored")

        if not proposal_id:
            return jsonify({"error": "proposal_id required"}), 400

        response = proposal_engine.handle_response(
            proposal_id=proposal_id, user_id=user_id,
            result=result, manifold_engine=manifold_engine)
        return jsonify(response), 200
    except Exception as e:
        import traceback
        print(f"[ServiceResponse Error] {e}\n{traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500


@app.route('/service_history', methods=['GET'])
def service_history():
    try:
        user_id = request.args.get("user_id")
        query = {"user_id": user_id} if user_id else {}
        proposals = list(
            db.service_proposals.find(query, {"_id": 0})
            .sort("created_at", -1).limit(50))
        for p in proposals:
            for k in ["created_at", "responded_at"]:
                if k in p and hasattr(p[k], "isoformat"):
                    p[k] = p[k].isoformat()
        stats = list(db.intent_stats.find(query, {"_id": 0}))
        return jsonify({"proposals": proposals, "intent_stats": stats,
                        "total": len(proposals)}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/track_position', methods=['POST'])
def track_position():
    try:
        data = request.get_json()
        user_id = data.get("userID", "Unknown")
        x = float(data.get("x", 0))
        z = float(data.get("z", 0))
        room_name = data.get("room_name", "")
        forward_x = float(data.get("forward_x", 0))
        forward_z = float(data.get("forward_z", 0))
        timestamp_str = data.get('timestamp', '')
        
        if timestamp_str:
            try:
                now = datetime.datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            except:
                now = datetime.datetime.utcnow()
        else:
            now = datetime.datetime.utcnow()

        update = {
            "user_id": user_id,
            "x": x,
            "z": z,
            "room": room_name,
            "updated_at": now,
        }
        update["forward"] = [forward_x, 0.0, forward_z]

        db.user_positions.update_one(
            {"user_id": user_id},
            {"$set": update},
            upsert=True
        )
        return jsonify({"status": "ok"}), 200
    except Exception as e:
        import traceback
        print(f"[TrackPosition Error] {e}\n{traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500


@app.route('/device_state', methods=['POST'])
def device_state():
    try:
        data = request.get_json()
        label = data.get('label', '').lower().strip()
        state = data.get('state', 'off')
        timestamp_str = data.get('timestamp', '')
        
        if timestamp_str:
            try:
                now = datetime.datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            except:
                now = datetime.datetime.utcnow()
        else:
            now = datetime.datetime.utcnow()
        
        if not label:
            return jsonify({"status": "error"}), 400
        db.device_states.update_one(
            {"label": label},
            {"$set": {
                "label": label,
                "state": state,
                "updated_at": now,
            }},
            upsert=True
        )
        print(f"[DeviceState] {label} -> {state} at {now}")
        return jsonify({"status": "ok", "label": label, "state": state}), 200
    except Exception as e:
        import traceback
        print(f"[DeviceState Error] {e}\n{traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500


@app.route('/log_navigation', methods=['POST'])
def log_navigation():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data"}), 400
        db["navigation_logs"].insert_one({
            "user_id": data.get("user_id"),
            "intent": data.get("intent"),
            "start_pos": data.get("start_pos"),
            "goal_pos": data.get("goal_pos"),
            "waypoints": data.get("waypoints", []),
            "success": data.get("success", False),
            "total_time": data.get("total_time", 0),
            "total_distance": data.get("total_distance", 0),
            "timestamp": datetime.datetime.utcnow(),
        })
        return jsonify({"status": "Success"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/manifold_status', methods=['GET'])
def manifold_status():
    try:
        status = {}
        for uid in ["User_Mom", "User_Dad"]:
            status[uid] = {
                "training_samples": manifold_engine.get_training_count(uid),
                "model_ready": manifold_engine._get_model(uid) is not None,
            }
        return jsonify({"status": "ok", "manifold": status}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/manifold_train', methods=['POST'])
def manifold_train():
    try:
        data = request.get_json(force=True, silent=True) or {}
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


@app.route('/habit_feedback', methods=['POST'])
def habit_feedback():
    try:
        data = request.get_json()
        user_id = data.get("user_id", "")
        result = data.get("result", "")
        intent = data.get("intent", "")
        item = data.get("item", "")
        if result == "rejected":
            habit_engine.handle_rejection(user_id, intent, item)
        elif result == "accepted":
            habit_engine.handle_acceptance(user_id, intent, item)
        return jsonify({"status": "ok"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/habit_check', methods=['POST'])
def habit_check():
    try:
        data = request.get_json()
        user_id = data.get("user_id", "")
        if not user_id:
            return jsonify({"error": "user_id required"}), 400
        habit_engine._check_and_update_skill(user_id)
        return jsonify({"status": "ok", "user_id": user_id}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/object_event', methods=['POST'])
def object_event():
    data = request.get_json()
    user_id = data.get('user_id')
    object_name = data.get('object')
    
    if 'pickup_time' in data:
        pickup_time = data.get('pickup_time')
        if pickup_time:
            try:
                pickup_time = datetime.datetime.fromisoformat(pickup_time.replace('Z', '+00:00'))
                pickup_time = pickup_time.replace(tzinfo=None)
            except:
                pickup_time = datetime.datetime.utcnow()
        else:
            pickup_time = datetime.datetime.utcnow()
        
        db.object_events.insert_one({
            "user": user_id,
            "object": object_name,
            "pickup_time": pickup_time,
            "putdown_time": None
        })
        print(f"[ObjectEvent] {user_id} picked up {object_name} at {pickup_time}")
        
    elif 'putdown_time' in data:
        putdown_time = data.get('putdown_time')
        if putdown_time:
            try:
                putdown_time = datetime.datetime.fromisoformat(putdown_time.replace('Z', '+00:00'))
                putdown_time = putdown_time.replace(tzinfo=None)
            except:
                putdown_time = datetime.datetime.utcnow()
        else:
            putdown_time = datetime.datetime.utcnow()
        
        db.object_events.update_one(
            {"user": user_id, "putdown_time": None},
            {"$set": {"putdown_time": putdown_time}}
        )
        print(f"[ObjectEvent] {user_id} put down at {putdown_time}")
    
    return jsonify({"status": "ok"}), 200





@app.route('/saycan', methods=['POST'])
def saycan():
    try:
        data = request.get_json()
        query = data.get("query", "")
        user_id = data.get("userID", "Unknown")
        user_pos_raw = data.get("user_pos", None)

        if not query:
            return jsonify({"error": "query is required"}), 400

        est_pos = None
        if user_pos_raw:
            est_pos = {"x": float(user_pos_raw.get("x", 0)),
                       "z": float(user_pos_raw.get("z", 0))}

        if est_pos is None:
            pos_doc = db.user_positions.find_one({"user_id": user_id})
            if pos_doc:
                est_pos = {"x": float(pos_doc.get("x", 0)),
                           "z": float(pos_doc.get("z", 0))}

        result = interaction_engine.process(
            query=query, user_id=user_id,
            user_pos=est_pos, room="")

        return jsonify({
            "status": "ok",
            "intent": result.get("intent_type", "need"),
            "best_action": result.get("recommendations", [{}])[0].get("label", ""),
            "best_score": result.get("confidence", 0.0),
            "explanation": result.get("answer", ""),
            "nav_target": result.get("nav_target"),
            "nav_label": result.get("nav_label"),
            "saycan_scores": result.get("saycan_scores", {}),
        }), 200

    except Exception as e:
        import traceback
        print(f"[SayCan Error] {e}\n{traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500


_predict_worker_thread = threading.Thread(target=_predict_worker, daemon=True)
_predict_worker_thread.start()
print("[Predict Queue] worker started")

if __name__ == "__main__":
    host = getattr(CONFIG, 'FLASK_HOST', '0.0.0.0')
    port = int(getattr(CONFIG, 'FLASK_PORT', 5000))
    print(f"\nRobot Brain Server on {host}:{port}")
    print(f"  VLM model : {CONFIG.VLM_MODEL}")
    print(f"  LLM model : {CONFIG.LLM_MODEL}")
    threading.Timer(86400, nightly_maintenance).start()
    app.run(host=host, port=port, debug=False)