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

from modules.perception.perception_engine import PerceptionEngine
from modules.perception.scene_engine      import SceneEngine

from modules.memory.observation_store import ObservationStore
from modules.memory.habit_learner      import HabitLearner
from modules.memory.skill_manager      import SkillManager
from modules.memory.manifold_engine    import ManifoldEngine
from modules.memory.memory_vector      import VectorMemory

from modules.service.reactive_service  import ReactiveService
from modules.service.proactive_service import ProactiveService
from modules.service.proposal_manager  import ProposalManager

from modules.utils.classifier      import ObjectClassifier, BASE_FURNITURE_KEYWORDS, OBJECT_CATEGORIES
from modules.utils.entropy_monitor import EntropyMonitor
from modules.utils.saycan_engine   import SayCanEngine

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



app    = Flask(__name__)
CONFIG = Config

werkzeug_logger = logging.getLogger("werkzeug")
werkzeug_logger.setLevel(logging.WARNING)
print(f"System init: device={CONFIG.DEVICE} "
      f"| LLM={CONFIG.LLM_MODEL} | VLM={CONFIG.VLM_MODEL}")

sbert_model = SentenceTransformer('all-MiniLM-L6-v2', device=CONFIG.DEVICE)
print("SBERT loaded on", CONFIG.DEVICE)

mongo_client = MongoClient(CONFIG.MONGO_URI)
db           = mongo_client[CONFIG.DB_NAME]

try:
    db.scene_snapshots.create_index([("pos", "2d")])
    db.observation_logs.create_index(
        [("last_seen", 1)],
        expireAfterSeconds=14 * 86400,
        name="observation_ttl_14d")
    db.transition_counts.create_index(
        [("user_id", 1), ("from_action", 1), ("time_slot", 1)])
    print("[MongoDB] indexes ready")
except Exception as e:
    print(f"[MongoDB] index notice: {e}")

_ontology = _load_yaml("config/robot_ontology.yaml")
_beh_cfg  = _load_yaml("config/behavior_config.yaml")
_sys_cfg  = _load_yaml("config/system_config.yaml")

behavior_labels = _beh_cfg.get("behavior_labels", [
    "Drinking", "SittingDrink", "Sitting", "Eating", "Cooking", "Opening",
    "Laying", "Watching", "Reading", "Cleaning", "PhoneUse",
    "Typing", "StandUp", "PickingUp", "PuttingDown", "Standing", "Walking",
])


scene_engine = SceneEngine(
    db=db, ollama_url=CONFIG.OLLAMA_URL,
    sbert_model=sbert_model, ontology=_ontology,
    system_cfg=_sys_cfg, behavior_labels=behavior_labels,
)

manifold_engine = ManifoldEngine(db=db, sbert_model=sbert_model)

vector_memory = VectorMemory(device=CONFIG.DEVICE)
vector_memory.sync_from_mongo(db.dynamic_objects)

perception = PerceptionEngine(
    db=db, ollama_url=CONFIG.OLLAMA_URL,
    vlm_model=CONFIG.VLM_MODEL,
    sbert_model=sbert_model, scene_engine=scene_engine,
)
perception.manifold_engine = manifold_engine

skill_manager = SkillManager(
    db_client=mongo_client,
    ollama_url=CONFIG.OLLAMA_URL,
    model_name=CONFIG.LLM_MODEL,
)

observation_store = ObservationStore(db=db)

habit_learner = HabitLearner(
    db=db,
    skill_manager=skill_manager,
)

proposal_manager = ProposalManager(db=db)

proactive_service = ProactiveService(
    db=db,
    habit_learner=habit_learner,
    manifold_engine=manifold_engine,
    proposal_manager=proposal_manager,
    ollama_url=CONFIG.OLLAMA_URL,
    llm_model=CONFIG.LLM_MODEL,
)

reactive_service = ReactiveService(
    db=db,
    skill_manager=skill_manager,
    vector_memory=vector_memory,
    ollama_url=CONFIG.OLLAMA_URL,
    llm_model=CONFIG.LLM_MODEL,
)

saycan_engine = SayCanEngine(
    db=db, manifold_engine=manifold_engine,
    ollama_url=CONFIG.OLLAMA_URL,
    llm_model=CONFIG.LLM_MODEL,
    sbert_model=sbert_model,
    vector_memory=vector_memory,
)

entropy_monitor = EntropyMonitor()
classifier      = ObjectClassifier(db)
classifier.start()

atexit.register(perception.shutdown)


_vlm_lock      = threading.Lock()
_predict_queue = _queue.Queue()
_gt_cache      = {}
_gt_cache_lock = threading.Lock()

_robot_state = {
    "nav_target": None,
    "nav_label":  "",
    "last_answer": "",
    "highlight":  "",
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


def _find_nearest_furniture(x: float, z: float, room: str,
                             max_dist: float = 3.0) -> str:
    _BLACKLIST = {
        "floor", "ceiling", "wall", "ground", "wooden floor",
        "tile floor", "carpet", "concrete floor", "baseboard",
        "white wall", "window", "door",
    }
    query = {}
    if room:
        query["room"] = {"$regex": room, "$options": "i"}
    docs = list(db.scene_snapshots.find(query, {"label": 1, "pos": 1}))
    if not docs:
        docs = list(db.scene_snapshots.find({}, {"label": 1, "pos": 1}))

    best_label = "Unknown_Area"
    best_dist  = float("inf")
    for doc in docs:
        label = doc.get("label", "").lower().strip()
        if label in _BLACKLIST:
            continue
        pos = doc.get("pos")
        if not isinstance(pos, list) or len(pos) < 2:
            continue
        dist = math.sqrt((x - pos[0]) ** 2 + (z - pos[1]) ** 2)
        if dist < best_dist:
            best_dist  = dist
            best_label = doc["label"]
    return best_label if best_dist <= max_dist else "Unknown_Area"


def _get_category_for_label(label: str) -> str:
    label_l = label.lower().strip()
    for cat, keywords in OBJECT_CATEGORIES.items():
        if label_l in keywords or any(kw in label_l for kw in keywords):
            return cat
    return "other"


def _get_time_slot(virtual_hour) -> str:
    try:
        h = float(virtual_hour) if virtual_hour is not None else \
            float(datetime.datetime.now().hour)
        if h < 10:  return "Morning"
        if h < 13:  return "Noon"
        if h < 18:  return "Afternoon"
        if h < 22:  return "Evening"
        return "Night"
    except Exception:
        return "Unknown"



def nightly_maintenance():
    print("[Maintenance] running nightly tasks...")
    try:
        # Habit decay
        db.observation_logs.update_many(
            {}, {"$mul": {"weight": CONFIG.HABIT_DECAY_FACTOR}})
        db.observation_logs.delete_many(
            {"weight": {"$lt": CONFIG.HABIT_MIN_WEIGHT}})

        # Transition counts recency decay
        habit_learner._apply_recency_decay()

        # Skill refactor
        for doc in db.user_skills.find({}, {"user_id": 1}):
            try:
                skill_manager.nightly_refactor(doc["user_id"])
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
        "ready":          status["ready"],
        "zone_count":     status["zone_count"],
        "affinity_count": status["affinity_count"],
        "zones":          status["zones"],
    }), 200


@app.route("/experiment_done", methods=["POST"])
def experiment_done():
    def _final_train():
        for uid in ["User_Mom", "User_Dad"]:
            n = db.manifold_training_data.count_documents({"user_id": uid})
            if n >= 20:
                manifold_engine.train_model(uid)
    threading.Thread(target=_final_train, daemon=True).start()
    return jsonify({"status": "ok"}), 200



@app.route('/nav_target',  methods=['GET'])
def get_nav_target():
    return jsonify({
        "nav_target": _robot_state["nav_target"],
        "nav_label":  _robot_state["nav_label"],
    })

@app.route('/highlight',   methods=['GET'])
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
        result = reactive_service.process(
            query=query, user_id=user_id, room=room)

        answer     = result.get("answer", "")
        nav_target = result.get("nav_target")
        nav_label  = result.get("nav_label", "")

        # Stream answer token by token
        for char in answer:
            yield f"data: {json.dumps({'type': 'token', 'content': char})}\n\n"

        # Update robot state
        _robot_state["last_answer"] = answer
        _robot_state["nav_target"]  = nav_target
        _robot_state["nav_label"]   = nav_label or ""
        _robot_state["highlight"]   = nav_label or ""

        yield f"data: {json.dumps({'type': 'done', **result})}\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype='text/event-stream',
        headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'},
    )


@app.route('/interact', methods=['POST'])
def interact():
    data    = request.get_json()
    query   = data.get('query', '')
    user_id = data.get('userID', 'Unknown')
    room    = data.get('room', '')
    if not query:
        return jsonify({"error": "Empty query"}), 400
    result = reactive_service.process(
        query=query, user_id=user_id, room=room)
    return jsonify(result), 200


@app.route('/interact/confirm', methods=['POST'])
def interact_confirm():
    data       = request.get_json()
    choice     = int(data.get('choice', 3))
    nav_target = data.get('nav_target')
    nav_label  = data.get('nav_label', '')
    user_id    = data.get('userID', 'Unknown')

    if choice == 1:
        return jsonify({
            "status":    "navigate",
            "nav_target": nav_target,
            "nav_label":  nav_label,
            "message":   f"Navigating to {nav_label}.",
        })
    if choice == 2:
        pos_str = (f"[{nav_target[0]:.1f}, {nav_target[1]:.1f}]"
                   if nav_target else "unknown")
        return jsonify({
            "status":  "info_only",
            "message": f"{nav_label} is at {pos_str}.",
        })
    return jsonify({"status": "cancelled", "message": "Cancelled."})



@app.route('/service_proposal', methods=['GET'])
def service_proposal():
    proposal = proposal_manager.get_next()
    if proposal:
        return jsonify(proposal), 200
    return jsonify({"status": "no_proposal"}), 200


@app.route('/service_response', methods=['POST'])
def service_response():
    data        = request.get_json()
    proposal_id = data.get("proposal_id", "")
    user_id     = data.get("user_id", "Unknown")
    result      = data.get("result", "ignored")
    if not proposal_id:
        return jsonify({"error": "proposal_id required"}), 400
    response = proposal_manager.handle_response(
        proposal_id=proposal_id, user_id=user_id,
        result=result, manifold_engine=manifold_engine)
    return jsonify(response), 200


@app.route('/service_history', methods=['GET'])
def service_history():
    user_id   = request.args.get("user_id")
    proposals = proposal_manager.get_history(user_id=user_id)
    return jsonify({"proposals": proposals, "total": len(proposals)}), 200



@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data received"}), 400

    episode_id = data.get("episode_id") or str(uuid.uuid4())
    existing   = db.eval_logs.find_one({"episode_id": episode_id})
    if existing:
        return jsonify({
            "status": "ok", "episode_id": episode_id, "cached": True}), 200

    t_capture    = data.get("t_capture", "")
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


def _process_predict(episode_id: str, data: dict):
    import time

    image_list   = data.get('image_list', [])
    hint_user_id = data.get('userID', 'Unknown_User')
    activity     = data.get('activity', '')
    user_pos_raw = data.get('user_pos')
    user_fwd_raw = data.get('user_forward')
    source_nodes = data.get('source_nodes', [])
    room_name    = data.get('room_name', '')
    virtual_hour = data.get('virtual_hour')
    virtual_day  = data.get('virtual_day', '')
    t_capture    = data.get('t_capture', '')

    if not room_name and source_nodes:
        first_node = source_nodes[0].split('_b')[0]
        camIdx     = first_node.rfind('_Cam')
        room_name  = first_node[:camIdx] if camIdx > 0 else first_node

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

    with _vlm_lock:
        t0     = time.time()
        result = perception.analyze_action_burst(data)
        vlm_ms = int((time.time() - t0) * 1000)

    _user_id       = result.get("user", "") or result.get("user_id", "")
    _user_pos      = result.get("user_pos") or {}
    _exp_mode      = result.get("experiment_mode", "habit")
    spatial_action = result.get("spatial_action") or result.get("action", "Unknown")
    zone_label     = result.get("zone_label") or result.get("zone_name") or ""
    time_slot      = _get_time_slot(virtual_hour)

    # ── Entropy monitoring ────────────────────────────────────────────────────
    _activity_votes = [result.get("action", "")] if result.get("action") else []
    _body_votes     = [result["result"].get("_body_position", "")]
    _held_votes     = [result["result"].get("_held_event", "none")]
    _entropy_info   = entropy_monitor.analyze(
        _user_id, _activity_votes, _body_votes, _held_votes)

    print(f"[VLM] vlm={result.get('action')} spatial={spatial_action} "
          f"entropy={_entropy_info['overall_entropy']:.2f} | {vlm_ms}ms")

    if spatial_action in ("none", "", "Unknown"):
        return

    # ── Memory Layer: record observation ─────────────────────────────────────
    if zone_label and spatial_action not in {"Walking", "Standing", "StandUp",
                                              "PickingUp", "PuttingDown"}:
        today   = datetime.datetime.utcnow().strftime("%Y-%m-%d")
        pos_xy  = [
            _user_pos.get("x", 0) / 10.0,
            _user_pos.get("z", 0) / 10.0,
        ]

        observation_store.record(
            user_id=_user_id,
            action=spatial_action,
            zone_name=zone_label,
            instance=zone_label,
            time_slot=time_slot,
            interacting_items=result.get("items", []),
            spatial_relations=result.get("spatial_relations", {}),
            pos_xy=pos_xy,
            room=result.get("room", ""),
            raw_desc=result["result"].get("context", ""),
            today=today,
        )

        # Get previous action for transition learning
        prev_seq = observation_store.get_recent_sequence(_user_id, limit=2)
        prev_action = prev_seq[-2] if len(prev_seq) >= 2 else "Standing"

        habit_learner.on_new_observation(
            user_id=_user_id,
            action=spatial_action,
            prev_action=prev_action,
            time_slot=time_slot,
            zone_name=zone_label,
        )

        scene_engine.update_user_affinity(
            user_id=_user_id,
            zone_name=zone_label,
            action=spatial_action,
            room=result.get("room", ""),
            virtual_day=virtual_day,
        )

    # ── Memory Layer: manifold training ──────────────────────────────────────
    NO_RECORD = {"Unknown", "Standing", "Walking", "StandUp",
                 "PickingUp", "PuttingDown"}
    if (spatial_action not in NO_RECORD and
            _exp_mode != "recognition" and
            result["result"].get("_vlm_confidence", 0) >= 0.20):

        prev_seq    = observation_store.get_recent_sequence(_user_id, limit=2)
        prev_action = prev_seq[-2] if len(prev_seq) >= 2 else "Standing"

        try:
            manifold_engine.record_training_sample(
                user_id=_user_id,
                virtual_hour=virtual_hour,
                user_pos=est_pos,
                prev_action=prev_action,
                current_action=spatial_action,
            )
        except Exception:
            pass

    # ── Service Layer: proactive service evaluation ───────────────────────────
    # Trigger only when user returns to Standing (finished an action)
    prev_seq    = observation_store.get_recent_sequence(_user_id, limit=2)
    prev_action = prev_seq[-2] if len(prev_seq) >= 2 else "Standing"

    if spatial_action == "Standing" and prev_action not in ("Standing", "Walking"):
        proposal = proactive_service.evaluate(
            user_id=_user_id,
            current_action=spatial_action,
            prev_action=prev_action,
            time_slot=time_slot,
            user_pos=est_pos,
        )
        if proposal:
            proposal_manager.push(_user_id, proposal)

    # ── Memory: vector memory ─────────────────────────────────────────────────
    bound_label = result.get("bound_instance", "Unknown_Area")
    detected    = result.get("items", [])
    all_items   = result.get("all_items", [])
    spatial_rels = result.get("spatial_relations", [])

    if bound_label and "Unknown" not in bound_label:
        furniture_doc = db.scene_snapshots.find_one({"label": bound_label})
        furniture_pos = furniture_doc.get('pos') if furniture_doc else None
        mongo_id      = furniture_doc.get('_id')  if furniture_doc else None
        vlm_desc      = result["result"].get("context", "")

        spatial_text = " ".join(
            [f"{r['subject']} {r['relation']} {r['object']}"
             for r in spatial_rels]) if spatial_rels else ""

        vector_memory.add_memory(
            user_id=_user_id,
            action=spatial_action,
            furniture_label=bound_label,
            vlm_description=f"{vlm_desc} {spatial_text}".strip(),
            detected_items=detected,
            all_items=all_items,
            spatial_relations=spatial_rels,
            furniture_pos=furniture_pos,
            mongo_id=mongo_id,
        )

    # ── Manifold: intent prediction for proposal ──────────────────────────────
    try:
        _prev_seq_doc = db.activity_sequences.find_one(
            {"user": _user_id}, sort=[("date", -1)])
        _real_prev = "Standing"
        if _prev_seq_doc and len(_prev_seq_doc.get("sequence", [])) >= 2:
            _real_prev = _prev_seq_doc["sequence"][-2].get("action", "Standing")

        intent_pred = manifold_engine.predict_intent(
            user_id=_user_id,
            virtual_hour=virtual_hour,
            user_pos=est_pos,
            prev_action=_real_prev,
        )
    except Exception as e:
        print(f"[Manifold] non-critical error: {e}")

    # ── Activity sequence (app-level, for bind label) ─────────────────────────
    final_bound = bound_label if "Unknown" not in bound_label else ""
    if not final_bound and est_pos:
        best_doc  = None
        best_dist = float("inf")
        query_r   = {"room": {"$regex": room_name, "$options": "i"}} if room_name else {}
        for doc in db.scene_snapshots.find(query_r, {"label": 1, "pos": 1}):
            pos = doc.get("pos")
            if not isinstance(pos, list) or len(pos) < 2:
                continue
            dist = math.sqrt(
                (est_pos["x"] - pos[0]) ** 2 +
                (est_pos["z"] - pos[1]) ** 2
            )
            if dist < best_dist:
                best_dist = dist
                best_doc  = doc
        if best_doc and best_dist <= 5.0:
            final_bound = best_doc["label"]

    final_bound = final_bound or "Unknown_Area"
    print(f"[Bind] '{bound_label}' -> '{final_bound}'")

    try:
        today = datetime.datetime.utcnow().strftime("%Y-%m-%d")
        db.activity_sequences.update_one(
            {"user": _user_id, "date": today},
            {
                "$push": {
                    "sequence": {
                        "action":    spatial_action,
                        "instance":  final_bound,
                        "timestamp": datetime.datetime.utcnow(),
                    }
                },
                "$setOnInsert": {"user": _user_id, "date": today},
            },
            upsert=True,
        )
    except Exception as e:
        print(f"[Sequence] {e}")

    # ── Update eval_logs ──────────────────────────────────────────────────────
    db.eval_logs.update_one(
        {"episode_id": episode_id},
        {"$set": {
            "status":    "done",
            "entropy":   _entropy_info["overall_entropy"],
            "forward_x": est_forward.get("x", 0) if est_forward else None,
            "forward_z": est_forward.get("z", 0) if est_forward else None,
            "vlm_ms":    vlm_ms,
        }},
    )

    print(f"[Predict] done | {_user_id} | "
          f"{result.get('action')} -> {spatial_action} | {vlm_ms}ms")



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
            if any(kw in label for kw in BASE_FURNITURE_KEYWORDS):
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
                "source":      obj.get('source', 'sensor'),
                "held_by":     obj.get('held_by', ''),
                "processed":   False,
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
        data    = request.get_json()
        objects = data.get('objects', [])
        if not objects:
            return jsonify({"status": "empty"}), 200

        timestamp_str = data.get('timestamp', '')
        try:
            now = datetime.datetime.fromisoformat(
                timestamp_str.replace('Z', '+00:00')) if timestamp_str \
                else datetime.datetime.utcnow()
        except Exception:
            now = datetime.datetime.utcnow()

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

            room         = obj.get('room', '')
            position     = obj.get('position', [0, 0])
            x, z         = float(position[0]), float(position[1])
            held_by      = obj.get('held_by', '')
            last_seen_on = _find_nearest_furniture(x, z, room)
            category     = _get_category_for_label(label)

            set_fields = {
                "room":         room,
                "sensor_pos":   position,
                "last_seen":    now,
                "status":       "active",
                "status_since": now,
                "source":       source,
                "category":     category,
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
                    "$set":         set_fields,
                    "$inc":         {"seen_count": 1},
                    "$setOnInsert": {"first_seen": now, "spatial_rel": "on"},
                },
                upsert=True,
            )

            dyn_doc = db.dynamic_objects.find_one({"label": label})
            if dyn_doc:
                vector_memory.upsert_dynamic_object(
                    label=label, room=room,
                    last_seen_on=last_seen_on, spatial_rel="on",
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
                "label":  {"$nin": unity_labels},
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
    data  = request.get_json()
    label = data.get('label', '')
    state = data.get('state', 'off')
    timestamp_str = data.get('timestamp', '')
    try:
        now = datetime.datetime.fromisoformat(
            timestamp_str.replace('Z', '+00:00')) if timestamp_str \
            else datetime.datetime.utcnow()
    except Exception:
        now = datetime.datetime.utcnow()
    if label:
        db.device_states.update_one(
            {'label': label},
            {'$set': {'state': state, 'updated_at': now}},
            upsert=True)
    return jsonify({'status': 'ok'}), 200


@app.route('/device_state', methods=['POST'])
def device_state():
    try:
        data  = request.get_json()
        label = data.get('label', '').lower().strip()
        state = data.get('state', 'off')
        timestamp_str = data.get('timestamp', '')
        try:
            now = datetime.datetime.fromisoformat(
                timestamp_str.replace('Z', '+00:00')) if timestamp_str \
                else datetime.datetime.utcnow()
        except Exception:
            now = datetime.datetime.utcnow()
        if not label:
            return jsonify({"status": "error"}), 400
        db.device_states.update_one(
            {"label": label},
            {"$set": {"label": label, "state": state, "updated_at": now}},
            upsert=True)
        print(f"[DeviceState] {label} -> {state} at {now}")
        return jsonify({"status": "ok", "label": label, "state": state}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/object_event', methods=['POST'])
def object_event():
    data        = request.get_json()
    user_id     = data.get('user_id')
    object_name = data.get('object')

    if 'pickup_time' in data:
        pickup_time = data.get('pickup_time')
        try:
            pickup_time = datetime.datetime.fromisoformat(
                pickup_time.replace('Z', '+00:00')).replace(tzinfo=None) \
                if pickup_time else datetime.datetime.utcnow()
        except Exception:
            pickup_time = datetime.datetime.utcnow()
        db.object_events.insert_one({
            "user":         user_id,
            "object":       object_name,
            "pickup_time":  pickup_time,
            "putdown_time": None,
        })
        print(f"[ObjectEvent] {user_id} picked up {object_name} at {pickup_time}")

    elif 'putdown_time' in data:
        putdown_time = data.get('putdown_time')
        try:
            putdown_time = datetime.datetime.fromisoformat(
                putdown_time.replace('Z', '+00:00')).replace(tzinfo=None) \
                if putdown_time else datetime.datetime.utcnow()
        except Exception:
            putdown_time = datetime.datetime.utcnow()
        db.object_events.find_one_and_update(
            {"user": user_id, "putdown_time": None},
            {"$set": {"putdown_time": putdown_time}},
            sort=[("pickup_time", -1)],
        )
        print(f"[ObjectEvent] {user_id} put down at {putdown_time}")

    return jsonify({"status": "ok"}), 200



@app.route('/track_position', methods=['POST'])
def track_position():
    try:
        data          = request.get_json()
        user_id       = data.get("userID", "Unknown")
        x             = float(data.get("x", 0))
        z             = float(data.get("z", 0))
        room_name     = data.get("room_name", "")
        forward_x     = float(data.get("forward_x", 0))
        forward_z     = float(data.get("forward_z", 0))
        timestamp_str = data.get('timestamp', '')
        try:
            now = datetime.datetime.fromisoformat(
                timestamp_str.replace('Z', '+00:00')) if timestamp_str \
                else datetime.datetime.utcnow()
        except Exception:
            now = datetime.datetime.utcnow()
        db.user_positions.update_one(
            {"user_id": user_id},
            {"$set": {
                "user_id":    user_id,
                "x":          x, "z": z,
                "room":       room_name,
                "forward":    [forward_x, 0.0, forward_z],
                "updated_at": now,
            }},
            upsert=True)
        return jsonify({"status": "ok"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


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

        obs    = db.observation_logs.find_one(
            {"user": user_id, "action": {"$regex": action, "$options": "i"}},
            sort=[("weight", -1)])
        weight = obs["weight"] if obs else 0

        similarity = 0.0
        try:
            results = vector_memory.search_habit(
                f"{user_id} {action}", user_id=user_id, top_k=1)
            if results:
                similarity = float(results[0].get("similarity", 0.0))
        except Exception:
            pass

        db["exp_checkpoint_logs"].insert_one({
            "experiment": experiment, "episode": episode,
            "user_id": user_id, "action": action,
            "weight": weight, "similarity": round(similarity, 4),
            "timestamp": datetime.datetime.utcnow(),
        })
        return jsonify({
            "status": "ok", "experiment": experiment,
            "episode": episode, "user_id": user_id,
            "action": action, "weight": weight,
            "similarity": round(similarity, 4),
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500



@app.route('/manifold_status', methods=['GET'])
def manifold_status():
    try:
        status = {}
        for uid in ["User_Mom", "User_Dad"]:
            status[uid] = {
                "training_samples": manifold_engine.get_training_count(uid),
                "model_ready":      manifold_engine._get_model(uid) is not None,
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
        return jsonify({
            "status": "training started", "user_id": user_id}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500



@app.route('/nav_target', methods=['GET'])
def nav_target():
    return jsonify({
        "nav_target": _robot_state["nav_target"],
        "nav_label":  _robot_state["nav_label"],
    })


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
            "success":        data.get("success", False),
            "total_time":     data.get("total_time", 0),
            "total_distance": data.get("total_distance", 0),
            "timestamp":      datetime.datetime.utcnow(),
        })
        return jsonify({"status": "Success"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500



_predict_worker_thread = threading.Thread(
    target=_predict_worker, daemon=True)
_predict_worker_thread.start()
print("[Predict Queue] worker started")

if __name__ == "__main__":
    host = getattr(CONFIG, 'FLASK_HOST', '0.0.0.0')
    port = int(getattr(CONFIG, 'FLASK_PORT', 5000))
    print(f"\nRobot Brain Server on {host}:{port}")
    print(f"  Perception : {CONFIG.VLM_MODEL} + {CONFIG.LLM_MODEL}")
    print(f"  Memory     : ObservationStore + HabitLearner + ManifoldEngine")
    print(f"  Service    : ProactiveService + ReactiveService")
    threading.Timer(86400, nightly_maintenance).start()
    app.run(host=host, port=port, debug=False)