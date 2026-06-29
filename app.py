import logging
import os
import json
import math
import uuid
import atexit
import base64
import datetime
import threading
import time
import queue as _queue
from collections import defaultdict
from datetime import date as _date, timedelta as _td

import numpy as np
import cv2
from flask import Flask, request, jsonify, Response, stream_with_context
from flask_cors import CORS
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer

from config import Config

from modules.perception.perception_engine import PerceptionEngine
from modules.perception.scene_engine      import SceneEngine
from modules.memory.observation_store     import ObservationStore
from modules.memory.habit_learner         import HabitLearner
from modules.memory.skill_manager         import SkillManager
from modules.service.reactive_service     import ReactiveService
from modules.service.proactive_service    import ProactiveService
from modules.service.proposal_manager     import ProposalManager
from modules.utils.classifier             import ObjectClassifier, BASE_FURNITURE_KEYWORDS, OBJECT_CATEGORIES

try:
    import yaml as _yaml
    _YAML_OK = True
except ImportError:
    _YAML_OK = False


# ─── helpers ──────────────────────────────────────────────────────────────────

def _load_yaml(path: str) -> dict:
    if _YAML_OK and os.path.exists(path):
        with open(path) as f:
            return _yaml.safe_load(f) or {}
    return {}


_BASE_VIRTUAL_DATE = _date(2025, 1, 1)


def _parse_virtual_time(data: dict) -> dict:
    virtual_hour = data.get("virtual_hour")
    virtual_day  = data.get("virtual_day")
    time_slot    = data.get("time_slot") or _get_time_slot(virtual_hour)
    virtual_date = (
        (_BASE_VIRTUAL_DATE + _td(days=int(virtual_day) - 1)).strftime("%Y-%m-%d")
        if virtual_day else
        datetime.datetime.utcnow().strftime("%Y-%m-%d")
    )
    return {
        "virtual_hour": virtual_hour,
        "virtual_day":  virtual_day,
        "time_slot":    time_slot,
        "virtual_date": virtual_date,
    }


def _get_time_slot(virtual_hour) -> str:
    try:
        h = float(virtual_hour) if virtual_hour is not None else float(datetime.datetime.now().hour)
        if h < 10:  return "Morning"
        if h < 13:  return "Noon"
        if h < 18:  return "Afternoon"
        if h < 22:  return "Evening"
        return "Night"
    except Exception:
        return "Unknown"


def _find_nearest_furniture(x: float, z: float, room: str, max_dist: float = 3.0) -> str:
    _BLACKLIST = {
        "floor", "ceiling", "wall", "ground", "wooden floor",
        "tile floor", "carpet", "concrete floor", "baseboard",
        "white wall", "window", "door",
    }
    query = {"room": {"$regex": room, "$options": "i"}} if room else {}
    docs  = list(db.scene_snapshots.find(query, {"label": 1, "pos": 1}))
    if not docs:
        docs = list(db.scene_snapshots.find({}, {"label": 1, "pos": 1}))
    best_label, best_dist = "Unknown_Area", float("inf")
    for doc in docs:
        label = doc.get("label", "").lower().strip()
        if label in _BLACKLIST:
            continue
        pos = doc.get("pos")
        if not isinstance(pos, list) or len(pos) < 2:
            continue
        dist = math.sqrt((x - pos[0]) ** 2 + (z - pos[1]) ** 2)
        if dist < best_dist:
            best_dist, best_label = dist, doc["label"]
    return best_label if best_dist <= max_dist else "Unknown_Area"


def _get_category_for_label(label: str) -> str:
    label_l = label.lower().strip()
    for cat, keywords in OBJECT_CATEGORIES.items():
        if label_l in keywords or any(kw in label_l for kw in keywords):
            return cat
    return "other"


def _wait_for_scene(max_wait: float = 12.0, poll: float = 1.0):
    waited = 0.0
    while waited < max_wait:
        if db.scene_snapshots.count_documents({}) > 0:
            return
        time.sleep(poll)
        waited += poll


def preview_images(image_list, source_nodes, hint_user_id, activity):
    save_dir = "debug_images"
    os.makedirs(save_dir, exist_ok=True)
    for i, img_b64 in enumerate(image_list):
        try:
            img_clean = img_b64.split(",")[1] if "," in img_b64 else img_b64
            nparr     = np.frombuffer(base64.b64decode(img_clean), np.uint8)
            frame     = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if frame is None:
                continue
            ts        = datetime.datetime.now().strftime("%H%M%S")
            node_name = source_nodes[i] if i < len(source_nodes) else f"img_{i}"
            cv2.imwrite(f"{save_dir}/{ts}_{hint_user_id}_{activity}_{node_name}.jpg", frame)
        except Exception as e:
            print(f"[Preview] {e}")


# ─── app & globals ────────────────────────────────────────────────────────────

app    = Flask(__name__)
CORS(app)
CONFIG = Config

logging.getLogger("werkzeug").setLevel(logging.WARNING)
print(f"[App] device={CONFIG.DEVICE} | LLM={CONFIG.LLM_MODEL} | VLM={CONFIG.VLM_MODEL}")

sbert_model  = SentenceTransformer("all-MiniLM-L6-v2", device=CONFIG.DEVICE)
mongo_client = MongoClient(CONFIG.MONGO_URI)
db           = mongo_client[CONFIG.DB_NAME]
print(f"[App] DB={CONFIG.DB_NAME} | SBERT loaded")

# indexes
try:
    db.scene_snapshots.create_index([("pos", "2d")])
    db.observation_logs.create_index(
        [("last_seen", 1)], expireAfterSeconds=14 * 86400, name="observation_ttl_14d")
    db.transition_counts.create_index(
        [("user_id", 1), ("from_action", 1), ("time_slot", 1)])
    print("[MongoDB] indexes ready")
except Exception as e:
    print(f"[MongoDB] index notice: {e}")

# yaml
_defs    = _load_yaml(CONFIG.DEFINITIONS_YAML)
_objects = _load_yaml(CONFIG.OBJECTS_YAML)

behavior_labels = _defs.get("behavior_labels", [
    "Drinking", "SeatedDrinking", "Sitting", "Eating", "Cooking", "Opening",
    "Laying", "Watching", "Reading", "Cleaning", "UsingPhone",
    "Typing", "StandUp", "PickingUp", "PuttingDown", "Standing", "Walking",
])

_sys_cfg_compat = {
    "hyperparameters": {
        "delta_threshold":      CONFIG.DELTA_THRESHOLD,
        "cross_room_gamma":     CONFIG.CROSS_ROOM_GAMMA,
        "base_mass_ch12":       CONFIG.BASE_MASS_CH12,
        "base_mass_ch3":        CONFIG.BASE_MASS_CH3,
        "base_mass_weak":       CONFIG.BASE_MASS_WEAK,
        "max_zone_search":      CONFIG.MAX_ZONE_SEARCH,
        "scene_retry_interval": CONFIG.SCENE_RETRY_INTERVAL,
        "scene_retry_max":      CONFIG.SCENE_RETRY_MAX,
    },
    "affordance_descriptions": _defs.get("affordance_descriptions", {}),
    "charades_affinity":       _defs.get("charades_affinity", {}),
}

_ontology_compat = {
    "structural_blacklist":      _objects.get("structural_blacklist", []),
    "static_fixtures":           _objects.get("static_fixtures", []),
    "high_affordance_furniture": _objects.get("high_affordance_furniture", []),
    "exclusive_behaviours":      _defs.get("exclusive_behaviours", []),
    "scene_objects":             list(_objects.get("item_to_action", {}).keys()),
    "item_to_action":            _objects.get("item_to_action", {}),
    "label_normalize_map":       _objects.get("label_normalize_map", {}),
    "object_vocab":              _objects.get("object_vocab", []),
}

# modules
scene_engine = SceneEngine(
    db=db, ollama_url=CONFIG.OLLAMA_URL,
    sbert_model=sbert_model,
    ontology=_ontology_compat,
    system_cfg=_sys_cfg_compat,
    behavior_labels=behavior_labels,
)

perception = PerceptionEngine(
    db=db, ollama_url=CONFIG.OLLAMA_URL,
    vlm_model=CONFIG.VLM_MODEL,
    sbert_model=sbert_model,
    scene_engine=scene_engine,
)

skill_manager     = SkillManager(db_client=mongo_client, ollama_url=CONFIG.OLLAMA_URL,
                                 model_name=CONFIG.LLM_MODEL, db_name=CONFIG.DB_NAME)
observation_store = ObservationStore(db=db)
habit_learner     = HabitLearner(db=db, skill_manager=skill_manager)
proposal_manager  = ProposalManager(db=db)
proactive_service = ProactiveService(db=db, habit_learner=habit_learner,
                                     proposal_manager=proposal_manager,
                                     ollama_url=CONFIG.OLLAMA_URL, llm_model=CONFIG.LLM_MODEL)
reactive_service  = ReactiveService(db=db, skill_manager=skill_manager,
                                    ollama_url=CONFIG.OLLAMA_URL, llm_model=CONFIG.LLM_MODEL)
classifier        = ObjectClassifier(db)
classifier.start()
atexit.register(perception.shutdown)

# state
_vlm_lock      = threading.Lock()
_predict_queue = _queue.Queue()
_is_processing = False
_gt_cache      = {}
_gt_cache_lock = threading.Lock()

_robot_state = {"nav_target": None, "nav_label": "", "last_answer": "", "highlight": ""}
_demo_state  = {"current_scene": 0, "scene_done": False, "scene_user": ""}
_speaking_state    = {"who": "none"}
_experiment_state  = {"mode": "baseline", "ablation_mode": "full",
                      "collection_suffix": "", "active": False}


# ─── DB switch ────────────────────────────────────────────────────────────────

def _switch_db(db_name: str):
    global db
    if db_name == db.name:
        return
    db = mongo_client[db_name]

    perception.db             = db
    perception.col_scene      = db.scene_snapshots
    perception.col_dynamic    = db.dynamic_objects
    perception.col_obs        = db.observation_logs
    perception.col_activity   = db.activity_sequences
    perception.col_user_aff   = db.user_spatial_affinity
    perception.col_aff_hist   = db.affinity_history
    perception._bulk_buf      = perception._bulk_buf.__class__(db.dynamic_objects)

    scene_engine.db           = db
    scene_engine.col_scene    = db.scene_snapshots
    scene_engine.col_affinity = db.affinity_matrix
    scene_engine.col_hist     = db.affinity_history
    scene_engine.col_user_aff = db.user_spatial_affinity

    observation_store.db      = db
    observation_store.col_obs  = db.observation_logs
    observation_store.col_snap = db.habit_snapshots
    observation_store.col_seq  = db.activity_sequences

    habit_learner.db              = db
    habit_learner.col_transitions = db.transition_counts
    habit_learner.col_obs         = db.observation_logs

    skill_manager.db = db[db_name]

    proposal_manager.db  = db
    proposal_manager.col = db.service_proposals

    reactive_service.db  = db
    proactive_service.db = db

    print(f"[App] Switched DB → {db_name}")


# ─── maintenance ──────────────────────────────────────────────────────────────

def nightly_maintenance():
    print("[Maintenance] running...")
    try:
        db.observation_logs.update_many({}, {"$mul": {"weight": CONFIG.HABIT_DECAY_FACTOR}})
        db.observation_logs.delete_many({"weight": {"$lt": CONFIG.HABIT_MIN_WEIGHT}})
        habit_learner._apply_recency_decay()
        for doc in db.user_skills.find({}, {"user_id": 1}):
            try:
                skill_manager.nightly_refactor(doc["user_id"])
            except Exception as e:
                print(f"[Maintenance] {e}")
        print("[Maintenance] done")
    except Exception as e:
        print(f"[Maintenance] error: {e}")
    threading.Timer(86400, nightly_maintenance).start()


# ─── system endpoints ─────────────────────────────────────────────────────────

@app.route("/ready", methods=["GET"])
def ready():
    status = scene_engine.status()
    return jsonify({
        "ready":          status["ready"],
        "zone_count":     status["zone_count"],
        "affinity_count": status["affinity_count"],
        "zones":          status["zones"],
    }), 200


@app.route("/start_experiment", methods=["POST"])
def start_experiment():
    data    = request.get_json() or {}
    db_name = data.get("db_name", CONFIG.DB_NAME)
    _switch_db(db_name)
    _experiment_state["mode"]              = data.get("experiment_mode",   "baseline")
    _experiment_state["ablation_mode"]     = data.get("ablation_mode",     "full")
    _experiment_state["collection_suffix"] = data.get("collection_suffix", "")
    _experiment_state["active"]            = True
    print(f"[Experiment] START mode={_experiment_state['mode']} "
          f"db={db_name} suffix='{_experiment_state['collection_suffix']}'")
    return jsonify({"status": "ok", **_experiment_state}), 200


@app.route("/experiment_done", methods=["POST"])
def experiment_done():
    max_wait, poll, waited = 300.0, 1.0, 0.0
    while waited < max_wait:
        if _predict_queue.empty() and not _is_processing:
            break
        time.sleep(poll)
        waited += poll
    _experiment_state["active"] = False
    print(f"[Experiment] DONE waited={waited:.1f}s")
    return jsonify({"status": "done", "waited_seconds": round(waited, 1)}), 200


# ─── robot state endpoints ────────────────────────────────────────────────────

@app.route("/nav_target",  methods=["GET"])
def get_nav_target():
    return jsonify({"nav_target": _robot_state["nav_target"],
                    "nav_label":  _robot_state["nav_label"]})


@app.route("/highlight",   methods=["GET"])
def get_highlight():
    return jsonify({"label": _robot_state["highlight"]})


@app.route("/last_answer", methods=["GET"])
def get_last_answer():
    return jsonify({"answer": _robot_state["last_answer"]})


# ─── interaction endpoints ────────────────────────────────────────────────────

@app.route("/interact/stream", methods=["POST"])
def interact_stream():
    data    = request.get_json()
    query   = data.get("query", "")
    user_id = data.get("userID", "Unknown")
    room    = data.get("room", "")
    if not query:
        return jsonify({"error": "Empty query"}), 400

    def generate():
        for event in reactive_service.process_stream(query=query, user_id=user_id, room=room):
            if event.get("type") == "done":
                _robot_state["last_answer"] = event.get("answer", "")
                _robot_state["nav_target"]  = event.get("nav_target")
                _robot_state["nav_label"]   = event.get("nav_label", "")
                _robot_state["highlight"]   = event.get("nav_label", "")
            yield f"data: {json.dumps(event)}\n\n"

    return Response(stream_with_context(generate()), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.route("/interact", methods=["POST"])
def interact():
    data    = request.get_json()
    query   = data.get("query", "")
    user_id = data.get("userID", "Unknown")
    room    = data.get("room", "")
    if not query:
        return jsonify({"error": "Empty query"}), 400
    return jsonify(reactive_service.process(query=query, user_id=user_id, room=room)), 200


@app.route("/interact/confirm", methods=["POST"])
def interact_confirm():
    data       = request.get_json()
    choice     = int(data.get("choice", 3))
    nav_target = data.get("nav_target")
    nav_label  = data.get("nav_label", "")
    if choice == 1:
        return jsonify({"status": "navigate", "nav_target": nav_target,
                        "nav_label": nav_label, "message": f"Navigating to {nav_label}."})
    if choice == 2:
        pos_str = (f"[{nav_target[0]:.1f}, {nav_target[1]:.1f}]" if nav_target else "unknown")
        return jsonify({"status": "info_only", "message": f"{nav_label} is at {pos_str}."})
    return jsonify({"status": "cancelled", "message": "Cancelled."})


# ─── service endpoints ────────────────────────────────────────────────────────

@app.route("/service_proposal", methods=["GET"])
def service_proposal():
    proposal = proposal_manager.get_next()
    return jsonify(proposal if proposal else {"status": "no_proposal"}), 200


@app.route("/service_response", methods=["POST"])
def service_response():
    data        = request.get_json()
    proposal_id = data.get("proposal_id", "")
    user_id     = data.get("user_id", "Unknown")
    result      = data.get("result", "ignored")
    if not proposal_id:
        return jsonify({"error": "proposal_id required"}), 400
    return jsonify(proposal_manager.handle_response(
        proposal_id=proposal_id, user_id=user_id, result=result)), 200


@app.route("/service_history", methods=["GET"])
def service_history():
    user_id   = request.args.get("user_id")
    proposals = proposal_manager.get_history(user_id=user_id)
    return jsonify({"proposals": proposals, "total": len(proposals)}), 200


# ─── demo endpoints ───────────────────────────────────────────────────────────

@app.route("/demo/habits", methods=["GET"])
def demo_habits():
    NO_WEIGHT        = {"PickingUp", "PuttingDown", "Walking", "Standing", "StandUp"}
    HABIT_THRESHOLD  = 5
    result           = {}
    for uid in ["User_Mom", "User_Dad"]:
        obs = list(db.observation_logs.find(
            {"user": uid, "action": {"$nin": list(NO_WEIGHT)}},
            {"action": 1, "zone_name": 1, "time_slot": 1, "weight": 1}
        ))
        agg = defaultdict(float)
        for d in obs:
            key = (d.get("action", ""), d.get("zone_name", ""), d.get("time_slot", ""))
            agg[key] += d.get("weight", 1)
        habits = sorted(
            [{"action": a, "zone": z, "slot": t, "weight": round(w, 1)}
             for (a, z, t), w in agg.items() if w >= HABIT_THRESHOLD],
            key=lambda x: -x["weight"]
        )[:8]
        trans    = list(db.transition_counts.find(
            {"user_id": uid},
            {"from_action": 1, "to_action": 1, "count": 1, "time_slot": 1, "_id": 0}
        ).sort("count", -1).limit(5))
        skill_doc = db.user_skills.find_one({"user_id": uid})
        result[uid] = {
            "habits":      habits,
            "transitions": trans,
            "skill_md":    skill_doc.get("skill_md", "") if skill_doc else "",
            "obs_count":   len(obs),
        }
    return jsonify(result), 200


@app.route("/demo/latest_har", methods=["GET"])
def demo_latest_har():
    doc = db.experiment_logs.find_one({}, sort=[("timestamp", -1)])
    if not doc:
        return jsonify({}), 200
    return jsonify({
        "user":           doc.get("user", ""),
        "spatial_action": doc.get("spatial_action", ""),
        "vlm_confidence": doc.get("vlm_confidence", 0),
        "upgrade_reason": doc.get("upgrade_reason", ""),
        "virtual_day":    doc.get("virtual_day"),
        "time_slot":      doc.get("time_slot", ""),
    }), 200


@app.route("/demo/trigger_proactive", methods=["POST"])
def demo_trigger_proactive():
    data        = request.get_json()
    user_id     = data.get("user_id",     "User_Mom")
    prev_action = data.get("prev_action", "Watching")
    time_slot   = data.get("time_slot",   "Evening")
    lookahead   = habit_learner.get_2step_lookahead(
        user_id=user_id, current_action=prev_action, time_slot=time_slot)
    cutoff = datetime.datetime.utcnow() - datetime.timedelta(hours=CONFIG.SNAPSHOT_TTL_HOURS)
    need   = lookahead.get("need", "drink") if lookahead else "drink"
    item   = db.dynamic_objects.find_one(
        {"category": need, "last_seen": {"$gte": cutoff}},
        sort=[("interact_count", -1)])
    if not item:
        item = db.dynamic_objects.find_one({"category": need}, sort=[("interact_count", -1)])
    if not item:
        return jsonify({"status": "no_item", "message": "No available item found"}), 200
    proposal    = proactive_service._generate_proposal(
        user_id=user_id,
        lookahead=lookahead or {"need": "drink", "confidence": 0.5,
                                "actionable": True, "step1": None, "step2": None},
        available_item=item, time_slot=time_slot)
    proposal_id = proposal_manager.push(user_id, proposal)
    return jsonify({
        "status":     "ok",
        "proposal_id": proposal_id,
        "message":    proposal.get("message", ""),
        "item":       item["label"],
        "item_loc":   item.get("last_seen_on", ""),
        "confidence": lookahead.get("confidence", 0) if lookahead else 0,
    }), 200


@app.route("/demo/action_event", methods=["POST"])
def demo_action_event():
    data        = request.get_json()
    user_id     = data.get("user_id",     "User_Mom")
    prev_action = data.get("prev_action", "Watching")
    time_slot   = data.get("time_slot",   "Evening")
    lookahead   = habit_learner.get_2step_lookahead(
        user_id=user_id, current_action=prev_action, time_slot=time_slot)
    cutoff = datetime.datetime.utcnow() - datetime.timedelta(hours=CONFIG.SNAPSHOT_TTL_HOURS)
    need   = lookahead.get("need", "drink") if lookahead else "drink"
    item   = db.dynamic_objects.find_one(
        {"category": need, "last_seen": {"$gte": cutoff}},
        sort=[("interact_count", -1)])
    if not item:
        item = db.dynamic_objects.find_one({"category": need}, sort=[("interact_count", -1)])
    if not item:
        return jsonify({"status": "no_item"}), 200
    proposal = proactive_service._generate_proposal(
        user_id=user_id,
        lookahead=lookahead or {"need": "drink", "confidence": 0.5,
                                "actionable": True, "step1": None, "step2": None},
        available_item=item, time_slot=time_slot)
    proposal_manager.push(user_id, proposal)
    return jsonify({"status": "ok"}), 200


@app.route("/demo/scene_ready", methods=["POST"])
def demo_scene_ready():
    data = request.get_json()
    _demo_state["current_scene"] = data.get("scene", 0)
    _demo_state["scene_done"]    = False
    _demo_state["scene_user"]    = data.get("user_id", "")
    return jsonify({"status": "ok"}), 200


@app.route("/demo/current_scene", methods=["GET"])
def demo_current_scene():
    return jsonify(_demo_state), 200


@app.route("/demo/scene_done", methods=["POST"])
def demo_scene_done():
    _demo_state["scene_done"] = True
    return jsonify({"status": "ok"}), 200


@app.route("/demo/wait_scene_done", methods=["GET"])
def demo_wait_scene_done():
    return jsonify({"done": _demo_state["scene_done"],
                    "scene": _demo_state["current_scene"]}), 200


@app.route("/demo/speaking", methods=["POST"])
def demo_speaking():
    _speaking_state["who"] = request.get_json().get("who", "none")
    return jsonify({"ok": True}), 200


@app.route("/demo/speaking_state", methods=["GET"])
def demo_speaking_state():
    return jsonify(_speaking_state), 200


@app.route("/demo/mami_status", methods=["GET"])
def demo_mami_status():
    return jsonify({
        "is_listening": _speaking_state["who"] != "none",
        "current_user": _demo_state.get("scene_user", ""),
    }), 200


# ─── predict pipeline ─────────────────────────────────────────────────────────

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data received"}), 400

    episode_id = data.get("episode_id") or str(uuid.uuid4())

    # deduplicate by episode_id in experiment_logs
    suffix = _experiment_state.get("collection_suffix", "")
    col    = f"experiment_logs{suffix}" if suffix else "experiment_logs"
    if db[col].find_one({"episode_id": episode_id}, {"_id": 1}):
        return jsonify({"status": "ok", "episode_id": episode_id, "cached": True}), 200

    t_capture    = data.get("t_capture", "")
    ground_truth = data.get("activity", "")
    if t_capture and ground_truth:
        with _gt_cache_lock:
            _gt_cache[t_capture] = ground_truth

    data["experiment_mode"]   = _experiment_state.get("mode",              "baseline")
    data["ablation_mode"]     = _experiment_state.get("ablation_mode",     "full")
    data["collection_suffix"] = _experiment_state.get("collection_suffix", "")

    print(f"[Predict] vday={data.get('virtual_day')} vhour={data.get('virtual_hour')} "
          f"user={data.get('userID')} mode={data['experiment_mode']}", flush=True)

    _predict_queue.put((episode_id, data))
    return jsonify({"status": "queued", "episode_id": episode_id}), 200


def _predict_worker():
    global _is_processing
    while True:
        try:
            episode_id, data = _predict_queue.get(timeout=1)
            _is_processing = True
            try:
                _process_predict(episode_id, data)
            except Exception as e:
                import traceback
                print(f"[PredictWorker] {e}\n{traceback.format_exc()}")
            finally:
                _is_processing = False
                _predict_queue.task_done()
        except _queue.Empty:
            continue


def _process_predict(episode_id: str, data: dict):
    image_list   = data.get("image_list", [])
    hint_user_id = data.get("userID", "Unknown_User")
    activity     = data.get("activity", "")
    user_pos_raw = data.get("user_pos")
    user_fwd_raw = data.get("user_forward")
    source_nodes = data.get("source_nodes", [])
    room_name    = data.get("room_name", "")

    vt           = _parse_virtual_time(data)
    virtual_hour = vt["virtual_hour"]
    virtual_day  = vt["virtual_day"]
    time_slot    = vt["time_slot"]
    virtual_date = vt["virtual_date"]

    if not room_name and source_nodes:
        first_node = source_nodes[0].split("_b")[0]
        cam_idx    = first_node.rfind("_Cam")
        room_name  = first_node[:cam_idx] if cam_idx > 0 else first_node

    if not image_list:
        return

    preview_images(image_list, source_nodes, hint_user_id, activity)

    est_pos = None
    if user_pos_raw:
        est_pos = {"x": float(user_pos_raw.get("x", 0)),
                   "z": float(user_pos_raw.get("z", 0))}

    est_forward = None
    if user_fwd_raw:
        est_forward = {"x": float(user_fwd_raw.get("x", 0)),
                       "y": float(user_fwd_raw.get("y", 0)),
                       "z": float(user_fwd_raw.get("z", 0))}

    if est_forward is None and hint_user_id:
        pos_doc = db.user_positions.find_one({"user_id": hint_user_id})
        if pos_doc and pos_doc.get("forward"):
            fwd = pos_doc["forward"]
            if isinstance(fwd, list) and len(fwd) >= 3:
                est_forward = {"x": fwd[0], "y": 0.0, "z": fwd[2]}
            elif isinstance(fwd, dict):
                est_forward = {"x": float(fwd.get("x", 0)), "y": 0.0,
                               "z": float(fwd.get("z", 0))}

    if est_pos and hint_user_id:
        db.user_positions.update_one(
            {"user_id": hint_user_id},
            {"$set": {"user_id": hint_user_id, "x": est_pos["x"], "z": est_pos["z"],
                      "room": room_name, "forward": est_forward,
                      "updated_at": datetime.datetime.utcnow()}},
            upsert=True)

    data["room_name"]         = room_name
    data["user_forward"]      = est_forward
    data["experiment_mode"]   = _experiment_state.get("mode",              "baseline")
    data["ablation_mode"]     = _experiment_state.get("ablation_mode",     "full")
    data["collection_suffix"] = _experiment_state.get("collection_suffix", "")

    _wait_for_scene(max_wait=12.0)

    with _vlm_lock:
        t0     = time.time()
        result = perception.analyze_action_burst(data)
        vlm_ms = int((time.time() - t0) * 1000)

    _user_id       = result.get("user", "") or hint_user_id
    _exp_mode      = result.get("experiment_mode", "baseline")
    spatial_action = result.get("spatial_action") or result.get("action", "Unknown")
    zone_label     = result.get("zone_label") or ""

    print(f"[VLM] spatial={spatial_action} | {vlm_ms}ms "
          f"day={virtual_day} slot={time_slot} mode={_exp_mode}")

    if spatial_action in ("none", "", "Unknown"):
        return

    NO_RECORD = {"Walking", "Standing", "StandUp", "PickingUp", "PuttingDown"}

    if zone_label and spatial_action not in NO_RECORD and _exp_mode == "baseline":
        pos_xy = [(est_pos["x"] if est_pos else 0) / 10.0,
                  (est_pos["z"] if est_pos else 0) / 10.0]
        observation_store.record(
            user_id=_user_id, action=spatial_action,
            zone_name=zone_label, instance=zone_label,
            time_slot=time_slot,
            interacting_items=result.get("items", []),
            spatial_relations=result.get("spatial_relations", {}),
            pos_xy=pos_xy, room=result.get("room", ""),
            raw_desc=result.get("result", {}).get("context", ""),
            today=virtual_date)

        prev_seq    = observation_store.get_recent_sequence(_user_id, limit=2)
        prev_action = prev_seq[-2] if len(prev_seq) >= 2 else "Standing"

        habit_learner.on_new_observation(
            user_id=_user_id, action=spatial_action,
            prev_action=prev_action, time_slot=time_slot,
            zone_name=zone_label)

        scene_engine.update_user_affinity(
            user_id=_user_id, zone_name=zone_label,
            action=spatial_action, room=result.get("room", ""),
            virtual_day=virtual_day)

    prev_seq    = observation_store.get_recent_sequence(_user_id, limit=2)
    prev_action = prev_seq[-2] if len(prev_seq) >= 2 else "Standing"

    if (spatial_action == "Standing" and
            prev_action not in ("Standing", "Walking") and
            _exp_mode == "baseline"):
        proposal = proactive_service.evaluate(
            user_id=_user_id, current_action=spatial_action,
            prev_action=prev_action, time_slot=time_slot, user_pos=est_pos)
        if proposal:
            proposal_manager.push(_user_id, proposal)

    bound_label = result.get("bound_instance", "Unknown_Area")
    final_bound = bound_label if "Unknown" not in bound_label else ""
    if not final_bound and est_pos:
        best_doc, best_dist = None, float("inf")
        query_r = {"room": {"$regex": room_name, "$options": "i"}} if room_name else {}
        for doc in db.scene_snapshots.find(query_r, {"label": 1, "pos": 1}):
            pos = doc.get("pos")
            if not isinstance(pos, list) or len(pos) < 2:
                continue
            dist = math.sqrt((est_pos["x"] - pos[0]) ** 2 + (est_pos["z"] - pos[1]) ** 2)
            if dist < best_dist:
                best_dist, best_doc = dist, doc
        if best_doc and best_dist <= 5.0:
            final_bound = best_doc["label"]
    final_bound = final_bound or "Unknown_Area"

    try:
        db.activity_sequences.update_one(
            {"user": _user_id, "date": virtual_date},
            {"$push": {"sequence": {"action": spatial_action, "instance": final_bound,
                                    "timestamp": datetime.datetime.utcnow()}},
             "$setOnInsert": {"user": _user_id, "date": virtual_date}},
            upsert=True)
    except Exception as e:
        print(f"[Sequence] {e}")

    print(f"[Predict] done | {_user_id} | spatial={spatial_action} | "
          f"{vlm_ms}ms | mode={_exp_mode}")


# ─── scene & object endpoints ─────────────────────────────────────────────────

@app.route("/scene", methods=["POST"])
def handle_scene():
    try:
        data    = request.get_json()
        objects = data.get("objects", [])
        if not objects:
            return jsonify({"status": "empty"}), 200
        now  = datetime.datetime.utcnow()
        docs = []
        for obj in objects:
            label = obj.get("label", "").lower().strip()
            if not label:
                continue
            if any(kw in label for kw in BASE_FURNITURE_KEYWORDS):
                db.scene_snapshots.update_one(
                    {"label": label},
                    {"$set": {"label": label,
                              "pos":   [obj.get("x", 0), obj.get("z", 0)],
                              "x": obj.get("x", 0), "y": obj.get("y", 0),
                              "z": obj.get("z", 0), "room": obj.get("room", ""),
                              "source": obj.get("source", "sensor"),
                              "last_updated": now, "is_static": True}},
                    upsert=True)
            docs.append({"label": label,
                         "x": obj.get("x", 0), "y": obj.get("y", 0),
                         "z": obj.get("z", 0), "room": obj.get("room", ""),
                         "source": obj.get("source", "sensor"),
                         "held_by": obj.get("held_by", ""),
                         "processed": False, "received_at": now})
        if docs:
            db.raw_objects.insert_many(docs)
        scene_engine.build()
        print(f"[Scene] received {len(docs)} objects")
        return jsonify({"status": "Success", "received": len(docs)}), 200
    except Exception as e:
        print(f"[Scene Error] {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/dynamic_sync", methods=["POST"])
def dynamic_sync():
    try:
        data    = request.get_json()
        objects = data.get("objects", [])
        if not objects:
            return jsonify({"status": "empty"}), 200
        now, count = datetime.datetime.utcnow(), 0
        for obj in objects:
            label  = obj.get("label", "").lower().strip()
            source = obj.get("source", "sensor")
            if not label:
                continue
            obj_vt = _parse_virtual_time(obj)
            if source == "unity_user":
                position = obj.get("position", [0, 0])
                db.user_positions.update_one(
                    {"user_id": label},
                    {"$set": {"user_id": label,
                              "x": float(position[0]), "z": float(position[1]),
                              "forward": obj.get("forward", [0, 0, 0]),
                              "activity": obj.get("activity", ""),
                              "room": obj.get("room", ""),
                              "virtual_hour": obj_vt["virtual_hour"],
                              "virtual_day":  obj_vt["virtual_day"],
                              "time_slot":    obj_vt["time_slot"],
                              "updated_at":   now}},
                    upsert=True)
                count += 1
                continue
            room         = obj.get("room", "")
            position     = obj.get("position", [0, 0])
            x, z         = float(position[0]), float(position[1])
            held_by      = obj.get("held_by", "")
            last_seen_on = _find_nearest_furniture(x, z, room)
            set_fields   = {
                "room": room, "sensor_pos": position,
                "last_seen": now, "status": "active", "status_since": now,
                "source": source, "category": _get_category_for_label(label),
                "virtual_hour": obj_vt["virtual_hour"],
                "virtual_day":  obj_vt["virtual_day"],
                "time_slot":    obj_vt["time_slot"],
            }
            if held_by:
                set_fields["held_by"] = held_by
                existing = db.dynamic_objects.find_one({"label": label}, {"held_by": 1})
                if not existing or existing.get("held_by") != held_by:
                    set_fields["held_since"] = now
            else:
                set_fields["last_seen_on"] = last_seen_on
            db.dynamic_objects.update_one(
                {"label": label},
                {"$set": set_fields,
                 "$inc": {"seen_count": 1},
                 "$setOnInsert": {"first_seen": now, "spatial_rel": "on"}},
                upsert=True)
            count += 1
        unity_labels = [
            obj.get("label", "").lower().strip()
            for obj in objects
            if obj.get("source") == "unity" and obj.get("label", "").strip()
        ]
        if unity_labels:
            stale = db.dynamic_objects.delete_many(
                {"source": "unity", "label": {"$nin": unity_labels}})
            if stale.deleted_count > 0:
                print(f"[DynamicSync] Removed {stale.deleted_count} stale objects")
        return jsonify({"status": "Success", "updated": count}), 200
    except Exception as e:
        import traceback
        print(f"[DynamicSync Error] {e}\n{traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500


@app.route("/set_device_state", methods=["POST"])
def set_device_state():
    data  = request.get_json()
    label = data.get("label", "")
    state = data.get("state", "off")
    if label:
        db.device_states.update_one(
            {"label": label},
            {"$set": {"state": state, "updated_at": datetime.datetime.utcnow()}},
            upsert=True)
    return jsonify({"status": "ok"}), 200


@app.route("/object_event", methods=["POST"])
def object_event():
    data        = request.get_json()
    user_id     = data.get("user_id")
    object_name = data.get("object")
    now         = datetime.datetime.utcnow()
    if "pickup_time" in data:
        try:
            pickup_time = datetime.datetime.fromisoformat(
                data["pickup_time"].replace("Z", "+00:00")).replace(tzinfo=None)
        except Exception:
            pickup_time = now
        db.object_events.insert_one({"user": user_id, "object": object_name,
                                     "pickup_time": pickup_time, "putdown_time": None})
        print(f"[ObjectEvent] {user_id} picked up {object_name}")
    elif "putdown_time" in data:
        try:
            putdown_time = datetime.datetime.fromisoformat(
                data["putdown_time"].replace("Z", "+00:00")).replace(tzinfo=None)
        except Exception:
            putdown_time = now
        db.object_events.find_one_and_update(
            {"user": user_id, "putdown_time": None},
            {"$set": {"putdown_time": putdown_time}},
            sort=[("pickup_time", -1)])
        print(f"[ObjectEvent] {user_id} put down")
    return jsonify({"status": "ok"}), 200


@app.route("/track_position", methods=["POST"])
def track_position():
    try:
        data    = request.get_json()
        user_id = data.get("userID", "Unknown")
        x, z    = float(data.get("x", 0)), float(data.get("z", 0))
        db.user_positions.update_one(
            {"user_id": user_id},
            {"$set": {"user_id": user_id, "x": x, "z": z,
                      "room": data.get("room_name", ""),
                      "forward": [float(data.get("forward_x", 0)), 0.0,
                                  float(data.get("forward_z", 0))],
                      "updated_at": datetime.datetime.utcnow()}},
            upsert=True)
        return jsonify({"status": "ok"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/set_virtual_hour", methods=["POST"])
def set_virtual_hour():
    try:
        data = request.get_json(force=True, silent=True) or {}
        hour = data.get("virtual_hour", -1)
        app.config["VIRTUAL_HOUR"] = float(hour)
        db.system_config.update_one(
            {"key": "virtual_hour"}, {"$set": {"value": float(hour)}}, upsert=True)
        return jsonify({"status": "ok", "virtual_hour": hour}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/log_navigation", methods=["POST"])
def log_navigation():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data"}), 400
        db.navigation_logs.insert_one({
            "user_id": data.get("user_id"), "intent": data.get("intent"),
            "start_pos": data.get("start_pos"), "goal_pos": data.get("goal_pos"),
            "waypoints": data.get("waypoints", []), "success": data.get("success", False),
            "total_time": data.get("total_time", 0),
            "total_distance": data.get("total_distance", 0),
            "timestamp": datetime.datetime.utcnow()})
        return jsonify({"status": "Success"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ─── startup ──────────────────────────────────────────────────────────────────

_predict_worker_thread = threading.Thread(target=_predict_worker, daemon=True)
_predict_worker_thread.start()
print("[Predict Queue] worker started")

if __name__ == "__main__":
    host = getattr(CONFIG, "FLASK_HOST", "0.0.0.0")
    port = int(getattr(CONFIG, "FLASK_PORT", 5000))
    print(f"\n[App] Robot Brain Server on {host}:{port}")
    print(f"  Perception : {CONFIG.VLM_MODEL} + {CONFIG.LLM_MODEL}")
    print(f"  Memory     : ObservationStore + HabitLearner")
    print(f"  Service    : ProactiveService + ReactiveService")
    threading.Timer(86400, nightly_maintenance).start()
    app.run(host=host, port=port, debug=False)