import os
import torch


def _resolve_db() -> str:
    env_db = os.environ.get("DB_NAME", "").strip()
    if env_db:
        print(f"[Config] DB_NAME from env: {env_db}")
        return env_db
    print("[Config] Defaulting to robot_exp_baseline")
    return "robot_exp_baseline"

def _resolve_system_mode() -> str:
    mode = os.environ.get("SYSTEM_MODE", "semantic").strip()
    if mode not in ("semantic", "vlm_som"):
        print(f"[Config] Unknown SYSTEM_MODE={mode}, defaulting to semantic")
        return "semantic"
    return mode


class Config:
    FLASK_HOST = "0.0.0.0"
    FLASK_PORT = 5000

    OLLAMA_URL = "http://localhost:11434"
    VLM_MODEL  = "gemma3:4b"
    LLM_MODEL  = "llama3.1:8b"

    MONGO_URI = "mongodb://127.0.0.1:27017/"
    DB_NAME      = _resolve_db()
    SYSTEM_MODE  = _resolve_system_mode()

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    FAISS_INDEX_PATH       = "robot_memory.index"
    FAISS_META_PATH        = "robot_memory_meta.json"
    DYNAMIC_INDEX_PATH     = "dynamic_memory.index"
    DYNAMIC_META_PATH      = "dynamic_memory_meta.json"
    SKILL_CHUNK_INDEX_PATH = "skill_chunks.index"
    SKILL_CHUNK_META_PATH  = "skill_chunks_meta.json"

    LLM_TEMPERATURE = 0.3
    LLM_MAX_TOKENS  = 500
    LLM_TIMEOUT     = 60
    VLM_TIMEOUT     = 120
    VLM_MAX_RETRIES = 2
    VLM_RETRY_DELAY = 2.0

    SNAPSHOT_TTL_HOURS = 2
    SNAPSHOT_MAX_ITEMS = 30

    HABIT_DECAY_FACTOR   = 0.95
    HABIT_MIN_WEIGHT     = 1.0
    OBSERVATION_TTL_DAYS = 14

    OBJECT_CONFUSION_ENABLED = False

    NORMALIZE_THRESHOLD      = 0.38
    SEMANTIC_THRESHOLD       = 0.35
    COORD_VERIFY_DIST        = 2.0
    COORD_MATCH_DIST         = 1.5
    BULK_WRITE_THRESHOLD     = 20
    BULK_WRITE_INTERVAL      = 30.0
    NEARBY_OBJECT_RADIUS     = 2.0
    HEADING_THRESHOLD        = 0.55
    VLM_CONFIDENCE_THRESHOLD = 0.50
    MIN_WRITE_CONFIDENCE     = 0.20

    DELTA_THRESHOLD      = 0.30
    CROSS_ROOM_GAMMA     = 10.0
    BASE_MASS_CH12       = 1.0
    BASE_MASS_CH3        = 1.2
    BASE_MASS_WEAK       = 0.5
    MAX_ZONE_SEARCH      = 5.0
    SCENE_RETRY_INTERVAL = 5.0
    SCENE_RETRY_MAX      = 60

    ENTROPY_HIGH_THRESHOLD   = 1.2
    ENTROPY_LOW_THRESHOLD    = 0.4
    ENTROPY_VLM_WEIGHT_HIGH  = 0.10
    ENTROPY_VLM_WEIGHT_LOW   = 0.30

    SAYCAN_ENV_FALLBACK   = 0.30
    SAYCAN_MIN_GATE_SCORE = 0.05

    MANIFOLD_MIN_TRAIN_SAMPLE = 20
    MANIFOLD_AUGMENT_FACTOR   = 100
    MANIFOLD_RETRAIN_EVERY    = 20
    MANIFOLD_TIME_NOISE_STD   = 0.5 / 24
    MANIFOLD_POS_NOISE_STD    = 0.05
    MANIFOLD_MIN_CONFIDENCE   = 0.60

    DEFINITIONS_YAML = "config/definitions.yaml"
    OBJECTS_YAML     = "config/objects.yaml"


print(f"[Config] DB={Config.DB_NAME} | device={Config.DEVICE} "
      f"| LLM={Config.LLM_MODEL} | VLM={Config.VLM_MODEL}")