import torch


class Config:
    FLASK_HOST = "0.0.0.0"
    FLASK_PORT = 5000

    OLLAMA_URL = "http://localhost:11434"
    VLM_MODEL  = "gemma3:4b"
    LLM_MODEL  = "llama3.1:8b"

    MONGO_URI = "mongodb://127.0.0.1:27017/"
    DB_NAME   = "robot_rag_db"

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    FAISS_INDEX_PATH         = "robot_memory.index"
    FAISS_META_PATH          = "robot_memory_meta.json"
    DYNAMIC_INDEX_PATH       = "dynamic_memory.index"
    DYNAMIC_META_PATH        = "dynamic_memory_meta.json"
    SKILL_CHUNK_INDEX_PATH   = "skill_chunks.index"
    SKILL_CHUNK_META_PATH    = "skill_chunks_meta.json"
    MAX_FAISS_VECTORS        = 5000

    LLM_TEMPERATURE  = 0.3
    LLM_MAX_TOKENS   = 500
    LLM_TIMEOUT      = 60

    SNAPSHOT_TTL_HOURS = 2
    SNAPSHOT_MAX_ITEMS = 30

    HABIT_DECAY_FACTOR = 0.95
    HABIT_MIN_WEIGHT   = 1.0
    OBSERVATION_TTL_DAYS = 14

    CLEANUP_RETAIN_DAYS    = 90
    CLEANUP_INTERVAL_HOURS = 24

    DEPTH_SCALE      = 0.5
    MIDAS_MODEL_TYPE = "MiDaS_small"


print(f"System init: device={Config.DEVICE} | LLM={Config.LLM_MODEL} | VLM={Config.VLM_MODEL}")