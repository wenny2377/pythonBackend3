import torch

class Config:
    # ── Server ──
    FLASK_HOST = "0.0.0.0"
    FLASK_PORT = 5000

    # ── Ollama / VLM ──
    OLLAMA_URL   = "http://localhost:11434"
    OLLAMA_MODEL = "llama3.1:8b"       # 向下相容保留

    VLM_MODEL = "llava-phi3"         # 視覺辨識 → perception.py
    LLM_MODEL = "llama3.1:8b"           # 語言推理 → interaction.py / RAG

    # ── MongoDB ──
    MONGO_URI = "mongodb://127.0.0.1:27017/"
    DB_NAME   = "robot_rag_db"

    # ── Spatial / MiDaS（保留，定點相機模式下不啟用）──
    MIDAS_MODEL_TYPE = "MiDaS_small"
    DEVICE           = "cuda" if torch.cuda.is_available() else "cpu"
    DEPTH_SCALE      = 0.5

    # ── FAISS ──
    FAISS_INDEX_PATH  = "robot_memory.index"
    FAISS_META_PATH   = "robot_memory_meta.json"
    MAX_FAISS_VECTORS = 5000

    # ── Cleanup ──
    CLEANUP_RETAIN_DAYS    = 90    # semantic_memories / activity_sequences 保留天數
    CLEANUP_INTERVAL_HOURS = 24    # 自動清理間隔（小時）

    # ── Habit Weight Decay ──
    HABIT_DECAY_FACTOR = 0.95      # 每次清理乘以此係數（0.95 = 每次衰減 5%）
    HABIT_MIN_WEIGHT   = 1.0       # weight 低於此值視為遺忘，刪除

print(f"  System initialization... Running on: {Config.DEVICE}")