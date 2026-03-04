import torch

class Config:
    # --- Server Settings ---
    FLASK_HOST = "0.0.0.0"
    FLASK_PORT = 5000
    
    # --- Ollama / VLM Settings ---
    OLLAMA_URL = "http://localhost:11434"  # PerceptionEngine 通常接 /api/generate
    # 🚀 改成 llava-phi3，適合視覺語言任務
    OLLAMA_MODEL = "llava-phi3:latest"  
    
    # --- MongoDB Settings ---
    MONGO_URI = "mongodb://127.0.0.1:27017/"
    DB_NAME = "robot_rag_db"
    
    # --- Spatial & MiDaS Settings ---
    MIDAS_MODEL_TYPE = "MiDaS_small"  # Demo 快速分析用
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 逆投影校準係數 (Scale Factor)
    DEPTH_SCALE = 0.5

    # 顯示目前運算裝置，方便除錯
    print(f"🖥️  System initialization... Running on: {DEVICE}")