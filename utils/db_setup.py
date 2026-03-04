from pymongo import MongoClient
import sys
import os

# 確保可以引入上一層的 config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import Config

def initialize_db():
    client = MongoClient(Config.MONGO_URI)
    db = client[Config.DB_NAME]
    
    print(f"--- 正在初始化資料庫: {Config.DB_NAME} ---")
    
    # 1. 為場景快照建立空間索引
    # 我們假設家具的座標存儲在 'pos' 欄位中
    db.scene_snapshots.create_index([("pos", "2dsphere")])
    print("✅ 已建立 scene_snapshots 的 2dsphere 空間索引")
    
    # 2. 為觀察日誌建立複合索引，優化查詢速度
    db.observation_logs.create_index([("user", 1), ("instance", 1)])
    print("✅ 已建立 observation_logs 的複合查詢索引")
    
    print("--- 資料庫環境準備完畢 ---")

if __name__ == "__main__":
    initialize_db()