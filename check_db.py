from pymongo import MongoClient
from config import Config

def check_database():
    client = MongoClient(Config.MONGO_URI)
    db = client[Config.DB_NAME]
    
    # 1. 檢查家具快照 (Scene Snapshots)
    print("\n=== [家具快照資料庫] ===")
    scene_count = db.scene_snapshots.count_documents({})
    print(f"總共有 {scene_count} 個家具物件。")
    
    if scene_count > 0:
        print("最新同步的 3 筆家具：")
        for doc in db.scene_snapshots.find().limit(3):
            print(f"- ID: {doc.get('id')}, 標籤: {doc.get('label')}, 座標: {doc.get('pos')}")

    # 2. 檢查觀察日誌 (Observation Logs)
    print("\n=== [行為觀察日誌] ===")
    log_count = db.observation_logs.count_documents({})
    print(f"總共有 {log_count} 筆行為紀錄 (無感學習數據)。")
    
    if log_count > 0:
        for log in db.observation_logs.find().sort("last_seen", -1).limit(5):
            print(f"- 使用者: {log.get('user')}, 動作: {log.get('action')}, 位置: {log.get('instance')}, 權重: {log.get('weight')}")

if __name__ == "__main__":
    check_database()