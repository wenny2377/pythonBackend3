from pymongo import MongoClient
from config import Config

def reset_database_to_2d():
    print("--- 正在重置 MongoDB 索引為 2D 平面模式 ---")
    client = MongoClient(Config.MONGO_URI)
    db = client[Config.DB_NAME]
    
    # 1. 清除舊的家具資料與紀錄 (避免格式衝突)
    db.scene_snapshots.drop()
    db.observation_logs.drop()
    print("🗑️  舊資料已清空 (scene_snapshots & observation_logs)")

    # 2. 建立正確的 2d 索引
    # 注意：這裡必須是 "2d"，不是 "2dsphere"
    db.scene_snapshots.create_index([("pos", "2d")])
    print("✅  2D 平面索引建立成功！")

    # 3. 列出索引確認
    print("\n目前的索引狀態:")
    for idx in db.scene_snapshots.list_indexes():
        print(f" - 名稱: {idx['name']}, 鍵值: {idx['key']}")

if __name__ == "__main__":
    reset_database_to_2d()