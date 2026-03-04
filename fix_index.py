from pymongo import MongoClient
from config import Config

def force_fix_index():
    client = MongoClient(Config.MONGO_URI)
    db = client[Config.DB_NAME]
    collection = db["scene_snapshots"]
    
    print(f"--- 正在檢查資料庫: {Config.DB_NAME} ---")
    
    # 1. 清除所有舊索引 (除了 _id)
    try:
        collection.drop_indexes()
        print("🗑️  舊索引已清理")
    except Exception as e:
        print(f"⚠️  清理索引時出錯 (可能原本就沒索引): {e}")

    # 2. 強制建立 2dsphere 空間索引
    try:
        # 注意：欄位名稱必須與你 memory.py 存入的一致，是 "pos"
        result = collection.create_index([("pos", "2dsphere")])
        print(f"✅ 成功建立空間索引: {result}")
    except Exception as e:
        print(f"❌ 建立索引失敗！錯誤訊息: {e}")
        print("\n💡 小提醒：如果報錯 'location object expected'，代表你資料庫裡有些舊家具的 pos 格式寫錯了。")
        print("建議進 mongosh 執行 db.scene_snapshots.drop() 徹底清空家具表後再同步一次。")

    # 3. 列出目前所有索引確認
    print("\n目前的索引清單:")
    for index in collection.list_indexes():
        print(f" - {index['name']}: {index['key']}")

if __name__ == "__main__":
    force_fix_index()