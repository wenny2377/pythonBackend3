import os
import shutil
from pymongo import MongoClient
from config import Config

def clean_all():
    print("🧹 開始清理實驗數據...")

    # 1. 清理 MongoDB
    try:
        client = MongoClient(Config.MONGO_URI)
        db_name = Config.DB_NAME
        client.drop_database(db_name)
        print(f"✅ MongoDB 資料庫 '{db_name}' 已完全刪除")
    except Exception as e:
        print(f"❌ MongoDB 清理失敗: {e}")

    # 2. 清理 FAISS 索引與向量記憶檔案
    # 這裡檢查常見的向量存檔名稱，請根據你 VectorMemory 的設定調整
    files_to_remove = [
        "vector_index.faiss", 
        "vector_meta.pkl", 
        "memory_index.bin",
        "vector_store.pkl"
    ]
    
    for file in files_to_remove:
        if os.path.exists(file):
            try:
                os.remove(file)
                print(f"✅ 已刪除向量記憶檔案: {file}")
            except Exception as e:
                print(f"❌ 無法刪除 {file}: {e}")

    # 3. 如果你有存儲機器人拍的照片紀錄 (Optional)
    capture_dir = "./captures"
    if os.path.exists(capture_dir):
        shutil.rmtree(capture_dir)
        os.makedirs(capture_dir)
        print("✅ 已清空機器人拍攝的照片紀錄快取")

    print("\n✨ 系統已恢復至「零記憶」狀態，可以開始全新的 Demo 了！")

if __name__ == "__main__":
    clean_all()