"""
reset_all.py
開發用完全重置腳本
執行方式：python reset_all.py
"""

import os
import shutil
from pymongo import MongoClient

MONGO_URI = "mongodb://127.0.0.1:27017/"
DB_NAME   = "robot_rag_db"

client = MongoClient(MONGO_URI)

print("\n📋 MongoDB 現有資料庫：", client.list_database_names())
print(f"🎯 目標 DB：{DB_NAME}\n")

db = client[DB_NAME]

print("📊 清除前各 collection 筆數：")
for col_name in db.list_collection_names():
    print(f"  [{col_name}] {db[col_name].count_documents({})} 筆")

print("\n⚠️  開始清空...\n")

cols = [
    "eval_logs",
    "observation_logs",
    "exp_checkpoints",
    "activity_sequences",
    "conversation_logs",
    "raw_objects",
    "dynamic_objects",
    "scene_snapshots",
    "semantic_memories",
    "navigation_logs",
    # ── 新增：Manifold 相關 ──
    "manifold_points",
    "behavior_clusters",
    "service_proposals",
    "intent_stats",
]
for col in cols:
    n = db[col].delete_many({}).deleted_count
    print(f"  [{col}] 刪除 {n} 筆")

# ── FAISS 檔案 ──
faiss_files = [
    "robot_memory.index",
    "robot_memory_meta.json",
    "dynamic_memory.index",
    "dynamic_memory_meta.json",
]
for path in faiss_files:
    if os.path.exists(path):
        os.remove(path)
        print(f"  [FAISS] 刪除 {path}")
    else:
        print(f"  [FAISS] 不存在（跳過）{path}")

# ── debug_images ──
if os.path.exists("debug_images"):
    shutil.rmtree("debug_images")
    print("  [debug_images] 已清空")

print("\n✅ 完成，重啟 Flask：python app.py\n")