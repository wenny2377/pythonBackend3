from pymongo import MongoClient

# 請確保資料庫名稱與你的設定一致
DB_NAME = "your_database_name"  # <--- 這裡改成你的資料庫名稱，例如 robot_brain 或 test

client = MongoClient("mongodb://localhost:27017/")
db = client[DB_NAME]
col_scene = db["scene_snapshots"]

print(f"--- 目前資料庫 [{DB_NAME}] 中的家具清單 ---")
cursor = col_scene.find({})

count = 0
for doc in cursor:
    count += 1
    label = doc.get("label", "N/A")
    x = doc.get("x", 0)
    z = doc.get("z", 0)
    room = doc.get("room", "N/A")
    print(f"[{count}] 標籤: {label:<15} | 房間: {room:<10} | 座標: ({x}, {z})")

if count == 0:
    print("❌ 目前資料庫中沒有任何家具資料！")
print("---------------------------------------")