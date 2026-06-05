from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
from modules.manifold_engine import ManifoldEngine

MONGO_URI = "mongodb://127.0.0.1:27017/"
DB_NAME   = "robot_rag_db"
USERS     = ["User_Mom", "User_Dad"]
MIN_SAMPLES = 20

db    = MongoClient(MONGO_URI)[DB_NAME]
sbert = SentenceTransformer("all-MiniLM-L6-v2")
me    = ManifoldEngine(db=db, sbert_model=sbert)

print(f"Connected → {DB_NAME}\n")

for uid in USERS:
    n = db.manifold_training_data.count_documents({"user_id": uid})
    print(f"{uid}: {n} samples", end="")
    if n >= MIN_SAMPLES:
        print(" → training...")
        me.train_model(uid)
    else:
        print(f" → skipped (need >= {MIN_SAMPLES})")

print("\nDone.")