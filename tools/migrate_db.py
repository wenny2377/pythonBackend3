"""
migrate_db.py
─────────────
Database initialization and migration tool.

Run once before starting the system:
  python3 tools/migrate_db.py

What it does:
  1. Creates TTL index on observation_logs (14 days)
  2. Creates index on transition_counts
  3. Initializes episodic_summaries and skill_chunks collections
  4. Builds FAISS skill chunk index from existing SKILL.md profiles
  5. Verifies collection state
"""

from pymongo import MongoClient, ASCENDING
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json
import re
import os
from datetime import datetime

MONGO_URI        = "mongodb://127.0.0.1:27017/"
DB_NAME          = "robot_rag_db"
TTL_DAYS         = 14
SBERT_MODEL      = "paraphrase-MiniLM-L6-v2"
DEDUP_THRESHOLD  = 0.85

client = MongoClient(MONGO_URI)
db     = client[DB_NAME]


def step1_create_indexes():
    print("\n[Step 1] Creating indexes...")
    try:
        # TTL index for observation_logs
        existing   = db.observation_logs.index_information()
        ttl_exists = any(
            "expireAfterSeconds" in str(info)
            for info in existing.values()
        )
        if not ttl_exists:
            db.observation_logs.create_index(
                [("last_seen", ASCENDING)],
                expireAfterSeconds=TTL_DAYS * 86400,
                name="observation_ttl_14d",
            )
            print(f"  TTL index created ({TTL_DAYS} days)")
        else:
            print("  TTL index already exists")

        # Transition counts index
        db.transition_counts.create_index(
            [("user_id", ASCENDING),
             ("from_action", ASCENDING),
             ("time_slot", ASCENDING)],
            name="transition_lookup",
        )
        print("  transition_counts index created")

    except Exception as e:
        print(f"  Index error: {e}")


def step2_create_collections():
    print("\n[Step 2] Initializing collections...")
    collections = db.list_collection_names()

    needed = [
        "observation_logs", "habit_snapshots", "activity_sequences",
        "transition_counts", "service_proposals", "object_events",
        "episodic_summaries", "skill_chunks", "user_skills",
        "manifold_training_data", "eval_logs",
    ]
    for col_name in needed:
        if col_name not in collections:
            db.create_collection(col_name)
            print(f"  Created: {col_name}")
        else:
            count = db[col_name].count_documents({})
            print(f"  Exists:  {col_name} ({count} records)")


def step3_build_skill_faiss():
    print("\n[Step 3] Building FAISS skill chunk index...")

    skill_docs = list(db.user_skills.find({}, {"user_id": 1, "skill_md": 1}))
    if not skill_docs:
        print("  No SKILL.md found. Run the system first.")
        return

    print(f"  Found {len(skill_docs)} user profiles")
    sbert = SentenceTransformer(SBERT_MODEL, device='cuda')
    dim   = 384
    index = faiss.IndexFlatIP(dim)
    meta  = []

    for skill_doc in skill_docs:
        user_id  = skill_doc.get("user_id", "unknown")
        skill_md = skill_doc.get("skill_md", "")
        if not skill_md:
            continue

        sections = re.split(r'\n(?=## )', skill_md)
        chunks   = 0
        for section in sections:
            section = section.strip()
            if len(section) < 10:
                continue

            title_match = re.match(r'## (.+)', section)
            title       = title_match.group(1).strip() if title_match else "general"
            vec         = sbert.encode(
                section, normalize_embeddings=True).astype(np.float32)

            db.skill_chunks.replace_one(
                {"user_id": user_id, "title": title},
                {
                    "user_id":    user_id,
                    "title":      title,
                    "content":    section,
                    "vector":     vec.tolist(),
                    "support":    1,
                    "updated_at": datetime.utcnow(),
                },
                upsert=True,
            )

            vec_norm = vec.reshape(1, -1).copy()
            faiss.normalize_L2(vec_norm)
            index.add(vec_norm)
            meta.append({"user_id": user_id, "title": title})
            chunks += 1

        print(f"  {user_id}: {chunks} chunks")

    if index.ntotal > 0:
        faiss.write_index(index, "skill_chunks.index")
        with open("skill_chunks_meta.json", "w") as f:
            json.dump(meta, f, indent=2)
        print(f"\n  FAISS index saved ({index.ntotal} chunks)")


def step4_verify():
    print("\n[Step 4] Verifying state...")
    collections = [
        ("observation_logs",   "Layer 1 — raw observations (TTL 14d)"),
        ("habit_snapshots",    "Layer 1 — daily habit snapshots"),
        ("activity_sequences", "Layer 1 — action sequences"),
        ("transition_counts",  "Layer 2 — learned transitions"),
        ("service_proposals",  "Service — proactive proposals"),
        ("object_events",      "Perception — pickup/putdown events"),
        ("user_skills",        "Layer 3 — SKILL.md profiles"),
        ("skill_chunks",       "Layer 3 — FAISS chunks"),
        ("eval_logs",          "Experiment — evaluation logs"),
    ]
    for col, desc in collections:
        count = db[col].count_documents({})
        print(f"  {col:25s} | {desc:40s} | {count}")


if __name__ == "__main__":
    print("=" * 60)
    print("migrate_db.py — Robot Brain DB initialization")
    print("=" * 60)
    step1_create_indexes()
    step2_create_collections()
    step3_build_skill_faiss()
    step4_verify()
    print("\n[Done] Database ready.")