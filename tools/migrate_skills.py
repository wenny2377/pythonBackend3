from pymongo import MongoClient, ASCENDING
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json
import re
import os
from datetime import datetime

MONGO_URI         = "mongodb://127.0.0.1:27017/"
DB_NAME           = "robot_rag_db"
TTL_DAYS          = 14
TOP_K             = 2
TARGET_TOKENS     = 400
SBERT_MODEL       = "paraphrase-MiniLM-L6-v2"
CHUNK_INDEX_PATH  = "skill_chunks.index"
CHUNK_META_PATH   = "skill_chunks_meta.json"
DEDUP_THRESHOLD   = 0.85

client = MongoClient(MONGO_URI)
db     = client[DB_NAME]


def step1_create_ttl_index():
    print("\n[Step 1] Creating observation_logs TTL index...")
    try:
        existing   = db.observation_logs.index_information()
        ttl_exists = any(
            "expireAfterSeconds" in str(info)
            for info in existing.values()
        )
        if ttl_exists:
            print("  TTL index already exists, skipping")
            return

        db.observation_logs.create_index(
            [("last_seen", ASCENDING)],
            expireAfterSeconds=TTL_DAYS * 86400,
            name="observation_ttl_14d",
        )
        print(f"  TTL index created ({TTL_DAYS} days)")
    except Exception as e:
        print(f"  TTL index failed: {e}")


def step2_create_collections():
    print("\n[Step 2] Initializing episodic_summaries and skill_chunks collections...")
    try:
        collections = db.list_collection_names()

        if "episodic_summaries" not in collections:
            db.create_collection("episodic_summaries")
            db.episodic_summaries.create_index([("user", ASCENDING)])
            db.episodic_summaries.create_index([("count", ASCENDING)])
            db.episodic_summaries.insert_one({
                "user":               "User_Mom",
                "original_request":   "coffee",
                "chosen_alternative": "tea",
                "context":            "User asked for coffee but coffee was unavailable",
                "count":              3,
                "confidence":         0.85,
                "source":             "observation_inference",
                "created_at":         datetime.utcnow(),
            })
            print("  episodic_summaries created with demo record (coffee->tea)")
        else:
            count = db.episodic_summaries.count_documents({})
            print(f"  episodic_summaries exists ({count} records)")

        if "skill_chunks" not in collections:
            db.create_collection("skill_chunks")
            db.skill_chunks.create_index([("user_id", ASCENDING)])
            db.skill_chunks.create_index([("support", ASCENDING)])
            print("  skill_chunks created")
        else:
            count = db.skill_chunks.count_documents({})
            print(f"  skill_chunks exists ({count} records)")

    except Exception as e:
        print(f"  Error: {e}")


def step3_chunk_all_skills():
    print("\n[Step 3] Building FAISS skill chunk index...")

    skill_docs = list(db.user_skills.find({}, {"user_id": 1, "skill_md": 1}))
    if not skill_docs:
        print("  No SKILL.md found. Run the system first to generate skill profiles.")
        return None, 0

    print(f"  Found {len(skill_docs)} user skill profiles")
    print(f"  Loading SBERT ({SBERT_MODEL})...")
    sbert = SentenceTransformer(SBERT_MODEL, device='cuda')

    dim   = 384
    index = faiss.IndexFlatIP(dim)
    meta  = []

    total_original_tokens = 0
    total_chunks          = 0

    for skill_doc in skill_docs:
        user_id  = skill_doc.get("user_id", "unknown")
        skill_md = skill_doc.get("skill_md", "")

        if not skill_md:
            continue

        original_tokens  = len(skill_md.split()) * 1.3
        total_original_tokens += original_tokens

        sections         = re.split(r'\n(?=## )', skill_md)
        chunks_this_user = []

        for section in sections:
            section = section.strip()
            if not section or len(section) < 10:
                continue

            title_match = re.match(r'## (.+)', section)
            title       = title_match.group(1).strip() if title_match else "general"
            vec         = sbert.encode(
                section, normalize_embeddings=True
            ).astype(np.float32)

            is_duplicate = False
            if index.ntotal > 0:
                q = vec.reshape(1, -1).copy()
                faiss.normalize_L2(q)
                scores, indices = index.search(q, 1)
                if scores[0][0] >= DEDUP_THRESHOLD:
                    existing_meta = meta[indices[0][0]]
                    if existing_meta.get("user_id") == user_id:
                        db.skill_chunks.update_one(
                            {"user_id": user_id, "title": title},
                            {"$inc": {"support": 1}},
                        )
                        is_duplicate = True

            if not is_duplicate:
                chunk_doc = {
                    "user_id":    user_id,
                    "title":      title,
                    "content":    section,
                    "vector":     vec.tolist(),
                    "support":    1,
                    "updated_at": datetime.utcnow(),
                }
                db.skill_chunks.replace_one(
                    {"user_id": user_id, "title": title},
                    chunk_doc,
                    upsert=True,
                )

                vec_norm = vec.reshape(1, -1).copy()
                faiss.normalize_L2(vec_norm)
                index.add(vec_norm)
                meta.append({"user_id": user_id, "title": title, "content": section})
                chunks_this_user.append(section)
                total_chunks += 1

        print(f"  {user_id}: {len(chunks_this_user)} chunks | ~{original_tokens:.0f} tokens")

    if index.ntotal > 0:
        faiss.write_index(index, CHUNK_INDEX_PATH)
        with open(CHUNK_META_PATH, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        print(f"\n  FAISS index saved: {CHUNK_INDEX_PATH} ({index.ntotal} chunks)")
    else:
        print("  No chunks indexed")

    return total_original_tokens, total_chunks


def step4_verify_compression(user_id: str = None):
    print("\n[Step 4] Verifying skill chunk compression...")

    if not user_id:
        doc = db.user_skills.find_one({})
        if not doc:
            print("  No SKILL.md to verify")
            return
        user_id = doc["user_id"]

    doc = db.user_skills.find_one({"user_id": user_id})
    if not doc:
        print(f"  No skill profile found for {user_id}")
        return

    full_skill_md = doc["skill_md"]
    full_tokens   = len(full_skill_md.split()) * 1.3
    full_chars    = len(full_skill_md)

    chunks = list(db.skill_chunks.find({"user_id": user_id}).limit(TOP_K))
    if not chunks:
        print(f"  No chunks found for {user_id}")
        return

    top_content  = "\n\n".join(c["content"] for c in chunks[:TOP_K])
    top_tokens   = len(top_content.split()) * 1.3
    top_chars    = len(top_content)
    compression  = (1 - top_tokens / full_tokens) * 100 if full_tokens > 0 else 0

    print(f"  User: {user_id}")
    print(f"  Full SKILL.md: {full_chars} chars ~ {full_tokens:.0f} tokens")
    print(f"  Top-{TOP_K} chunks: {top_chars} chars ~ {top_tokens:.0f} tokens")
    print(f"  Compression: {compression:.1f}%")

    if top_tokens <= TARGET_TOKENS:
        print(f"  Target achieved (< {TARGET_TOKENS} tokens)")
    else:
        print(f"  Target not yet reached ({TARGET_TOKENS} tokens)")

    return {
        "user_id":     user_id,
        "full_tokens": full_tokens,
        "top_tokens":  top_tokens,
        "compression": compression,
    }


def step5_verify_collections():
    print("\n[Step 5] Verifying collection state...")

    collections_to_check = [
        ("observation_logs",   "Layer 1 raw logs (TTL 14 days)"),
        ("semantic_memories",  "Layer 1 sensor data"),
        ("episodic_summaries", "Layer 2 episodic summaries"),
        ("user_skills",        "Layer 3 SKILL.md"),
        ("skill_chunks",       "Layer 3 FAISS chunks"),
    ]

    for col_name, desc in collections_to_check:
        count = db[col_name].count_documents({})
        print(f"  {col_name:25s} | {desc:35s} | {count} records")

    print("\n  TTL index status:")
    try:
        indexes = db.observation_logs.index_information()
        for name, info in indexes.items():
            if "expireAfterSeconds" in info:
                days = info["expireAfterSeconds"] // 86400
                print(f"    {name}: expires after {days} days")
    except Exception as e:
        print(f"    Could not read indexes: {e}")


if __name__ == "__main__":
    print("=" * 60)
    print("migrate_skills.py — three-layer memory system init")
    print("=" * 60)

    step1_create_ttl_index()
    step2_create_collections()
    result, total_chunks = step3_chunk_all_skills()
    step4_verify_compression()
    step5_verify_collections()

    print("\n" + "=" * 60)
    print("Done.")
    if result:
        print(f"  Total chunks processed: {total_chunks}")
    print("  Next: run /interact to test intent classification")
    print("=" * 60)