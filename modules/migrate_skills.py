from pymongo import MongoClient, ASCENDING
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json
import re
import os
from datetime import datetime

# ── 設定 ──────────────────────────────────────────────────────────────────────
MONGO_URI  = "mongodb://127.0.0.1:27017/"
DB_NAME    = "robot_rag_db"
TTL_DAYS   = 14          # Layer 1 TTL（天）
TOP_K      = 2           # FAISS 切片 Top-K
TARGET_TOKENS = 400      # 目標 token 數

SBERT_MODEL = "paraphrase-MiniLM-L6-v2"
CHUNK_INDEX_PATH = "skill_chunks.index"
CHUNK_META_PATH  = "skill_chunks_meta.json"

client = MongoClient(MONGO_URI)
db     = client[DB_NAME]


def step1_create_ttl_index():
    """Step 1：observation_logs TTL Index（14 天，Layer 1）"""
    print("\n[Step 1] 建立 observation_logs TTL Index...")
    try:
        # 先檢查是否已存在
        existing = db.observation_logs.index_information()
        ttl_exists = any(
            "expireAfterSeconds" in str(info)
            for info in existing.values()
        )
        if ttl_exists:
            print("  ✓ TTL Index 已存在，略過")
            return

        db.observation_logs.create_index(
            [("last_seen", ASCENDING)],
            expireAfterSeconds = TTL_DAYS * 86400,
            name = "observation_ttl_14d"
        )
        print(f"  ✓ TTL Index 建立完成（{TTL_DAYS} 天後過期）")
    except Exception as e:
        print(f"  ✗ TTL Index 建立失敗：{e}")


def step2_create_episodic_collection():
    """Step 2：建立 episodic_summaries collection（Layer 2）"""
    print("\n[Step 2] 初始化 episodic_summaries collection（Layer 2）...")
    try:
        collections = db.list_collection_names()
        if "episodic_summaries" not in collections:
            db.create_collection("episodic_summaries")
            db.episodic_summaries.create_index([("user", ASCENDING)])
            db.episodic_summaries.create_index([("count", ASCENDING)])
            print("  ✓ episodic_summaries collection 建立完成")

            # 插入示範資料（咖啡→茶場景，論文 §4.3）
            demo = {
                "user":               "User_Mom",
                "original_request":   "coffee",
                "chosen_alternative": "tea",
                "context":            "User asked for coffee but coffee was unavailable",
                "count":              3,
                "confidence":         0.85,
                "source":             "observation_inference",
                "created_at":         datetime.utcnow(),
            }
            db.episodic_summaries.insert_one(demo)
            print("  ✓ 示範 Episodic Summary 已插入（User_Mom: coffee→tea）")
        else:
            count = db.episodic_summaries.count_documents({})
            print(f"  ✓ 已存在（{count} 筆紀錄）")

        # 確保 skill_chunks collection 存在（Layer 3 切片）
        if "skill_chunks" not in collections:
            db.create_collection("skill_chunks")
            db.skill_chunks.create_index([("user_id", ASCENDING)])
            db.skill_chunks.create_index([("support", ASCENDING)])
            print("  ✓ skill_chunks collection 建立完成")
        else:
            count = db.skill_chunks.count_documents({})
            print(f"  ✓ skill_chunks 已存在（{count} 筆）")

    except Exception as e:
        print(f"  ✗ 錯誤：{e}")


def step3_chunk_all_skills():
    """Step 3：把現有所有 SKILL.md 切割並建立 FAISS Index（論文 §4.2）"""
    print("\n[Step 3] 建立 FAISS 技能切片 Index...")

    skill_docs = list(db.user_skills.find({}, {"user_id":1, "skill_md":1}))
    if not skill_docs:
        print("  ⚠ 沒有找到任何 SKILL.md，略過")
        print("  → 請先執行系統讓機器人生成 SKILL.md，再重新執行此腳本")
        return

    print(f"  找到 {len(skill_docs)} 個使用者的 SKILL.md")

    # 載入 SBERT
    print(f"  載入 SBERT ({SBERT_MODEL})...")
    sbert = SentenceTransformer(SBERT_MODEL, device='cuda')

    dim   = 384
    index = faiss.IndexFlatIP(dim)
    meta  = []

    total_original_tokens = 0
    total_chunks = 0

    for skill_doc in skill_docs:
        user_id  = skill_doc.get("user_id", "unknown")
        skill_md = skill_doc.get("skill_md", "")

        if not skill_md:
            continue

        # 計算原始 token 估計
        original_tokens = len(skill_md.split()) * 1.3
        total_original_tokens += original_tokens

        # 按 ## 切割
        sections = re.split(r'\n(?=## )', skill_md)
        chunks_this_user = []

        for section in sections:
            section = section.strip()
            if not section or len(section) < 10:
                continue

            title_match = re.match(r'## (.+)', section)
            title = title_match.group(1).strip() if title_match else "general"

            # 編碼
            vec = sbert.encode(section, normalize_embeddings=True).astype(np.float32)

            # 檢查語義去重（> 0.85 不重複新增）
            is_duplicate = False
            if index.ntotal > 0:
                q = vec.reshape(1, -1).copy()
                faiss.normalize_L2(q)
                scores, indices = index.search(q, 1)
                if scores[0][0] >= 0.85:
                    # 相似塊已存在，只更新 MongoDB support
                    existing_meta = meta[indices[0][0]]
                    if existing_meta.get("user_id") == user_id:
                        db.skill_chunks.update_one(
                            {"user_id": user_id, "title": title},
                            {"$inc": {"support": 1}}
                        )
                        is_duplicate = True

            if not is_duplicate:
                # 寫入 MongoDB
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
                    upsert=True
                )

                # 加入 FAISS
                vec_norm = vec.reshape(1, -1).copy()
                faiss.normalize_L2(vec_norm)
                index.add(vec_norm)
                meta.append({"user_id": user_id, "title": title, "content": section})
                chunks_this_user.append(section)
                total_chunks += 1

        print(f"  User: {user_id} | {len(chunks_this_user)} chunks | "
              f"原始: ~{original_tokens:.0f} tokens")

    # 儲存 FAISS index
    if index.ntotal > 0:
        faiss.write_index(index, CHUNK_INDEX_PATH)
        with open(CHUNK_META_PATH, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        print(f"\n  ✓ FAISS Index 儲存完成（{CHUNK_INDEX_PATH}）")
        print(f"  ✓ {index.ntotal} 個技能塊已索引")
    else:
        print("  ⚠ 沒有技能塊被索引")

    return total_original_tokens, total_chunks


def step4_verify_compression(user_id: str = None):
    """Step 4：驗證 Token 壓縮效果（目標 1300+ → < 400）"""
    print("\n[Step 4] 驗證 FAISS 技能切片壓縮效果...")

    # 找一個有 SKILL.md 的使用者
    if not user_id:
        doc = db.user_skills.find_one({})
        if not doc:
            print("  ⚠ 沒有 SKILL.md 可以驗證")
            return
        user_id = doc["user_id"]

    doc = db.user_skills.find_one({"user_id": user_id})
    if not doc:
        print(f"  ⚠ 找不到 {user_id} 的 SKILL.md")
        return

    full_skill_md  = doc["skill_md"]
    full_tokens    = len(full_skill_md.split()) * 1.3
    full_chars     = len(full_skill_md)

    # 模擬用 FAISS 切片（Top-2）
    chunks = list(db.skill_chunks.find({"user_id": user_id}).limit(TOP_K))
    if not chunks:
        print(f"  ⚠ {user_id} 沒有技能切片")
        return

    top2_content = "\n\n".join(c["content"] for c in chunks[:TOP_K])
    top2_tokens  = len(top2_content.split()) * 1.3
    top2_chars   = len(top2_content)

    compression  = (1 - top2_tokens / full_tokens) * 100 if full_tokens > 0 else 0

    print(f"  使用者：{user_id}")
    print(f"  完整 SKILL.md：{full_chars} 字元 ≈ {full_tokens:.0f} tokens")
    print(f"  Top-{TOP_K} 切片：{top2_chars} 字元 ≈ {top2_tokens:.0f} tokens")
    print(f"  壓縮率：{compression:.1f}%")

    if top2_tokens <= TARGET_TOKENS:
        print(f"  ✓ 達成目標（< {TARGET_TOKENS} tokens）")
    else:
        print(f"  ⚠ 尚未達到目標（{TARGET_TOKENS} tokens），建議增加技能塊分割粒度")

    return {
        "user_id":      user_id,
        "full_tokens":  full_tokens,
        "top2_tokens":  top2_tokens,
        "compression":  compression,
    }


def step5_verify_collections():
    """Step 5：驗證所有 collection 狀態"""
    print("\n[Step 5] Collection 狀態驗證...")

    collections_to_check = [
        ("observation_logs",    "Layer 1 Raw Logs（TTL 14天）"),
        ("semantic_memories",   "Layer 1 原始感測資料"),
        ("episodic_summaries",  "Layer 2 Episodic Summaries"),
        ("user_skills",         "Layer 3 Skills (SKILL.md)"),
        ("skill_chunks",        "Layer 3 FAISS 技能切片"),
    ]

    for col_name, desc in collections_to_check:
        count = db[col_name].count_documents({})
        print(f"  {col_name:25s} | {desc:30s} | {count} 筆")

    # 確認 TTL Index
    print("\n  TTL Index 狀態：")
    try:
        indexes = db.observation_logs.index_information()
        for name, info in indexes.items():
            if "expireAfterSeconds" in info:
                days = info["expireAfterSeconds"] // 86400
                print(f"    ✓ {name}：{days} 天後過期")
    except Exception as e:
        print(f"    ✗ 無法讀取 Index：{e}")


if __name__ == "__main__":
    print("=" * 60)
    print("migrate_skills.py — 三層記憶體系初始化")
    print("=" * 60)

    step1_create_ttl_index()
    step2_create_episodic_collection()
    result = step3_chunk_all_skills()
    step4_verify_compression()
    step5_verify_collections()

    print("\n" + "=" * 60)
    print("完成！")
    if result:
        total_tokens, total_chunks = result
        print(f"  總計處理 {total_chunks} 個技能塊")
    print("  下一步：執行 /interact 測試三類意圖分類")
    print("=" * 60)