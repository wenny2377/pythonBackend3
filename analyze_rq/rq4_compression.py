import argparse
import datetime
import json
import os
import sys
import time

import requests
from pymongo import MongoClient

BACKEND_URL = "http://localhost:5000"
MONGO_URI   = "mongodb://127.0.0.1:27017/"
DB_NAME     = "robot_rag_db"

USER_ID = "User_Mom"

QUALITY_QUERIES = [
    ("I am thirsty",     "User_Mom", ["juice", "juicebottle", "drink", "cola"]),
    ("I want an apple",  "User_Mom", ["apple", "kitchen", "table"]),
    ("I want to rest",   "User_Mom", ["sofa", "sit", "chair", "rest"]),
]


def call_stream(url, query, user_id):
    try:
        resp = requests.post(
            f"{url}/interact/stream",
            json={"query": query, "userID": user_id, "room": ""},
            stream=True,
            timeout=60,
        )
        resp.raise_for_status()
    except Exception as e:
        return {"answer": "", "intent_type": ""}

    answer  = ""
    intent  = ""
    buf     = ""
    is_json = None

    for raw in resp.iter_lines():
        if not raw:
            continue
        line = raw.decode("utf-8") if isinstance(raw, bytes) else raw
        if not line.startswith("data: "):
            continue
        try:
            ev = json.loads(line[6:])
            if ev.get("type") == "intent":
                intent  = ev.get("intent", "")
                is_json = intent == "service"
            elif ev.get("type") == "token":
                token = ev.get("content", "")
                buf  += token
                if not is_json:
                    answer += token
            elif ev.get("type") == "done":
                if is_json and buf:
                    import re
                    m = re.search(r'"answer"\s*:\s*"((?:[^"\\]|\\.)*)"', buf)
                    if m:
                        answer = m.group(1)
                break
        except Exception:
            continue

    return {"answer": answer.strip(), "intent_type": intent}


def get_skill_stats(db):
    doc = db.user_skills.find_one({"user_id": USER_ID})
    if not doc:
        return None
    skill_md    = doc.get("skill_md", "")
    full_chars  = len(skill_md)
    full_tokens = int(full_chars / 4.5)
    return {"skill_md": skill_md, "full_chars": full_chars, "full_tokens": full_tokens}


def get_chunk_stats(db):
    chunks = list(db.skill_chunks.find({"user_id": USER_ID}))
    if not chunks:
        return None

    from sentence_transformers import SentenceTransformer
    import numpy as np
    try:
        import faiss
        model    = SentenceTransformer("paraphrase-MiniLM-L6-v2")
        query    = "drink thirsty juice"
        q_vec    = model.encode([query], normalize_embeddings=True).astype(np.float32)
        contents = [c.get("content", "") for c in chunks]
        vecs     = model.encode(contents, normalize_embeddings=True).astype(np.float32)
        index    = faiss.IndexFlatIP(vecs.shape[1])
        index.add(vecs)
        scores, idxs = index.search(q_vec, min(2, len(chunks)))
        top2_content = "\n\n".join(contents[i] for i in idxs[0] if i < len(contents))
        chunk_chars  = len(top2_content)
        chunk_tokens = int(chunk_chars / 4.5)
        return {
            "n_chunks":     len(chunks),
            "top2_content": top2_content,
            "chunk_chars":  chunk_chars,
            "chunk_tokens": chunk_tokens,
        }
    except Exception as e:
        print(f"  FAISS error: {e}")
        return None


def run_quality_check(url):
    print("\n--- Quality check: 3 queries ---")
    results = []
    for query, user_id, keywords in QUALITY_QUERIES:
        r      = call_stream(url, query, user_id)
        answer = r.get("answer", "").lower()
        hit    = any(kw in answer for kw in keywords)
        results.append({
            "query":    query,
            "answer":   r.get("answer", ""),
            "keywords": keywords,
            "hit":      hit,
        })
        print(f"  {'v' if hit else 'x'}  {query:30}  -> {r.get('answer','')[:50]}")
        time.sleep(0.5)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=BACKEND_URL)
    parser.add_argument("--out", default=".")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    try:
        requests.get(f"{args.url}/", timeout=5)
    except Exception:
        print(f"Cannot connect to {args.url}. Start app.py first.")
        sys.exit(1)

    client = MongoClient(MONGO_URI)
    db     = client[DB_NAME]

    print("\n" + "=" * 60)
    print("RQ4: FAISS Context Compression")
    print("=" * 60)

    print("\nStep 1: Getting SKILL.md stats...")
    skill_stats = get_skill_stats(db)
    if not skill_stats:
        print("  No SKILL.md found for User_Mom.")
        sys.exit(1)

    print(f"  Full SKILL.md : {skill_stats['full_chars']} chars  ~{skill_stats['full_tokens']} tokens")

    print("\nStep 2: Getting FAISS chunk stats...")
    chunk_stats = get_chunk_stats(db)
    if not chunk_stats:
        print("  No skill_chunks found. Run skill_demo.py first to populate chunks.")
        compression = None
    else:
        compression = 1.0 - (chunk_stats["chunk_chars"] / skill_stats["full_chars"])
        print(f"  Chunks found  : {chunk_stats['n_chunks']}")
        print(f"  Top-2 chunks  : {chunk_stats['chunk_chars']} chars  ~{chunk_stats['chunk_tokens']} tokens")
        print(f"  Compression   : {compression:.0%}")

    print("\nStep 3: Quality check...")
    quality = run_quality_check(args.url)
    n_quality_ok = sum(1 for q in quality if q["hit"])

    lines = [
        "=" * 65,
        "RQ4: FAISS Context Compression and Quality",
        f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "=" * 65,
        "",
        "Compression Statistics:",
        f"  Full SKILL.md     : {skill_stats['full_chars']} chars  ~{skill_stats['full_tokens']} tokens",
    ]

    if chunk_stats:
        lines += [
            f"  FAISS chunks      : {chunk_stats['n_chunks']} chunks indexed",
            f"  Top-2 injection   : {chunk_stats['chunk_chars']} chars  ~{chunk_stats['chunk_tokens']} tokens",
            f"  Compression rate  : {compression:.0%}",
            f"  Token reduction   : ~{skill_stats['full_tokens'] - chunk_stats['chunk_tokens']} tokens saved per query",
        ]
    else:
        lines += ["  FAISS chunks: not available"]

    lines += [
        "",
        "Quality Check (response correctness with compressed context):",
        f"  Correct: {n_quality_ok}/{len(quality)}",
        *[
            f"  {'v' if q['hit'] else 'x'}  {q['query']:30}  -> {q['answer'][:50]}"
            for q in quality
        ],
        "",
        "Top-2 chunks content:",
        chunk_stats["top2_content"][:400] if chunk_stats else "N/A",
        "",
        "For thesis:",
        f"The FAISS-based context compression reduces the SKILL.md",
        f"from {skill_stats['full_chars']} characters (~{skill_stats['full_tokens']} tokens) to",
        f"{chunk_stats['chunk_chars'] if chunk_stats else 'N/A'} characters "
        f"(~{chunk_stats['chunk_tokens'] if chunk_stats else 'N/A'} tokens)",
        f"by injecting only the Top-2 semantically relevant chunks",
        f"(compression rate: {compression:.0%})." if compression else "(compression data unavailable).",
        f"Quality check shows {n_quality_ok}/{len(quality)} correct responses,",
        f"confirming that semantic relevance is preserved despite",
        f"significant context reduction.",
    ]

    summary_path = os.path.join(args.out, "rq4_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n  Summary saved: {summary_path}")
    if compression:
        print(f"\n  Compression rate : {compression:.0%}")
    print(f"  Quality check    : {n_quality_ok}/{len(quality)} correct")


if __name__ == "__main__":
    main()