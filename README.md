# 🤖 Context-Aware Home Service Robot
### VLM + Retrieval-Augmented Personalized Memory

> Master's Thesis — NCKU, Taiwan  
> Exchange Student @ RWTH Aachen University 2025–2026

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-3.0-black?logo=flask)](https://flask.palletsprojects.com)
[![MongoDB](https://img.shields.io/badge/MongoDB-7.0-green?logo=mongodb)](https://mongodb.com)
[![Unity](https://img.shields.io/badge/Unity-2022.3-black?logo=unity)](https://unity.com)
[![Ollama](https://img.shields.io/badge/Ollama-llava--phi3-orange)](https://ollama.com)
[![FAISS](https://img.shields.io/badge/FAISS-IndexFlatIP-blue)](https://github.com/facebookresearch/faiss)

---

## Overview

A context-aware home service robot system that **understands who you are and what you need** — even when you express it vaguely.

The system observes daily behavior through simulated cameras, builds personalized long-term memory per user, and reasons about fuzzy needs using a RAG pipeline:

> *"I'm hungry"* → **Mom**: navigates to kitchen table (banana, seen 32×)  
> *"I'm hungry"* → **Dad**: navigates to living room shelf (snacks, seen 8×)  
> *"Where's my glasses?"* → *"Last seen on the desk in Study room"*

---

## System Architecture

```
Unity Simulation
  ├── Fixed Camera Nodes     (simulates real-world mounted cameras)
  ├── UserEntity             (Mom / Dad behavioral sequences)
  └── ProxyExportManager     (syncs furniture positions → /scene)
          │
          │  POST /predict   (images + user_pos + room_name)
          │  POST /interact  (natural language query)
          ▼
Flask AI Brain  (app.py)
  ├── perception.py     →  VLM action recognition + scene graph binding
  ├── memory.py         →  Furniture binding + observation_logs
  ├── memory_vector.py  →  FAISS dual-index (habit + dynamic objects)
  └── interaction.py    →  LLM intent analysis + personalized RAG
          │
          ├── MongoDB   scene_snapshots / observation_logs /
          │             dynamic_objects / activity_sequences /
          │             conversation_logs / eval_logs
          └── FAISS     habit memory index + dynamic object index
```

---

## Key Contributions

### 1. Multi-Stage Perception Pipeline (`perception.py`)
- Fixed camera nodes send image bursts → **llava-phi3** via Ollama
- Multi-frame majority voting reduces single-frame VLM noise
- **Room Embedding Cache** — SBERT encodes furniture only when user switches rooms (event-driven, not every frame)
- **Top-K Semantic Filter** — Top-3 semantic candidates + distance decision for robust furniture binding
- **Change Stream Sync** — MongoDB scene updates propagate to in-memory cache instantly (Polling fallback for dev)
- **Diff + Async Bulk Write** — state comparison intercepts unchanged objects; flushes in batch (20 ops or 30s)

### 2. Personalized Long-Term Memory (`memory.py` + `memory_vector.py`)
- Per-user **habit weight accumulation** in `observation_logs`
- **FAISS dual-index**: habit memory (behavior layer) + dynamic objects (item layer)
- **Event-driven FAISS encode**: dynamic objects only re-encode when physically moved
- `sync_from_mongo()` at startup syncs colleague sensor data + VLM data into FAISS

### 3. Personalized Fuzzy Need Reasoning (`interaction.py`)
- **LLM intent analysis** replaces rule-based INTENT_MAP — handles any natural language input
- **Cross-matching**: habit items (high weight) ∩ currently available objects → ranked recommendations
- Recommendation score: `interact_count × 0.4 + habit_count × 0.4 + FAISS_similarity × 0.2`
- LLM generates personalized answer with explanation of *why* it recommends

---

## Performance Optimizations

| Optimization | Before | After |
|---|---|---|
| Room furniture binding | Full DB query every frame | In-memory matrix multiply (μs) |
| scene_snapshots sync | `find()` every request | MongoDB watch() / 10s polling |
| dynamic_objects write | `upsert` every frame | Diff intercept + bulk write |
| FAISS dynamic encode | Every upsert | Only on position change |

---

## Experimental Design

| # | Experiment | Metric |
|---|---|---|
| Exp 1 | VLM Action Recognition (80 samples) | Accuracy, Confusion Matrix |
| Exp 2 | Furniture Binding Ablation (coord / sbert / combined) | Top-1 Accuracy |
| Exp 3A | Habit Weight Accumulation (Mom drinking ×30) | Weight curve |
| Exp 3B | Action Sequence Prediction (5-day) | Top-1 / Top-3 |
| Exp 4 | RAG Dialogue Quality (25Q × 3 conditions × 3 reviewers) | Avg Score 1–5 |
| Exp 5 | End-to-End Integration (3 scenarios × 5 runs) | Success Rate |
| Exp 6 | Personalized Fuzzy Need (30Q, Mom vs Dad) | Intent Accuracy |

> ⚙️ Experiments in progress — results will be updated upon completion

---

## Tech Stack

| Layer | Technology |
|---|---|
| Simulation | Unity 2022.3, C# |
| AI Brain | Python 3.10, Flask |
| VLM | llava-phi3 via Ollama |
| LLM | Gemma3:4b via Ollama |
| Embedding | SBERT `paraphrase-MiniLM-L6-v2` (CUDA) |
| Vector Search | FAISS `IndexFlatIP` |
| Database | MongoDB 7.0 |
| Hardware (planned) | ROS2 Humble |

---

## Project Structure

```
robotBrain/
├── app.py                    # Flask server, GPU, eval_logs auto-record
├── config.py                 # Ollama URL, MongoDB URI, model names
├── reset_all.py              # Dev: wipe all data and FAISS files
├── auto_eval.py              # Automated 55-question evaluation
├── eval_all.py               # Chart generation (matplotlib)
├── interact_client.py        # Terminal dialogue test client
├── modules/
│   ├── perception.py         # PerceptionEngine v5
│   │                         #   RoomEmbeddingCache, ChangeStreamSync
│   │                         #   BulkWriteBuffer, Top-K Semantic Filter
│   ├── memory.py             # MemoryManager v3
│   │                         #   observation_logs, activity_sequences
│   ├── memory_vector.py      # VectorMemory v2
│   │                         #   FAISS dual-index, sync_from_mongo
│   ├── interaction.py        # InteractionEngine v2
│   │                         #   LLM intent analysis, cross-match recommender
│   └── training_exporter.py  # Export training data (JSONL)
└── ExperimentRunner.cs       # Unity auto-experiment controller (C#)
```

---

## Quick Start

```bash
# 1. Pull models
ollama pull llava-phi3
ollama pull gemma3:4b

# 2. Install dependencies
pip install flask pymongo faiss-cpu sentence-transformers \
            numpy opencv-python requests matplotlib torch

# 3. Start services
ollama serve            # Terminal 1
python app.py           # Terminal 2

# 4. Test dialogue
python interact_client.py   # Terminal 3

# 5. Reset all data (dev)
python reset_all.py
```

---

## MongoDB Collections

| Collection | Description |
|---|---|
| `scene_snapshots` | Static furniture positions (synced from Unity) |
| `observation_logs` | Per-user habit weights (`user`, `action`, `weight`) |
| `dynamic_objects` | Movable item locations (`label`, `last_seen_on`, `room`) |
| `activity_sequences` | Daily action timeline per user |
| `conversation_logs` | Full RAG dialogue history |
| `eval_logs` | Auto-recorded VLM evaluation data |
| `exp_checkpoints` | Experiment 3A/3B weight snapshots |

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| POST | `/predict` | Main perception: image → action + binding |
| POST | `/interact` | Natural language query → personalized answer |
| POST | `/interact/confirm` | Confirm navigation target |
| POST | `/scene` | Sync furniture from Unity |
| GET | `/exp_checkpoint` | Record experiment checkpoint |

---

## ROS2 Integration (Planned)

Bridge designed for physical hardware migration:
```
camera_node → POST /predict  → perception pipeline
memory_node → POST /interact → publishes /nav_goal
```
See [`ros2_bridge.py`](ros2_bridge.py)

---

## Thesis

**Context-Aware Home Service Robot System Based on VLM and Retrieval-Augmented Memory**  
基於視覺語言模型與檢索增強記憶之居家服務機器人情境感知系統

Three core contributions:
1. Multi-stage VLM perception pipeline with event-driven semantic caching
2. Personalized long-term memory with per-user habit weight accumulation
3. Personalized fuzzy need reasoning — same input, different response per user

---

## Author

**Hui-Hsin Huang (黃慧心)** · M.S. Computer Science, NCKU, Taiwan  
Exchange Student @ RWTH Aachen University 2025–2026

[![GitHub](https://img.shields.io/badge/GitHub-your--handle-black?logo=github)](https://github.com/your-handle)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?logo=linkedin)](https://linkedin.com/in/your-handle)