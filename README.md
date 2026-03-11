# 🤖 Context-Aware Home Service Robot

### VLM + Retrieval-Augmented Personalized Memory

> Master's Thesis — NCKU Computer Science, Taiwan  
> Exchange Student @ RWTH Aachen University 2025–2026

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-3.0-black?logo=flask)](https://flask.palletsprojects.com)
[![MongoDB](https://img.shields.io/badge/MongoDB-7.0-green?logo=mongodb)](https://mongodb.com)
[![Unity](https://img.shields.io/badge/Unity-2022.3-black?logo=unity)](https://unity.com)
[![Ollama](https://img.shields.io/badge/Ollama-llava--phi3%20%2B%20gemma3-orange)](https://ollama.com)
[![FAISS](https://img.shields.io/badge/FAISS-IndexFlatIP-blue)](https://github.com/facebookresearch/faiss)

---

## Overview

A context-aware home service robot system that **understands who you are and what you need** — even when you express it vaguely.

The system observes daily behavior through simulated cameras, builds personalized long-term memory per user, and reasons about fuzzy needs using a RAG pipeline:

> *"I'm hungry"* → **Mom**: navigates to kitchen table (banana seen there 20×)  
> *"I want to rest"* → **Mom**: navigates to mom's bed (sleeping observed 15×)  
> *"Where's my phone?"* → *"Last seen on the desk in the bedroom"*

---

## System Architecture

```
Unity Simulation
  ├── ProxyExportManager     全部物件一起送 /scene（靜態 + 動態）
  ├── VirtualCameraBrain     定點相機拍攝 → /predict
  ├── ExperimentRunner       實驗腳本（Exp1–6）
  └── UserEntity             Mom / Dad 行為序列 + 物件 attach/detach
          │
          │  POST /scene     物件位置（全部，含家具 + 動態物件）
          │  POST /predict   影像 + user_pos + room_name
          │  POST /interact  自然語言查詢
          ▼
Flask AI Backend  (app.py)
  ├── classifier.py     背景執行緒：raw_objects → 靜態/動態分類 + 空間錨點綁定
  ├── perception.py     VLM 行為辨識 + 語意 binding + dynamic_objects 更新
  ├── memory.py         家具 binding + observation_logs
  ├── memory_vector.py  FAISS 雙索引（習慣記憶 + 動態物件）
  └── interaction.py    LLM 意圖分析 + 三層 RAG 回答
          │
          ▼
MongoDB
  ├── raw_objects          Unity 原始輸入（processed flag）
  ├── scene_snapshots      靜態家具位置（2D index）
  ├── dynamic_objects      動態物件（sensor_pos + VLM 補充的 last_seen_on）
  ├── observation_logs     行為記錄 + habit weight
  ├── semantic_memories    個人化長期記憶
  ├── activity_sequences   行為序列
  ├── conversation_logs    對話記錄
  ├── eval_logs            實驗評估記錄
  └── exp_checkpoints      實驗 checkpoint
```

---

## Three Core Contributions

### 1. Multi-Stage Perception Pipeline

Fixed camera nodes capture images → VLM recognizes action and objects → scene graph binding maps objects to furniture using semantic + coordinate cross-validation.

```
Images (llava-phi3)
  → action + main_object + spatial_relations
        ↓
Action-Guided Semantic Binding
  action="sleeping" → candidates: [bed, sofa]
  SBERT Top-K + coordinate distance → final bound_doc
        ↓
  scene_snapshots（家具層）
  dynamic_objects（VLM 補充 last_seen_on + spatial_rel）
```

**Key design — two models, two tasks:**
- `llava-phi3` → visual recognition (accurate scene understanding)
- `gemma3:4b` → language reasoning (natural RAG answers)

### 2. Personalized Long-Term Memory

Each user accumulates independent habit memory. The system tracks:
- **Weight accumulation**: more observations → higher retrieval score
- **Spatial binding**: *where* the user does *what* with *which objects*
- **FAISS dual index**: fast similarity search for habits and dynamic objects separately

```
User_Mom drinking × 30  →  weight=30  →  RAG recommends kitchen table
User_Dad drinking × 10  →  weight=10  →  RAG recommends living room sofa
```

### 3. Fuzzy Need Reasoning (RAG)

Intent analysis + cross-matching + three-layer fallback:

```
User: "I'm hungry"
  ↓
LLM intent analysis
  → intent_type = fuzzy_need
  → keywords    = [eating, food, kitchen, apple, banana]
  ↓
FAISS Layer 1: 個人習慣記憶（observation_logs weight）
FAISS Layer 2: 動態物件（dynamic_objects，目前家裡有什麼）
  ↓
Cross-match: 習慣物件 ∩ 現有物件 → 個人化推薦分數
  ↓
Fallback:
  有習慣記憶 → 推薦常用物件 + 導航座標
  無習慣記憶 → 告知空間現有物件
  都沒有     → 說需要更多時間觀察
```

---

## File Structure

```
robotBrain/
├── app.py                  Flask 主程式，所有 API endpoints
├── classifier.py           背景執行緒：raw_objects 分類 + 空間錨點綁定
├── config.py               MongoDB / Ollama / FAISS 設定
├── interact_client.py      終端機對話介面（含 check_backend + help）
├── reset_all.py            開發用：清空所有 DB + FAISS
├── auto_eval.py            自動評估（Exp1/2）
├── eval_all.py             評估結果彙整
│
├── modules/
│   ├── perception.py       PerceptionEngine v5：VLM 感知 + 四大效能優化
│   ├── memory.py           MemoryManager：家具 binding + observation_logs
│   ├── memory_vector.py    VectorMemory：FAISS 雙索引
│   ├── interaction.py      InteractionEngine v2：LLM 意圖 + RAG
│   └── training_exporter.py 訓練資料匯出
│
└── Unity/                  （C# scripts，詳見 README_Unity.md）
    ├── ProxyExportManager.cs
    ├── ExperimentRunner.cs
    ├── VirtualCameraBrain.cs
    ├── StaticCameraManager.cs
    └── SharedDataStructures.cs
```

---

## Model Configuration

```python
# config.py
VLM_MODEL = "llava-phi3"   # perception.py  → 視覺辨識
LLM_MODEL = "gemma3:4b"    # interaction.py → 語言推理 / RAG
```

| 模組 | 模型 | 職責 |
|------|------|------|
| `perception.py` | `llava-phi3` | 看圖認人、辨識行為、定位物件、輸出 JSON |
| `interaction.py` | `gemma3:4b` | 意圖分析、個人化回答生成 |

---

## PerceptionEngine v5 — 四大效能優化

| 優化 | 機制 | 效益 |
|------|------|------|
| Room Embedding Cache | 切房間才重建 SBERT 向量 | 避免每幀重新 encode，同房間只做矩陣乘法 |
| Change Streams Sync | MongoDB `watch()` 被動訂閱，fallback Polling（10s）| 平時只讀記憶體，IO 最小化 |
| Top-K Semantic Filter | 語意 Top-3 + 座標距離交叉驗證 | 家具密集區綁定容錯率提升 |
| Diff & Bulk Write | 狀態比對攔截無效更新，累積 20 筆或 30s 批量寫入 | 高幀率下減少 DB 寫入壓力 |

動態物件綁定優先順序：
```
1. relation == in_hand  → 跟人走（bound_label）
2. VLM 給出家具名稱    → Top-K 語意 + 座標距離決策
3. fallback             → bound_label（當前最近家具）
```

---

## ObjectClassifier — 空間錨點綁定

背景執行緒每 5 秒從 `raw_objects` 讀取未處理資料並分類：

```
label in FURNITURE_LABELS → scene_snapshots（靜態家具）
label 不在               → dynamic_objects（動態物件）
                                 ↓
                    _find_closest_furniture(x, z, room)
                      Pass 1：同房間（模糊比對）<= 0.8m → 回傳家具名
                      Pass 2：全場景 fallback     <= 0.8m → 回傳家具名
                      兩者都 > 0.8m              → "floor"
```

房間名模糊比對（大小寫、空格、底線不敏感）：
```python
f_norm   = f_room.lower().replace(" ", "").replace("_", "")
obj_norm = obj_room.lower().replace(" ", "").replace("_", "")
# "BedRoom(mom)" 和 "bedroom_mom" 都能匹配
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/scene` | 接收 Unity 物件資料 → `raw_objects` |
| `POST` | `/predict` | 接收影像 → VLM 推理 → 更新記憶 |
| `POST` | `/interact` | 自然語言查詢 → 個人化回答 |
| `POST` | `/interact/confirm` | 確認導航目標（navigate / info / cancel）|
| `GET`  | `/exp_checkpoint` | 實驗 checkpoint 記錄（Exp3A/B）|
| `POST` | `/query` | 直接查詢習慣記憶（debug 用）|
| `POST` | `/log_navigation` | 記錄導航結果 |
| `POST` | `/export_training` | 匯出訓練資料 |
| `POST` | `/dynamic_sync` | sensor 直接同步動態物件位置 |

---

## MongoDB Collections

| Collection | 寫入者 | 用途 |
|------------|--------|------|
| `raw_objects` | `/scene` | Unity 原始輸入，classifier 分類用（`processed` flag）|
| `scene_snapshots` | `ObjectClassifier` | 靜態家具：`label` + `pos`（2D index）+ `image` |
| `dynamic_objects` | `Classifier` + `perception.py` | 動態物件，欄位見下表 |
| `observation_logs` | `memory.py` | 行為記錄：`user` + `action` + `instance` + `weight` |
| `semantic_memories` | `perception.py` | 個人化長期記憶（RAG 主要來源）|
| `activity_sequences` | `memory.py` | 行為序列（Exp3B）|
| `conversation_logs` | `interaction.py` | 對話記錄 + 導航確認 |
| `eval_logs` | `app.py` | VLM 辨識評估（Exp1/2）|
| `exp_checkpoints` | `app.py` | 實驗三A/B checkpoint |

### dynamic_objects 欄位分工

| 欄位 | 寫入者 | 說明 |
|------|--------|------|
| `sensor_pos` | `classifier.py` | 物件真實座標（sensor 提供），VLM 不覆蓋 |
| `last_seen_on` | `classifier.py` / `perception.py` | 語意位置（"table", "desk", "in_hand"...）|
| `spatial_rel` | `perception.py` | 空間關係（"on", "in", "held_by", "at"）|
| `furniture_pos` | `perception.py` | 最近家具的座標（VLM binding 後補充）|
| `is_movable` | `classifier.py` | `True`（動態物件標記）|
| `interacted_by` | `perception.py` | 互動過的 user ID 列表（個人化用）|

---

## Quick Start

### Prerequisites

```bash
# 兩個 Ollama models 都需要 pull
ollama pull llava-phi3
ollama pull gemma3:4b

# Python dependencies
pip install flask pymongo sentence-transformers faiss-cpu \
            requests opencv-python numpy

# MongoDB
brew services start mongodb-community  # macOS
sudo systemctl start mongod            # Linux
```

### Run

```bash
# Terminal 1：啟動 Ollama
ollama serve

# Terminal 2：啟動後端
cd robotBrain
python3 reset_all.py     # 清空 DB（第一次執行或重跑實驗前）
python3 app.py

# Terminal 3：對話介面
python3 interact_client.py
```

### Expected Startup Output

```
🖥️  System initialization... Running on: cuda
✅ SBERT loaded on CUDA
✅ MongoDB 2D Index ready
   📦 [ChangeSync] Loaded 14 scene objects into memory
   ℹ️  [ChangeSync] Polling 模式（每 10 秒）
[Classifier] ✅ 背景分類執行緒啟動 (含空間語意對齊)
🚀 Robot Brain Server on 0.0.0.0:5000
   SBERT device : cuda
   VLM model    : llava-phi3   ← perception
   LLM model    : gemma3:4b   ← interaction/RAG
```

---

## Experiment Design

| Exp | 目的 | 樣本數 | 記錄方式 |
|-----|------|--------|---------|
| Exp 1 | VLM 行為辨識準確率 | 80 筆（Mom+Dad × 4 動作 × 10）| `eval_logs` |
| Exp 2 | 家具語意綁定消融 | 同 Exp1 | `eval_logs.bound_label` |
| Exp 3A | 習慣 weight 累積驗證 | Mom drinking × 30，每 5 筆 checkpoint | `exp_checkpoints` |
| Exp 3B | 行為序列預測 | 5 天（Mom 6 步 + Dad 5 步 / 天）| `exp_checkpoints` |
| Exp 4 | RAG 對話品質 | 25 題 × 3 條件 × 3 評審 | 人工評分 |
| Exp 5 | 端到端整合 | 3 情境 × 5 次 | 人工評分 |
| Exp 6 | 動態物件移動偵測 | 物件移動腳本，驗證 pipeline | `dynamic_objects` |

---

## Development Commands

```bash
# 清空所有資料重新開始
python3 reset_all.py

# 確認 MongoDB 狀態
mongosh
use robot_rag_db
db.scene_snapshots.countDocuments()          # 應為家具數量（約 14）
db.raw_objects.find({processed: false})      # 應為空
db.dynamic_objects.find({label: "banana"}).pretty()
db.observation_logs.find({user: "User_Mom"}).sort({weight: -1}).limit(5)

# 手動觸發 checkpoint（debug）
curl "http://localhost:5000/exp_checkpoint?experiment=exp3a&step=5&user=User_Mom&action=drinking"

# 查看評估結果
python3 eval_all.py
```

---

## Known Issues & Design Notes

**`/predict` 比 `/scene` 早到**  
若 `scene_snapshots` 尚未建立（classifier 還沒跑完），`/predict` 會等待最多 12 秒，直到 classifier 完成第一次掃描再繼續 VLM 推理。ExperimentRunner 也應在 `Start()` 等待至少 8 秒再送第一筆 `/predict`。

**VLM JSON 截斷**  
`num_predict=800`（llava-phi3 輸出比 gemma3 長）。`_extract_json()` 有截斷修復邏輯：從尾端找最後完整 field，截斷後補 `}}`，避免整幀資料丟失。

**房間名格式不一致**  
`_find_closest_furniture()` 使用模糊比對（大小寫、空格、底線不敏感），同房間找不到時 fallback 到全場景座標距離計算。

**Sensor Assumption**  
In simulation, static/dynamic classification is handled in Python using `FURNITURE_LABELS`. In real deployment, this distinction would come from depth sensors or semantic segmentation (e.g., PointNet, Mask R-CNN).

---

## Thesis

**Title**: 基於視覺語言模型與檢索增強記憶之居家服務機器人情境感知系統

**Three Core Contributions**:
1. Multi-stage perception with VLM action recognition and action-guided semantic binding
2. Personalized long-term memory with habit weight accumulation and FAISS dual index
3. Fuzzy need reasoning via RAG with LLM intent analysis and three-layer fallback

---

## Author

**Hui-Hsin Huang**  
National Cheng Kung University, Computer Science  
Exchange @ RWTH Aachen University 2025–2026