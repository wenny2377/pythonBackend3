# 🤖 Robot RAG System — Python Backend

> VLM Perception · MongoDB Memory · FAISS Vector Search · Scene Graph · AI Training Export

A multi-stage perception and memory system for home robots. The backend receives camera images from Unity, runs VLM inference via Ollama (Gemma3:4b), binds observations to spatial context in MongoDB, indexes memories in FAISS, and exports structured training data for downstream AI model training.

---

## 📁 File Structure

```
project/
├── app.py                        ← Flask server, all API routes
├── config.py                     ← Centralised config (model, DB, paths)
├── interact_client.py            ← Terminal client for /interact
└── modules/
    ├── perception.py             ← Stage 1: VLM inference + multi-frame voting
    ├── memory.py                 ← Stage 2: furniture binding + MongoDB writes
    ├── memory_vector.py          ← Stage 3: FAISS vector memory (habit + dynamic)
    ├── interaction.py            ← /interact NLU + Gemma3 response + bilingual
    ├── training_exporter.py      ← /export_training → JSONL datasets
    ├── cleanup.py                ← Scheduled data cleanup & weight decay
    └── spatial.py                ← MiDaS depth estimation（停用中，Unity 模式用 user_pos 替代）
```

---

## 🔄 Perception Pipeline

```
Unity MultiImagePayload
  { image_list, userID, activity, room_name, user_pos, source_nodes }
        ↓
preview_images()  →  debug_images/

        ↓
Stage 1: perception.analyze_action_burst()
  - room_name → MongoDB scene_snapshots → 注入 VLM prompt 家具清單
  - Gemma3:4b 每幀輸出:
      action / main_object / interacting_items
      scene_items / spatial_relations / description
  - 多幀投票決定最終結果

        ↓
Stage 2: memory.bind_and_update()
  A. 家具綁定: est_pos 距離搜尋 + SBERT 語義驗證（限同房間）
  B. 物品三軌:
     scene_snapshots  ← 所有物品（家具表面有什麼）
     observation_logs ← 互動物品（這個人用了什麼）
     dynamic_objects  ← 動態物件（seen_count / interact_count / last_seen_on）
  C. 空間關係存入 scene_snapshots.spatial_relations
  D. activity_sequences 即時 append + transition 計算

        ↓
Stage 3: vector_memory.add_memory() + upsert_dynamic_object()
  習慣記憶文字:
    "User_Mom drinking near kitchen_counter with cup.
     apple on kitchen_counter. cup in_hand_of user_mom."
  動態物件文字:
    "apple on kitchen_counter in Kitchen. seen 8 times. interacted 3 times."
  分別存入兩個 FAISS index + metadata

        ↓
回傳 JSON response → Unity / 機器人執行
```

---

## 🧠 VLM Model

**Current model: `gemma3:4b` via Ollama**

| | LLaVA-phi3 | Gemma3:4b |
|---|---|---|
| JSON 輸出穩定度 | ⚠️ 常跑版 | ✅ 穩定 |
| 中文指令理解 | ❌ 差 | ✅ 好 |
| 空間關係描述精度 | ⚠️ 普通 | ✅ 較精準 |
| Context Window | 短 | ✅ 128K |
| VRAM 需求 | ~4GB | ~4GB |
| Ollama API 相容 | ✅ | ✅ |

```bash
ollama pull gemma3:4b
```

```python
# config.py
OLLAMA_MODEL = "gemma3:4b"
```

---

## 📊 VLM Prompt Output Format

每幀 Gemma3:4b 輸出：

```json
{
  "action": "drinking",
  "main_object": "kitchen_counter",
  "interacting_items": ["cup"],
  "scene_items": ["apple", "cutting_board", "knife"],
  "spatial_relations": [
    { "subject": "cup",   "relation": "in_hand_of", "object": "user_mom" },
    { "subject": "apple", "relation": "on",         "object": "kitchen_counter" }
  ],
  "description": "Person is drinking water near the kitchen counter."
}
```

**支援的空間介係詞：** `on` / `in` / `next_to` / `above` / `below` / `in_hand_of` / `on_top_of`

> `activity` 欄位是 Unity 模擬的 ground truth，**只用於評估，不傳給 VLM。**

---

## 🗄️ MongoDB Data Structure

**資料庫名稱：** `robot_rag_db`

### scene_snapshots（家具層）

Unity `/scene` 同步建立，每次 `/predict` 累積更新。

| 欄位 | 說明 | 範例 |
|------|------|------|
| `id` | 家具唯一 ID | `12345` |
| `label` | 家具名稱 | `kitchen_counter` |
| `pos` | 2D 座標 [x, z] | `[3.2, 1.5]` |
| `room` | 所在房間 | `Kitchen` |
| `items` | 歷史累積物品（只增不減） | `["cup", "apple"]` |
| `current_contents` | 當前鏡頭物品（每次覆蓋） | `["cup"]` |
| `spatial_relations` | 空間關係陣列 | `[{subject, relation, object}]` |
| `spatial_counts` | 各關係出現次數 | `{"apple\|on\|kitchen_counter": 8}` |
| `last_observation` | 最後觀測時間 | `2025-01-01T12:00:00` |

### observation_logs（用戶行為層）

記錄「誰」在「哪裡」做了「什麼」。`weight` 每次 +1，代表習慣強度。

| 欄位 | 說明 |
|------|------|
| `user` | 用戶 ID，例如 `User_Mom` |
| `instance` | 家具 label |
| `action` | 動作，例如 `drinking` |
| `weight` | 累積觀測次數（習慣強度） |
| `interacting_items` | 互動物品清單（addToSet） |
| `observed_relations` | 此行為中的空間關係 |
| `pos` | 家具座標（供機器人導航） |
| `last_seen` | 最後觀測時間 |
| `raw_vlm_desc` | VLM 原始描述 |

### dynamic_objects（動態物件層）

VLM 觀測到的非家具物品，記錄位置與使用頻率。唯一鍵為 `label`，last seen 覆蓋。

| 欄位 | 說明 | 範例 |
|------|------|------|
| `label` | 物品名稱（唯一鍵） | `apple` |
| `room` | last seen 的房間 | `Kitchen` |
| `last_seen_on` | last seen 的家具 | `kitchen_counter` |
| `spatial_rel` | 與家具的空間關係 | `on` |
| `furniture_pos` | 家具座標（導航用） | `[3.2, 1.5]` |
| `seen_count` | 在畫面中出現次數 | `8` |
| `interact_count` | 被人直接使用次數 | `3` |
| `interacted_by` | 使用過的用戶（addToSet） | `["User_Mom"]` |
| `first_seen` | 首次觀測時間 | `2026-01-01T...` |
| `last_seen` | 最後觀測時間 | `2026-03-08T...` |

### semantic_memories（感知日誌）

每次 `/predict` 的完整感知快照，只增不改，用於訓練資料匯出。

### activity_sequences（行為時間序列）

每天一份文件，記錄用戶當天的行為序列與行為轉換（transitions）。用於滑動視窗訓練資料生成。

### conversation_logs（對話紀錄）

每次 `/interact` 的完整對話快照，記錄 query、answer、nav_target、用戶選擇。

---

## 🔍 FAISS Vector Memory

兩個獨立 FAISS index：

**習慣記憶（robot_memory.index）**

```
"User_Mom drinking near kitchen_counter with cup.
 apple on kitchen_counter. cup in_hand_of user_mom."
```

**動態物件（dynamic_memory.index）**

```
"apple on kitchen_counter in Kitchen. seen 8 times. interacted 3 times. used by User_Mom."
```

查詢範例：
- `search_habit("媽媽喝水的地方")` → 行為記憶
- `search_dynamic("甜的東西在哪")` → apple / cookie 向量召回
- `search_dynamic("媽媽常用的東西", user_filter="User_Mom")` → 只回傳 User_Mom 互動過的

**Metadata 欄位（習慣記憶）：**

| 欄位 | 用途 |
|------|------|
| `user / action / instance` | 基本感知結果 |
| `interacting_items` | 互動物品 |
| `all_items` | 畫面全部物品 |
| `spatial_relations` | 空間介係詞關係 |
| `furniture_pos` | 家具座標（導航用） |
| `mongo_id` | 對應 MongoDB document |
| `memory_text` | 完整向量化文字 |
| `timestamp` | 寫入時間 |

---

## 💬 Interaction Engine

Terminal 啟動方式：

```bash
python3 interact_client.py
```

**流程：**

```
用戶輸入問題
    ↓
POST /interact
    ↓
FAISS search_habit()    → 行為記憶
FAISS search_dynamic()  → 物品位置
MongoDB 補最新座標
Gemma3:4b 生成回答（中英文化，問中回中，問英回英）
    ↓
顯示回答 + 選項：
  1. 導航到位置
  2. 只告訴我位置
  3. 取消
    ↓
POST /interact/confirm { choice }
    ↓
記錄進 conversation_logs
```

---

## ⚖️ Habit Weight System

`observation_logs.weight` 每次觀測到相同 `(user, instance, action)` 時 +1。

| 功能 | 說明 |
|------|------|
| **位置預測** | 預測用戶最可能去的地點（weight 排名） |
| **主動備料** | 根據高權重習慣提前準備物品 |
| **異常偵測** | 低權重行為組合觸發警告 |
| **導航目標** | 直接取 `furniture_pos` 給機器人導航 |

```python
# config.py
HABIT_DECAY_FACTOR = 0.95   # 每次清理乘以此係數（5% 衰減）
HABIT_MIN_WEIGHT   = 1.0    # 低於此值視為遺忘，自動刪除
```

---

## 🌐 Scene Graph

基於 [3D Dynamic Scene Graphs (Rosinol, MIT-SPARK 2020)](https://arxiv.org/abs/2002.06289) 設計，分四層：

```
L4 · Agents     👤 User_Mom          👤 User_Dad
                     |                    |
L3 · Rooms      🍳 Kitchen         🛋 LivingRoom    🛏 Bedroom
                  |      |              |      |
L2 · Furniture  🪵Counter  🚿Sink   🛋Sofa  ☕Table   🛏Bed
                  |    |         |      |
L1 · Items     🍎apple 🥤cup  📺remote 📖magazine
```

| Layer | 節點類型 | 資料來源 |
|-------|----------|----------|
| L4 · Agents | 用戶 | `observation_logs` |
| L3 · Rooms | 房間 | `scene_snapshots.room` |
| L2 · Furniture | 家具 | `scene_snapshots` |
| L1 · Items | 物品 | `dynamic_objects` + `spatial_relations` |

**Edge 類型：** `in_room` / `contains` / `on` / `in_hand_of` / `next_to`
**Edge 粗細：** 代表習慣 weight

---

## 📦 AI Training Data Export

呼叫 `POST /export_training` 從 MongoDB 自動匯出到 `training_data/`：

| 檔案 | 用途 | 格式 |
|------|------|------|
| `perception_data.jsonl` | VLM fine-tuning | LLaVA instruction format |
| `dialogue_data.jsonl` | 對話模型訓練 | OpenAI chat format |
| `navigation_data.jsonl` | 路徑規劃訓練 | state-action pairs |
| `scene_graph_data.jsonl` | 空間關係 QA | Question-Answer pairs |
| `habit_sequence_data.jsonl` | 行為序列預測 | 滑動視窗 window=3 |

```bash
POST /export_training  { "type": "all" }
POST /export_training  { "type": "habit" }
POST /export_training  { "userID": "User_Mom" }
```

---

## 🌐 API Routes

| Route | Method | 說明 |
|-------|--------|------|
| `/predict` | POST | 主感知路由：影像 → VLM → MongoDB → FAISS |
| `/scene` | POST | Unity 場景同步，建立 scene_snapshots |
| `/query` | POST | RAG 查詢：FAISS 搜尋 + MongoDB 補最新座標 |
| `/interact` | POST | 人機對話：NLU + Gemma3 回答 + 選項 |
| `/interact/confirm` | POST | 用戶選擇確認（導航 / 僅告知 / 取消） |
| `/log_navigation` | POST | 接收 Unity 導航路徑 → navigation_logs |
| `/export_training` | POST | 匯出 JSONL 訓練資料 |
| `/cleanup` | POST | 手動觸發資料清理與 weight decay |
| `/cleanup/status` | GET | 查詢各 collection 筆數 |

---

## ⚙️ Configuration

```python
# config.py

# VLM
OLLAMA_URL   = "http://localhost:11434"
OLLAMA_MODEL = "gemma3:4b"

# MongoDB
MONGO_URI = "mongodb://127.0.0.1:27017/"
DB_NAME   = "robot_rag_db"

# FAISS
FAISS_INDEX_PATH         = "robot_memory.index"
FAISS_META_PATH          = "robot_memory_meta.json"
DYNAMIC_INDEX_PATH       = "dynamic_memory.index"
DYNAMIC_META_PATH        = "dynamic_memory_meta.json"
MAX_FAISS_VECTORS        = 5000

# Cleanup
CLEANUP_RETAIN_DAYS      = 90
CLEANUP_INTERVAL_HOURS   = 24

# Habit Weight Decay
HABIT_DECAY_FACTOR       = 0.95
HABIT_MIN_WEIGHT         = 1.0
```

---

## 🚀 Quick Start

```bash
# 1. 安裝 Python 依賴
pip install flask pymongo faiss-cpu sentence-transformers numpy opencv-python requests

# 2. 啟動 Ollama 並下載模型
ollama serve
ollama pull gemma3:4b

# 3. 啟動 MongoDB（預設 port 27017）

# 4. 啟動 Flask server
python app.py

# 5. （選用）啟動 terminal 對話介面
python interact_client.py
```

Unity 端先送 `/scene` 同步場景，再送 `/predict` 開始感知。

---

## 📚 References

- [3D Dynamic Scene Graphs: Actionable Spatial Perception with Semantic Landmarks](https://arxiv.org/abs/2002.06289) — Rosinol et al., MIT-SPARK Lab, 2020
- [Gemma 3 Technical Report](https://arxiv.org/abs/2503.19786) — Google DeepMind, 2025
- [FAISS](https://github.com/facebookresearch/faiss) — Facebook AI Similarity Search
- [Sentence Transformers](https://www.sbert.net/) — paraphrase-MiniLM-L6-v2
