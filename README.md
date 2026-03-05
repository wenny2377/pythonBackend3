# 🐍 Python AI Pipeline

智慧家庭感知後端 — Flask + LLaVA + MongoDB + FAISS

---

## 📁 專案結構

```
backend/
├── app.py                   # Flask 主程式，API 路由入口
├── config.py                # 環境設定（URL、DB 名稱等）
├── modules/
│   ├── perception.py        # VLM 感知引擎（LLaVA 多幀投票 + 空間關係抽取）
│   ├── memory.py            # MongoDB 記憶管理（家具綁定 + 物品雙軌記錄）
│   └── memory_vector.py     # FAISS 向量記憶（自然語言模糊查詢）
├── debug_images/            # Unity 傳來的影像自動存檔（自動建立）
├── robot_memory.index       # FAISS 向量索引（自動生成）
├── robot_memory_meta.json   # FAISS metadata（自動生成）
└── requirements.txt
```

---

## 🔧 環境需求

```bash
pip install flask pymongo opencv-python numpy \
            sentence-transformers requests faiss-cpu
```

**Ollama（本地 LLM）：**
```bash
ollama pull llava
ollama serve
```

**MongoDB：**
```bash
sudo systemctl start mongod
# 或
mongod --dbpath /data/db
```

---

## ⚙️ config.py

```python
class Config:
    OLLAMA_URL   = "http://127.0.0.1:11434"
    OLLAMA_MODEL = "llava"
    MONGO_URI    = "mongodb://127.0.0.1:27017/"
    DB_NAME      = "robot_rag_db"
    FLASK_HOST   = "0.0.0.0"
    FLASK_PORT   = 5000
```

---

## 🚀 啟動

```bash
ollama serve &
sudo systemctl start mongod
python app.py
```

成功啟動：
```
✅ MongoDB 2D Index ready
Robot Brain Server Running on 0.0.0.0:5000
```

---

## 🧠 感知流程（完整說明）

```
Unity MultiImagePayload
        ↓
preview_images()
→ 影像存至 debug_images/ 供確認

        ↓
Stage 1：perception.analyze_action_burst()
→ 依 node_scores 選最優 1–3 幀送 LLaVA
→ 每幀輸出：
    - action          （人在做什麼）
    - main_object     （人在哪個家具旁）
    - interacting_items（人直接接觸的物品）
    - scene_items     （畫面中其他所有物品）
    - spatial_relations（物品空間介係詞關係）
→ 多幀投票決定最終結果

        ↓
Stage 2：memory.bind_and_update()
→ A. 家具綁定：est_pos 距離搜尋候選 + SBERT 語義驗證
→ B. 物品雙軌記錄：
     scene_snapshots  ← 所有物品（家具表面有什麼）
     observation_logs ← 互動物品（這個人用了什麼）
→ C. 空間關係存入 scene_snapshots.spatial_relations

        ↓
Stage 3：vector_memory.add_memory()
→ 向量化文字：
   "User_Mom drinking near kitchen_counter
    with cup. apple on kitchen_counter. cup in_hand_of user_mom."
→ 存入 FAISS + metadata（含家具座標）

        ↓
回傳 JSON response → Unity / 機器人執行
```

---

## 📊 VLM Prompt 輸出格式

每幀 LLaVA 輸出：

```json
{
  "action": "drinking",
  "main_object": "kitchen_counter",
  "interacting_items": ["cup"],
  "scene_items": ["apple", "cutting_board", "knife"],
  "spatial_relations": [
    { "subject": "cup",           "relation": "in_hand_of", "object": "user_mom" },
    { "subject": "apple",         "relation": "on",         "object": "kitchen_counter" },
    { "subject": "knife",         "relation": "on",         "object": "knife_rack" },
    { "subject": "cutting_board", "relation": "on",         "object": "kitchen_counter" }
  ],
  "description": "Person is drinking water near the kitchen counter."
}
```

支援的空間介係詞：`on` / `in` / `next_to` / `above` / `below` / `in_hand_of` / `on_top_of`

---

## 🗄️ MongoDB 資料結構

資料庫名稱：`robot_rag_db`

### `scene_snapshots`（家具層）

Unity `/scene` 同步建立，每次 `/predict` 累積更新。

```json
{
  "id":    12345,
  "label": "kitchen_counter",
  "pos":   [3.2, 1.5],
  "room":  "Kitchen",
  "items": ["cup", "apple", "knife", "cutting_board"],
  "current_contents": ["cup", "apple"],
  "spatial_relations": [
    { "subject": "apple", "relation": "on",    "object": "kitchen_counter" },
    { "subject": "cup",   "relation": "next_to","object": "sink" }
  ],
  "spatial_counts": {
    "apple|on|kitchen_counter": 8,
    "cup|next_to|sink": 3
  },
  "last_observation": "2025-01-01T12:00:00"
}
```

### `observation_logs`（用戶行為層）

記錄「誰」在「哪裡」做了「什麼」，用於習慣學習與頻率統計。

```json
{
  "user":               "User_Mom",
  "instance":           "kitchen_counter",
  "action":             "drinking",
  "weight":             12,
  "interacting_items":  ["cup", "water_bottle"],
  "observed_relations": [
    { "subject": "cup", "relation": "in_hand_of", "object": "user_mom" }
  ],
  "pos":         [3.2, 1.5],
  "last_seen":   "2025-01-05T14:32:00",
  "raw_vlm_desc":"Person is drinking water near the kitchen counter."
}
```

### `semantic_memories`（感知日誌）

每次 `/predict` 的原始感知結果快照，不更新只新增。

```json
{
  "user":         "User_Mom",
  "action":       "drinking",
  "bound_to":     "kitchen_counter",
  "details": {
    "interacting_items": ["cup"],
    "scene_items":       ["apple", "knife"],
    "spatial_relations": [...],
    "context":           "Person is drinking water..."
  },
  "source_nodes": ["Kitchen_Cam1", "Kitchen_Cam2"],
  "timestamp":    "2025-01-01T12:00:00Z"
}
```

---

## 🔍 FAISS 向量記憶設計

### 儲存格式

每筆記憶向量化的文字：
```
"User_Mom drinking near kitchen_counter with cup.
 apple on kitchen_counter. cup in_hand_of user_mom."
```

對應的 metadata：
```json
{
  "user":          "User_Mom",
  "action":        "drinking",
  "instance":      "kitchen_counter",
  "interacting_items": ["cup"],
  "all_items":     ["cup", "apple"],
  "spatial_relations": [...],
  "furniture_pos": [3.2, 1.5],
  "mongo_id":      "ObjectId..."
}
```

### 支援的查詢類型

| 查詢 | 機制 | 回傳 |
|------|------|------|
| 「媽媽通常在哪裡喝水？」| FAISS 向量搜尋 → 頻率聚合 | `nav_target` 座標 |
| 「刀子通常放在哪裡？」| spatial_relations 文字向量化 | 家具名稱 + 座標 |
| 「流理台上通常有什麼？」| instance 匹配 → MongoDB items | 物品清單 |
| 「媽媽用過什麼廚具？」| user + action 過濾 | `interacting_items` |

---

## 📡 API 路由

### `POST /predict`

接收 Unity MultiImagePayload，執行完整感知流程。

**Request：**
```json
{
  "image_list":        ["<base64_jpeg>", "<base64_jpeg>"],
  "image_count":       2,
  "source_nodes":      ["Kitchen_Cam1", "Kitchen_Cam2"],
  "node_scores":       [0.91, 0.74],
  "userID":            "User_Mom",
  "activity":          "drinking",
  "user_pos":          { "x": 3.2, "y": 0.0, "z": 1.5 },
  "timestamp":         "2025-01-01 12:00:00"
}
```

**Response：**
```json
{
  "status":            "Success",
  "user":              "User_Mom",
  "action":            "drinking",
  "bound_to":          "kitchen_counter",
  "interacting_items": ["cup"],
  "all_items":         ["cup", "apple", "knife"],
  "spatial_relations": [
    { "subject": "cup", "relation": "in_hand_of", "object": "user_mom" }
  ],
  "description":       "Person is drinking water near the kitchen counter.",
  "estimated_pos":     { "x": 3.2, "z": 1.5 },
  "furniture_pos":     [3.2, 1.5]
}
```

---

### `POST /scene`

Unity `ProxyExportManager` 呼叫，同步場景家具座標到 MongoDB。

**Request：**
```json
{
  "objects": [
    { "id": 12345, "label": "kitchen_counter", "x": 3.2, "y": 0.9, "z": 1.5, "room": "Kitchen" },
    { "id": 12346, "label": "fridge",          "x": 1.0, "y": 0.0, "z": 2.0, "room": "Kitchen" }
  ]
}
```

**Response：**
```json
{ "status": "Success", "synced_count": 2 }
```

---

### `POST /query`

自然語言查詢，RAG 回答 + 導航座標。

**Request：**
```json
{
  "query":  "Where does mom usually drink water?",
  "userID": "User_Mom"
}
```

**Response：**
```json
{
  "status":    "Success",
  "answer":    "Based on observations, User_Mom usually drinks near kitchen_counter (seen 12 times), typically interacting with: cup.",
  "nav_target": [3.2, 1.5],
  "top_habit": {
    "instance":          "kitchen_counter",
    "action":            "drinking",
    "count":             12,
    "interacting_items": ["cup", "water_bottle"],
    "furniture_pos":     [3.2, 1.5]
  },
  "semantic_results": [...]
}
```

---

## 🔁 定點相機模式（現行）

原始機器人架構中的兩個模組已停用，由 Unity 直接提供：

| 原始模組 | 用途 | 現行替代 |
|---------|------|---------|
| `InsightFace` | 從影像辨識用戶身份 | Unity `UserEntity.userID` 直接傳入 |
| `SpatialReasoning` | 相機矩陣反推世界座標 | Unity `transform.position` 直接傳入 |

如需恢復機器人模式，在 `PerceptionEngine.__init__()` 重新傳入 `face_analyzer` 和 `spatial_module` 即可，其餘邏輯不需修改。

---

## 🐛 常見問題

**MongoDB Connection Refused**
```bash
sudo systemctl start mongod
```

**Ollama 沒有回應**
```bash
ollama serve
ollama list        # 確認 llava 已下載
ollama pull llava  # 若沒有
```

**VLM 回傳不是 JSON**
`perception.py` 已自動清洗 ` ```json ``` ` 標記，若仍失敗會印 `JSON Parse Error` 並跳過該幀，不中斷整體流程。

**速度優化**

| 瓶頸 | 解法 |
|------|------|
| VLM 推理（主要瓶頸，每幀約 15s）| GPU 執行 Ollama |
| 傳送 3 張 | 調高 `singleViewThreshold`，多數情況只傳 1 張 |
| SBERT 家具驗證 | 候選家具 ≤ 1 個時自動跳過，不呼叫 SBERT |
