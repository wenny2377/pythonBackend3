
# 🤖 Robot Intelligence Backend: Multimodal Perception & Memory Pipeline

本後端系統採用 **Flask** 框架，作為 Home Service Robot 的智慧核心，整合了身份辨識、視覺語言模型 (VLM) 以及基於 MongoDB 與 FAISS 的雙軌記憶系統，達成高效的無感學習 (Implicit Learning) 與語義檢索 (RAG)。

## 🏗️ 系統架構 (Architecture)

本系統將機器人大腦拆分為核心層級，確保感知、推理與記憶的解耦協作：

### 1. 感知層 (Perception Layer)

* **InsightFace**: 提取 512 維特徵向量進行餘弦相似度比對，精準鎖定使用者身分（如：**User_Mom**）。
* **Gemma 3 (via Ollama)**: 利用視覺語言模型分析影像內容，產生「sitting」、「typing」等語義行為標籤。

### 2. 空間層 (Spatial Layer)

* **Matrix Inversion**: 將 Unity 傳來的 $4 \times 4$ 相機外部參數矩陣 (Extrinsics) 轉化為世界座標系中的實體位置。
* **Instance Mapping**: 在 MongoDB 中透過歐幾里得距離搜尋最靠近機器人的家具實例，解決動畫切換 (T-Pose) 造成的空間誤差。

### 3. 記憶層 (Memory Layer)

* **Deterministic Memory (MongoDB)**: 儲存家具位置 (`scene_snapshots`) 與使用者行為習慣權重 (`observation_logs`)，作為無感學習的基礎。
* **🧠 Semantic Memory (FAISS)**: 提供語義級別的聯想記憶。系統不需完全匹配關鍵字，而是透過向量相似度搜尋最接近的歷史行為紀錄。
* **Encoding**: 使用 `SentenceTransformer` 將 VLM 文字描述轉化為 384 維向量。
* **Indexing**: 存入 FAISS，並將其與 MongoDB 中的語義日誌 ID 綁定。
* **Retrieval**: 透過 L2 距離進行近鄰搜尋，找出語義最相關的行為片段。



---

## 📂 檔案結構 (Project Structure)

```text
Robot-Intelligence-Backend/
├── app.py                 # 程式入口 (API Routes & RAG Query)
├── config.py              # 全域配置 (Models, URLs, DB Connection)
├── core/
│   ├── perception.py      # 感知模組: FaceID 辨識與 VLM 分析邏輯
│   ├── spatial.py         # 空間模組: 矩陣運算與最近鄰鎖定算法
│   ├── memory.py          # 記憶模組: MongoDB 資料處理與權重更新
│   └── vector_db.py       # 向量模組: FAISS 索引與語義搜尋
├── faces/                 # 存放已知使用者人臉數據庫
├── robot_memory.index     # FAISS 向量索引檔
└── requirements.txt       # 環境依賴清單

```

---

## 📡 API 端點說明 (API Endpoints)

#### 1. 行為觀察與學習 (`POST /predict`)

* **用途**: 處理機器人拍下的即時影像並進行學習。
* **流程**: 接收影像 ➔ 辨識身分 ➔ 行為分析 ➔ 空間鎖定 ➔ 語義向量化存入 FAISS ➔ 更新 MongoDB 權重。

#### 2. 動態場景同步 (`POST /scene`)

* **用途**: 初始化或定期更新機器人的「世界觀」。
* **流程**: 接收來自 Unity `ProxyExportManager` 的家具資料，確保虛擬與實體空間同步。

#### 3. 個性化對話查詢 (`GET /query`)

* **用途**: 實作語義檢索 (RAG)，回答使用者的個性化習慣問題。
* **流程**: 意圖提取 ➔ FAISS/MongoDB 聯合檢索 ➔ 回傳最符合使用者偏好的回覆與座標。

---

## 🚀 論文研究亮點 (Research Highlights)

* **無感學習機制 (Implicit Learning)**: 機器人透過日常觀察自動紀錄使用者習慣，無需手動標記，達成智慧化環境適應。
* **雙軌記憶檢索**: 結合 MongoDB 的結構化數據與 FAISS 的非結構化語義向量，提升機器人對人類行為理解的深度。
* **多模態 RAG 應用**: 結合視覺、空間幾何與 LLM 推理，使機器人具備從「看到」到「理解」再到「決策」的完整能力環。

這是一個完整的**「從視覺感知到語義檢索」**的閉環邏輯。為了讓你應對論文 5.2 的實作描述與 5.3 的分析，我們將整個流程分為**「平時的觀察積累（存入）」**與**「當下的問答推理（提取）」**兩個階段。

---

### 1. 觀察積累階段：從 Unity 到 MongoDB & FAISS

當機器人在場景中巡邏，`ProxyExportManager` 傳出數據後，後端的處理如下：

1. **感知 (Perception)**：
* **InsightFace**: 識別出 `User_Mom`。
* **Gemma 3**: 識別出動作 `Sitting`（坐著）。


2. **空間轉化 (Spatial)**：
* 利用相機矩陣將照片中的像素點轉為 Unity 世界座標 `(2.5, 0, -1.2)`。


3. **實體綁定 (Instance Mapping)**：
* Python 查詢 **MongoDB (`scene_snapshots`)**：找到距離該座標最近的家具為 `Sofa_01`。


4. **雙重儲存 (Hybrid Storage)**：
* **MongoDB (`observation_logs`)**: 存入結構化日誌（包含座標、家具 ID、時間、使用者、動作標籤）。
* **FAISS (Vector Index)**：將動作標籤 `Sitting` 轉化為 384 維向量，並與該筆 MongoDB 的 `_id` 綁定。



---

### 2. 用戶問答階段：語義檢索與空間引導

當使用者（例如 Mom）說：**「我累了，想休息。」**

#### Step A: 意圖解析 (Intent Analysis)

* **Gemma 3** 接收文字，判斷意圖為「找地方坐下/休息」，輸出語義關鍵字：`"Resting"`, `"Sitting"`。

#### Step B: 語義檢索 (FAISS Retrieval)

* 系統將 `"Resting"` 轉為向量，在 **FAISS** 中搜尋與其最接近的歷史動作。
* FAISS 回傳數個相關的 `observation_logs` ID（這些 ID 指向過去 Mom 坐下的紀錄）。

#### Step C: 數據聚合與過濾 (MongoDB Query)

* Python 拿著這些 ID 去 **MongoDB** 撈取資料，並根據 `user_id: "User_Mom"` 進行過濾。
* **邏輯運算**：系統發現 Mom 在 `Sofa_01` 坐下的頻率最高（權重最高）。

#### Step D: 座標合成與執行 (Execution)

* Python 從 `scene_snapshots` 確認 `Sofa_01` 目前最新的世界座標。
* **回傳 Unity**：`ProxyExportManager` 接收到目標座標 `(2.5, 0, -1.2)`。
* **機器人動作**：`RobotPatro.cs` 導航至該處，並對 Mom 說：「這是我為您準備的休息位置。」

---

### 3. 整體架構數據串接表 (論文 5.2 核心)

這份表格能幫你理清 Python Backend 內部的資料交換：

| 階段 | 輸入來源 | 處理模組 | 輸出至 (Target) | 儲存/查詢內容 |
| --- | --- | --- | --- | --- |
| **感知** | Unity 照片 | InsightFace / Gemma | **Internal State** | 使用者 ID + 動作語義 |
| **空間** | Unity 矩陣 | Matrix Inversion | **MongoDB** | 世界座標 $(x, y, z)$ |
| **記憶** | 語義標籤 | SentenceTransformer | **FAISS** | 384 維行為特徵向量 |
| **檢索** | 用戶語音 | FAISS + MongoDB | **Unity** | 目標家具座標 (Target Vector) |

---

### 4. 對接 5.3 節：這套架構如何產生論文圖表？

* **MRR (5.3.1)**：
* 測試 30 組「我累了」、「我渴了」等問句。
* 計算 FAISS 回傳的第幾個結果（Rank）是 Mom 真正想去的習慣點。


* **KDE 熱圖 (5.3.2)**：
* 從 **MongoDB (`observation_logs`)** 匯出所有 `action_label: "Sitting"` 的座標。
* 透過 Python 產出熱圖，展現 Mom 的語義重心（如：沙發區顏色最深）。



### 💡 總結

你的架構是一個典型的 **「語義映射空間 (Semantic-to-Spatial Mapping)」**。

1. **Unity** 負責提供物理真值。
2. **MongoDB** 負責記住「在哪裡」與「是什麼」。
3. **FAISS** 負責聯想「這句話是什麼意思」。

