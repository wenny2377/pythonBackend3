import datetime
import requests
import json
import re
import logging

logger = logging.getLogger(__name__)

MAX_STEPS   = 3
LLM_TIMEOUT = 60
LLM_TEMP    = 0.3
LLM_TOKENS  = 300

# ── SBERT 意圖分類門檻（論文 §3.1）──────────────────────────────────────────
SBERT_THRESHOLD_CHAT    = 0.70
SBERT_THRESHOLD_QUERY   = 0.65
SBERT_THRESHOLD_SERVICE = 0.60

# ── ReAct system prompt（補空間推理邏輯，論文 §5.2）─────────────────────────
REACT_SYSTEM_PROMPT = """You are a home service robot assistant.

You have two sources of information:
- search_habit: what this user PREFERS based on past behavior (may be outdated)
- search_object: what is CURRENTLY in the home right now (ground truth from sensors)

Use BOTH to give the best answer:
  "User usually drinks juice (habit) + juice bottle is on table right now (object) → recommend juice on table"
  "User usually drinks juice (habit) + only cola available now (object) → recommend cola, mention preference"
  "No habit found + apple and banana available (object) → recommend what's available"

Tools:

1. search_habit(query: str)
   Find what this user usually does/uses. Returns past behavioral patterns.

2. search_object(query: str)
   Find what is currently in the home. Returns real-time object locations.
   Each result includes: label, last_seen_on (furniture), room, Camera_ID.
   Always use this to confirm something is actually available right now.

3. finish(answer: str, nav_target: str, nav_label: str)
   Give the final answer combining habit + current availability.
   - answer: natural sentence in the SAME language as the user
   - nav_target: furniture label where the recommended object is (e.g. "table") or "unknown"
   - nav_label: same as nav_target

SPATIAL REASONING RULES (具身空間推理):
- Always check the room field in search_object results
- If the object is NOT in the robot's current room, nav_target MUST be set to the furniture
  label where the object was seen (do NOT set "unknown")
- Example: robot is in bedroom, apple is in kitchen on table → nav_target="table", nav_label="table"
- If the object IS in the current room, nav_target is still the furniture label

DECISION FLOW:
- Conversational (chat/greeting/opinion) → finish immediately, no tools, nav_target="unknown"
- Service request → search_habit + search_object → finish with best recommendation
- If object from habit is currently available → recommend it
- If object from habit is NOT available → recommend what IS available, mention the preference
- If nothing found → be honest, ask user what they need

LIMITS: max 3 tool calls, no repeated tools, always end with finish.
Respond ONLY with valid JSON: {"thought": "...", "tool": "...", "input": {...}}
"""


# ── Tool executor ─────────────────────────────────────────────────────────────
class ToolExecutor:
    def __init__(self, vector_memory, db):
        self.vector = vector_memory
        self.db     = db

    def execute(self, tool_name: str, tool_input: dict) -> str:
        try:
            if tool_name == "search_habit":
                return self._search_habit(tool_input.get("query", ""))
            elif tool_name == "search_object":
                return self._search_object(tool_input.get("query", ""))
            elif tool_name == "get_user_history":
                return self._get_user_history(
                    tool_input.get("user_id", ""),
                    tool_input.get("action", "")
                )
            else:
                return f"Unknown tool: {tool_name}"
        except Exception as e:
            logger.error(f"Tool {tool_name} failed: {e}")
            return f"Tool error: {str(e)}"

    def _search_habit(self, query: str) -> str:
        results = self.vector.search_habit(query, top_k=3)
        if not results:
            return "No habit records found."
        lines = []
        for r in results:
            items = r.get("interacting_items", [])
            items_str = f", often uses: {items}" if items else ""
            lines.append(
                f"- {r.get('user','?')} usually {r.get('action','?')} "
                f"near {r.get('instance','?')} "
                f"(seen {r.get('weight', 1)} times{items_str})"
            )
        return "\n".join(lines)

    def _search_object(self, query: str) -> str:
        """
        三層搜尋（論文 §5.1 全知視角工具）：
        1. FAISS 語意搜尋（快速候選）
        2. MongoDB 最近看到的物件（TTL 過濾）
        3. 合併去重，回傳含 Camera_ID / Room_Name 的格式讓 LLM 做空間推理
        """
        from datetime import datetime, timedelta
        TTL_HOURS = 2

        faiss_results = self.vector.search_dynamic(query, top_k=5)
        seen_labels   = {r.get("label","") for r in faiss_results}

        cutoff = datetime.utcnow() - timedelta(hours=TTL_HOURS)
        recent_docs = list(self.db.dynamic_objects.find(
            {"last_seen": {"$gte": cutoff}},
            {"label":1,"room":1,"last_seen_on":1,
             "furniture_pos":1,"interact_count":1,"last_seen":1}
        ).sort("interact_count", -1).limit(12))

        if not recent_docs:
            recent_docs = list(self.db.dynamic_objects.find(
                {}, {"label":1,"room":1,"last_seen_on":1,
                     "furniture_pos":1,"interact_count":1}
            ).sort("interact_count", -1).limit(12))

        combined = list(faiss_results)
        for doc in recent_docs:
            label = doc.get("label","").lower()
            if label not in seen_labels:
                combined.append({
                    "label":         doc.get("label","?"),
                    "last_seen_on":  doc.get("last_seen_on","?"),
                    "room":          doc.get("room","?"),
                    "furniture_pos": doc.get("furniture_pos"),
                    "interact_count":doc.get("interact_count",0),
                    "similarity":    0.0,
                })
                seen_labels.add(label)

        if not combined:
            return "No objects currently visible in the home."

        # 格式包含 Room 讓 LLM 做空間推理（論文 §5.1）
        lines = []
        for r in combined[:8]:
            interact = r.get("interact_count", 0)
            freq     = f", used {interact}x" if interact > 0 else ""
            room     = r.get("room","?")
            lines.append(
                f"- {r.get('label','?')}: "
                f"on {r.get('last_seen_on','?')} "
                f"in Room={room}{freq}"
            )
        return "\n".join(lines)

    def _get_user_history(self, user_id: str, action: str) -> str:
        docs = list(self.db.observation_logs.find(
            {"user": user_id, "action": action},
            {"instance": 1, "weight": 1, "interacting_items": 1}
        ).sort("weight", -1).limit(3))
        if not docs:
            return f"No history for {user_id} doing {action}."
        return "\n".join(
            f"- {d.get('instance','?')}: "
            f"{d.get('weight', 0)} times, "
            f"items: {d.get('interacting_items', [])}"
            for d in docs
        )


# ── LLM helpers ───────────────────────────────────────────────────────────────
def _call_llm(ollama_url, model, system, user, max_tokens=LLM_TOKENS):
    try:
        resp = requests.post(
            f"{ollama_url}/api/chat",
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user}
                ],
                "stream": False,
                "options": {"temperature": LLM_TEMP, "num_predict": max_tokens}
            },
            timeout=LLM_TIMEOUT
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"].strip()
    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        return None


def _call_llm_json(ollama_url, model, system, user):
    raw = _call_llm(ollama_url, model, system, user, max_tokens=200)
    if not raw:
        return None
    try:
        clean = re.sub(r'```(?:json)?\s*', '', raw).strip().rstrip('`')
        match = re.search(r'\{.*\}', clean, re.DOTALL)
        if match:
            return json.loads(match.group(0))
    except json.JSONDecodeError as e:
        logger.warning(f"JSON parse failed: {e} | raw: {raw[:150]}")
    return None


# ── Main InteractionEngine ────────────────────────────────────────────────────
class InteractionEngine:

    def __init__(self, mongo_client, vector_memory, ollama_url, model_name):
        self.db            = mongo_client["robot_rag_db"]
        self.vector        = vector_memory
        self.ollama_url    = ollama_url
        self.model_name    = model_name
        self.conv_logs     = self.db["conversation_logs"]
        self.tool_executor = ToolExecutor(vector_memory, self.db)

        # SkillManager
        try:
            from modules.skill_manager import SkillManager
            self.skill_manager = SkillManager(
                db_client  = mongo_client,
                ollama_url = ollama_url,
                model_name = model_name,
            )
            self._has_skill_manager = True
            logger.info("[InteractionEngine] SkillManager loaded")
        except Exception as e:
            logger.warning(f"[InteractionEngine] SkillManager not available: {e}")
            self._has_skill_manager = False

        # SBERT 語義比對用（論文 §3.1）
        self._sbert = None
        self._sbert_templates = None
        self._init_sbert()

    # ── SBERT 初始化（懶載入，避免佔 VRAM）─────────────────────────────────
    def _init_sbert(self):
        try:
            # 重用 vector_memory 已載入的 SBERT，不重複佔 VRAM
            if hasattr(self.vector, 'model'):
                self._sbert = self.vector.model
                self._sbert_templates = self._build_sbert_templates()
                logger.info("[IntentClassify] SBERT loaded from VectorMemory")
        except Exception as e:
            logger.warning(f"[IntentClassify] SBERT init failed: {e}")

    def _build_sbert_templates(self):
        """
        三類代表句子（論文 §3.1 意圖模板）
        每類 3 句，取平均向量作為類別代表
        """
        import numpy as np
        templates = {
            "chat": [
                "你好，我好累，今天天氣真好",
                "謝謝你，你好棒，再見",
                "hello how are you thank you goodbye",
            ],
            "query": [
                "鑰匙在哪裡，冰箱有牛奶嗎，客廳有人嗎",
                "家裡有什麼吃的，爸爸在哪，現在幾度",
                "where is the key, is there milk in the fridge",
            ],
            "service": [
                "幫我拿水，帶我去沙發，打開燈",
                "我餓了，幫我倒咖啡，把杯子拿到廚房",
                "get me water, take me to the sofa, turn on the light",
            ],
        }
        vecs = {}
        for intent, sentences in templates.items():
            encoded = self._sbert.encode(sentences, normalize_embeddings=True)
            vecs[intent] = encoded.mean(axis=0)
        return vecs

    # ── 快速意圖分類（關鍵字優先 → SBERT 備援，論文 §3.1）─────────────────
    #
    # 決策優先順序：
    # 1. 中斷關鍵字 → interrupt
    # 2. 動作關鍵字 → service
    # 3. 查詢關鍵字 → query
    # 4. 生理需求   → service（預設）
    # 5. SBERT 語義比對（未命中時）
    # 6. 預設 chat
    #
    INTERRUPT_KEYWORDS_ZH = {"停下", "停止", "算了", "取消", "不用了"}
    INTERRUPT_KEYWORDS_EN = {"stop", "cancel", "never mind", "forget it"}

    ACTION_KEYWORDS_ZH = {"幫我", "拿去", "帶我", "幫忙", "帶到", "送到", "打開", "關掉",
                          "開燈", "關燈", "倒", "煮", "準備", "導航", "去拿"}
    ACTION_KEYWORDS_EN = {"get me", "bring me", "take me", "help me", "fetch",
                          "navigate to", "turn on", "turn off", "prepare", "make"}

    QUERY_KEYWORDS_ZH = {"在哪", "在哪裡", "有沒有", "有嗎", "有什麼", "幾個", "幾顆",
                         "過期", "什麼時候", "幾點", "多少", "哪裡", "在嗎", "看到"}
    QUERY_KEYWORDS_EN = {"where is", "where are", "is there", "do we have",
                         "how many", "what is available", "show me", "list",
                         "what food", "what drink", "options", "choices"}

    PHYSICAL_NEED_ZH = {"餓了", "渴了", "累了想坐", "想喝", "想吃", "想睡"}
    PHYSICAL_NEED_EN = {"i'm hungry", "i'm thirsty", "i want to eat", "i want to drink"}

    def _classify_intent(self, query: str) -> str:
        """
        回傳: 'interrupt' | 'chat' | 'query' | 'service'

        論文 §3.1 Intent Disambiguation with Hybrid Routing：
        - Step 1: 關鍵字快速匹配（< 1ms）
        - Step 2: SBERT 語義比對（關鍵字未命中時，~10ms）
        """
        q = query.lower().strip()

        # Step 1a：中斷關鍵字（最高優先級）
        if any(kw in q for kw in self.INTERRUPT_KEYWORDS_ZH):
            return "interrupt"
        if any(kw in q for kw in self.INTERRUPT_KEYWORDS_EN):
            return "interrupt"

        # Step 1b：動作關鍵字 → service
        if any(kw in q for kw in self.ACTION_KEYWORDS_ZH):
            return "service"
        if any(kw in q for kw in self.ACTION_KEYWORDS_EN):
            return "service"

        # Step 1c：查詢關鍵字 → query
        if any(kw in q for kw in self.QUERY_KEYWORDS_ZH):
            return "query"
        if any(kw in q for kw in self.QUERY_KEYWORDS_EN):
            return "query"

        # Step 1d：生理需求 → service（預設）
        if any(kw in q for kw in self.PHYSICAL_NEED_ZH):
            return "service"
        if any(kw in q for kw in self.PHYSICAL_NEED_EN):
            return "service"

        # Step 2：SBERT 語義比對（關鍵字都未命中）
        if self._sbert and self._sbert_templates:
            sbert_result = self._classify_by_sbert(query)
            if sbert_result:
                return sbert_result

        # 預設 chat
        return "chat"

    def _classify_by_sbert(self, query: str) -> str | None:
        """
        SBERT 語義比對：計算 query 與三類模板的餘弦相似度
        門檻：chat=0.70, query=0.65, service=0.60（論文 §3.1）
        """
        try:
            import numpy as np
            q_vec = self._sbert.encode(query, normalize_embeddings=True)

            scores = {}
            for intent, template_vec in self._sbert_templates.items():
                sim = float(np.dot(q_vec, template_vec))
                scores[intent] = sim

            logger.debug(f"[SBERT] scores={scores}")

            # 按門檻值判斷（高門檻優先）
            if scores.get("chat", 0) >= SBERT_THRESHOLD_CHAT:
                return "chat"
            if scores.get("query", 0) >= SBERT_THRESHOLD_QUERY:
                return "query"
            if scores.get("service", 0) >= SBERT_THRESHOLD_SERVICE:
                return "service"

            # 沒有任何類別達到門檻 → 取最高分
            best = max(scores, key=scores.get)
            logger.debug(f"[SBERT] no threshold met, best={best}")
            return best

        except Exception as e:
            logger.warning(f"[SBERT classify] failed: {e}")
            return None

    # ── 主入口 ────────────────────────────────────────────────────────────────
    def process(self, query, user_id="Unknown", robot_pos=None,
                user_pos=None, room=""):

        print(f"\n[Interact] user={user_id} | query='{query}' | room={room}")

        intent = self._classify_intent(query)
        print(f"[Classify] intent={intent}")

        # 中斷指令
        if intent == "interrupt":
            return self._interrupt_response(query, user_id)

        # 純聊天 → 不進 ReAct，直接 LLM 回應
        if intent == "chat":
            return self._chat_response(query, user_id)

        # 查詢類 → MongoDB 直查（論文 §2.2）
        if intent == "query":
            return self._query_response(query, user_id, room)

        # 執行類（service）→ ReAct + FAISS
        if self._has_skill_manager:
            try:
                result = self._react_process(query, user_id, room)
                if result:
                    self._log_conversation(
                        query=query, expanded_query=query,
                        intent_type="react", user_id=user_id,
                        answer=result["answer"],
                        nav_target=result["nav_target"],
                        nav_label=result["nav_label"],
                        room=room, recommendations=[],
                        is_personalized=True,
                    )
                    return result
            except Exception as e:
                logger.warning(f"[ReAct] Failed, falling back: {e}")

        # Fallback pipeline
        logger.info("[Interact] Using fixed pipeline fallback")
        return self._pipeline_process(query, user_id, robot_pos, user_pos, room)

    # ── 中斷回應 ──────────────────────────────────────────────────────────────
    def _interrupt_response(self, query: str, user_id: str) -> dict:
        print(f"[Interrupt] Task cancelled by user")
        is_chinese = any('\u4e00' <= c <= '\u9fff' for c in query)
        answer = "好的，已停止。" if is_chinese else "Understood, stopping now."
        return {
            "status": "Interrupted", "answer": answer,
            "nav_target": None, "nav_label": None,
            "options": [{"id": 3, "label": "關閉" if is_chinese else "Close"}],
            "confidence": 1.0, "intent_type": "interrupt",
            "recommendations": [], "is_personalized": False,
        }

    # ── 聊天回應（不進 ReAct）────────────────────────────────────────────────
    def _chat_response(self, query: str, user_id: str) -> dict:
        print(f"[Chat] Responding directly without ReAct")
        answer = _call_llm(
            self.ollama_url, self.model_name,
            "You are a friendly home robot companion. "
            "Reply warmly and briefly in the same language as the user. "
            "1-2 sentences max. Do NOT suggest navigation or services.",
            f'{user_id} said: "{query}"'
        ) or "I'm here for you!"
        return {
            "status": "Success", "answer": answer,
            "nav_target": None, "nav_label": None,
            "options": [{"id": 3, "label": "Close"}],
            "confidence": 1.0, "intent_type": "chat",
            "recommendations": [], "is_personalized": False,
        }

    # ── 查詢回應（Query 類，MongoDB 直查，論文 §2.2）─────────────────────────
    def _query_response(self, query: str, user_id: str, room: str) -> dict:
        """
        Query 類處理：直接查 MongoDB，不走 ReAct loop
        包含物品位置、存在性、人物位置、空間狀態等查詢
        """
        print(f"[Query] Direct MongoDB lookup for: {query}")
        from datetime import datetime, timedelta

        q = query.lower()

        # ── 人物位置查詢 ──
        if any(kw in q for kw in ["在哪", "在嗎", "在客廳", "人在", "where is"]):
            # 查 user_positions collection
            users = list(self.db.user_positions.find(
                {}, {"user_id":1,"room":1,"updated_at":1}
            ))
            if users:
                is_chinese = any('\u4e00' <= c <= '\u9fff' for c in query)
                user_strs = []
                for u in users:
                    user_strs.append(f"{u.get('user_id','?')} 在 {u.get('room','?')}")
                answer = "、".join(user_strs) + "。" if is_chinese else \
                         ", ".join(f"{u.get('user_id','?')} is in {u.get('room','?')}" for u in users)
            else:
                answer = "目前沒有追蹤到家庭成員位置。"
            return {
                "status": "Success", "answer": answer,
                "nav_target": None, "nav_label": None,
                "options": [{"id": 3, "label": "關閉"}],
                "confidence": 0.9, "intent_type": "query",
                "recommendations": [], "is_personalized": False,
            }

        # ── 物品查詢（位置 / 存在性）──
        cutoff = datetime.utcnow() - timedelta(hours=2)
        docs = list(self.db.dynamic_objects.find(
            {"last_seen": {"$gte": cutoff}},
            {"label":1,"room":1,"last_seen_on":1,"interact_count":1}
        ).sort("interact_count", -1))

        if not docs:
            docs = list(self.db.dynamic_objects.find(
                {}, {"label":1,"room":1,"last_seen_on":1,"interact_count":1}
            ).sort("interact_count", -1).limit(15))

        # 排除人物
        EXCLUDE = {"user_mom","user_dad","user","person","people"}
        docs = [d for d in docs if d.get("label","").lower() not in EXCLUDE]

        is_chinese = any('\u4e00' <= c <= '\u9fff' for c in query)

        if not docs:
            answer = "目前家裡沒有偵測到相關物品。" if is_chinese else \
                     "No items currently detected in the home."
            return {
                "status": "Success", "answer": answer,
                "nav_target": None, "nav_label": None,
                "options": [{"id": 3, "label": "關閉" if is_chinese else "Close"}],
                "confidence": 0.5, "intent_type": "query",
                "recommendations": [], "is_personalized": False,
            }

        # 用 SBERT 找最相關的物品
        relevant_docs = []
        if self._sbert:
            try:
                import numpy as np
                q_vec = self._sbert.encode(query, normalize_embeddings=True)
                scored = []
                for d in docs:
                    label_vec = self._sbert.encode(d.get("label",""), normalize_embeddings=True)
                    sim = float(np.dot(q_vec, label_vec))
                    scored.append((d, sim))
                scored.sort(key=lambda x: x[1], reverse=True)
                relevant_docs = [d for d, s in scored[:5] if s > 0.3]
            except Exception:
                relevant_docs = docs[:5]
        else:
            relevant_docs = docs[:5]

        if not relevant_docs:
            relevant_docs = docs[:3]

        # 組回答
        if is_chinese:
            items_str = "、".join(
                f"{d['label']}（在{d.get('room','?')}的{d.get('last_seen_on','?')}上）"
                for d in relevant_docs[:3]
            )
            answer = f"我找到以下相關物品：{items_str}。"
        else:
            items_str = ", ".join(
                f"{d['label']} (on {d.get('last_seen_on','?')} in {d.get('room','?')})"
                for d in relevant_docs[:3]
            )
            answer = f"I found: {items_str}."

        nav_label  = relevant_docs[0].get("last_seen_on") if relevant_docs else None
        nav_target = self._resolve_pos(nav_label)

        return {
            "status": "Success", "answer": answer,
            "nav_target": nav_target, "nav_label": nav_label,
            "options": self._build_options(nav_target, nav_label, query),
            "confidence": 0.85, "intent_type": "query",
            "recommendations": [
                {"label": d["label"], "last_seen_on": d.get("last_seen_on"),
                 "room": d.get("room"), "score": 0.8}
                for d in relevant_docs[:4]
            ],
            "is_personalized": False,
        }

    # ── 列表回應（query 的清單子類型，論文 §2.2 清單查詢）───────────────────
    def _list_response(self, query: str, user_id: str) -> dict:
        print(f"[List] Fetching available options from MongoDB")
        from datetime import datetime, timedelta

        cutoff = datetime.utcnow() - timedelta(hours=2)
        docs = list(self.db.dynamic_objects.find(
            {"last_seen": {"$gte": cutoff}},
            {"label":1,"room":1,"last_seen_on":1,"furniture_pos":1,"interact_count":1}
        ).sort("interact_count", -1))

        if not docs:
            docs = list(self.db.dynamic_objects.find(
                {}, {"label":1,"room":1,"last_seen_on":1,"furniture_pos":1,"interact_count":1}
            ).sort("interact_count", -1).limit(15))

        EXCLUDE = {"user_mom","user_dad","user","person","people"}
        docs = [d for d in docs if d.get("label","").lower() not in EXCLUDE]

        if not docs:
            is_chinese = any('\u4e00' <= c <= '\u9fff' for c in query)
            answer = "目前家裡沒有找到任何物品。" if is_chinese else \
                     "I don't see any items available right now."
            return {
                "status": "Success", "answer": answer,
                "nav_target": None, "nav_label": None,
                "options": [{"id": 3, "label": "Close"}],
                "confidence": 0.5, "intent_type": "query",
                "recommendations": [], "is_personalized": False,
            }

        all_items_str = "\n".join(
            f"- {d['label']}: on {d.get('last_seen_on','?')} in {d.get('room','?')}"
            for d in docs
        )
        is_chinese = any('\u4e00' <= c <= '\u9fff' for c in query)
        filter_prompt = (
            f'User asked: "{query}"\n\n'
            f"All items currently in the home:\n{all_items_str}\n\n"
            f"Select ONLY items relevant to the user's request. "
            f"For food queries, only include edible items. "
            f"For drink queries, only include beverages. "
            f"Reply JSON only: "
            f'{{"relevant": ["label1", "label2", ...], "none": false}}\n'
            f'If nothing is relevant, set "none": true and "relevant": []'
        )
        filter_result = _call_llm_json(
            self.ollama_url, self.model_name,
            "You are a home robot filtering objects by relevance. Reply JSON only.",
            filter_prompt
        )

        relevant_labels = []
        if filter_result and not filter_result.get("none", False):
            relevant_labels = [l.lower() for l in filter_result.get("relevant", [])]

        label_to_doc = {d["label"].lower(): d for d in docs}
        filtered_docs = [label_to_doc[l] for l in relevant_labels if l in label_to_doc]

        if not filtered_docs:
            filtered_docs = docs[:6]

        habit_results = self.vector.search_habit(query, user_id=user_id, top_k=5)
        habit_items   = set()
        for h in habit_results:
            habit_items.update([i.lower() for i in h.get("interacting_items", [])])

        filtered_docs.sort(key=lambda d: (
            0 if d.get("label","").lower() in habit_items else 1,
            -d.get("interact_count", 0)
        ))

        items_str = ", ".join(
            f"{d['label']} ({d.get('last_seen_on','?')} in {d.get('room','?')})"
            for d in filtered_docs[:5]
        )
        if is_chinese:
            answer = f"目前可以吃/喝的有：{items_str}。您想要哪一個？"
        else:
            answer = f"Currently available: {items_str}. Which one would you like?"

        options = []
        for i, d in enumerate(filtered_docs[:4], 1):
            label = d.get("label","?")
            loc   = d.get("last_seen_on","?")
            options.append({"id": i, "label": f"Get {label} from {loc}"})
        options.append({"id": len(options)+1, "label": "Cancel"})

        nav_label  = filtered_docs[0].get("last_seen_on","unknown") if filtered_docs else None
        nav_target = self._resolve_pos(nav_label)

        return {
            "status": "Success", "answer": answer,
            "nav_target": nav_target, "nav_label": nav_label,
            "options": options,
            "confidence": 0.9, "intent_type": "query",
            "recommendations": [
                {"label": d["label"], "last_seen_on": d.get("last_seen_on"),
                 "room": d.get("room"), "score": 0.8}
                for d in filtered_docs[:4]
            ],
            "is_personalized": len(habit_items) > 0,
        }

    # ── ReAct 流程（Service 類）───────────────────────────────────────────────
    def _react_process(self, query, user_id, room):
        sm = self.skill_manager

        # 1. 讀取或生成 SKILL.md（使用 FAISS 切片版本，論文 §4.2）
        skill_md = sm.get_skill_chunks(user_id, query)  # Top-2 切片
        if not skill_md:
            skill_md = sm.get_skill(user_id)
        if not skill_md:
            skill_md = sm.generate(user_id)

        # 2. GapDetector
        has_gap, missing = sm.detect_gap(user_id, query)
        if has_gap and missing:
            print(f"[GapDetector] Missing: {missing} → fill_gap()")
            # 走非同步隊列，避免卡住即時回應（論文 §3.2）
            try:
                from app import submit_llm_task
                skill_md_updated = submit_llm_task(sm.fill_gap, user_id, query, missing)
                if skill_md_updated:
                    skill_md = skill_md_updated
            except ImportError:
                skill_md = sm.fill_gap(user_id, query, missing)

        # 3. 更新 last_used
        self.db.user_skills.update_one(
            {"user_id": user_id},
            {"$set": {"last_used": datetime.datetime.utcnow()}}
        )

        # 4. system prompt = ReAct + SKILL 切片（< 400 tokens）
        system = REACT_SYSTEM_PROMPT
        if skill_md:
            system += f"\n\n## This user's skill profile (relevant rules only)\n{skill_md}"

        # 5. 取得已知家具
        furniture_labels = [
            d["label"] for d in
            self.db.scene_snapshots.find({}, {"label": 1})
            if "label" in d
        ]
        furniture_str = ", ".join(furniture_labels) or "unknown"

        # 6. ReAct loop
        observations = []
        used_tools   = []

        for step in range(MAX_STEPS):
            obs_text = ""
            if observations:
                obs_text = "\n\nPrevious steps:\n"
                for o in observations:
                    obs_text += (
                        f"Step {o['step']}:\n"
                        f"  Thought: {o['thought']}\n"
                        f"  Tool: {o['tool']}({o['input']})\n"
                        f"  Result: {o['result']}\n"
                    )

            remaining = MAX_STEPS - step
            used_str  = f"Already used: {used_tools}" if used_tools else "No tools used yet"
            next_hint = "You MUST call finish now." if remaining == 1 else \
                        f"Remaining steps: {remaining} (including this one)"

            user_prompt = (
                f"User: {query}\n"
                f"User ID: {user_id}\n"
                f"Robot current room: {room}\n"
                f"Known furniture: {furniture_str}\n"
                f"Time: {datetime.datetime.now().strftime('%H:%M')}\n"
                f"{used_str}. {next_hint}"
                f"{obs_text}\n"
                f"What is your next action? Reply JSON only."
            )

            action = _call_llm_json(
                self.ollama_url, self.model_name, system, user_prompt
            )

            if action is None:
                logger.warning(f"[ReAct] Step {step+1}: LLM returned None")
                break

            thought    = action.get("thought", "")
            tool_name  = action.get("tool", "").strip()
            tool_input = action.get("input", {}) or {}

            print(f"[ReAct] Step {step+1} | tool={tool_name} | "
                  f"thought={thought[:60]}")

            if not tool_name:
                logger.warning(f"[ReAct] Step {step+1}: empty tool name")
                break

            if tool_name == "finish":
                answer     = tool_input.get("answer", "")
                nav_target = tool_input.get("nav_target", "unknown")
                nav_label  = tool_input.get("nav_label", nav_target)

                if not observations:
                    nav_target = None
                    nav_label  = None

                nav_pos = self._resolve_pos(nav_target) if nav_target and nav_target != "unknown" else None

                # RelevanceGate → 非同步更新 SKILL.md
                if sm.should_update(user_id, query, answer, observations):
                    print(f"[RelevanceGate] Queuing SKILL.md update...")
                    try:
                        from app import submit_llm_task
                        submit_llm_task(sm.update, user_id, query, answer, observations)
                    except ImportError:
                        sm.update(user_id, query, answer, observations)

                sm.check_stale(user_id)

                return {
                    "status":          "Success",
                    "answer":          answer,
                    "nav_target":      nav_pos or nav_target,
                    "nav_label":       nav_label,
                    "options":         self._build_options(nav_pos, nav_label, query),
                    "confidence":      0.8,
                    "intent_type":     "react",
                    "recommendations": [],
                    "is_personalized": True,
                    "react_steps":     step + 1,
                    "skill_version":   sm.get_version(user_id),
                }

            if tool_name in used_tools:
                print(f"[ReAct] Step {step+1}: '{tool_name}' already used → force finish")
                all_results = "\n".join(
                    f"From {o['tool']}: {o['result']}"
                    for o in observations
                    if o["result"] and "No " not in o["result"][:3]
                )
                if all_results:
                    summary_prompt = (
                        f'User said: "{query}"\n\n'
                        f"Information found:\n{all_results}\n\n"
                        f"Write ONE natural friendly sentence answering the user. "
                        f"Use the same language as the user. "
                        f"Do not copy raw data — summarize naturally. "
                        f"Also extract the most relevant furniture label for navigation "
                        f"(or 'unknown' if not applicable).\n"
                        f"Reply JSON only: "
                        f'{{"answer": "...", "nav_target": "table or unknown"}}'
                    )
                    summary = _call_llm_json(
                        self.ollama_url, self.model_name,
                        "You are a home robot. Summarize search results into a natural reply.",
                        summary_prompt
                    )
                    if summary and summary.get("answer"):
                        forced_answer = summary["answer"]
                        forced_nav    = summary.get("nav_target", "unknown")
                    else:
                        forced_answer = "I found some items that might help. Would you like me to navigate there?"
                        forced_nav    = "unknown"
                else:
                    forced_answer = "I couldn't find relevant information. Could you be more specific?"
                    forced_nav    = "unknown"

                nav_pos = self._resolve_pos(forced_nav)
                return {
                    "status": "Success", "answer": forced_answer,
                    "nav_target": nav_pos or forced_nav,
                    "nav_label": forced_nav,
                    "options": self._build_options(nav_pos, forced_nav, query),
                    "confidence": 0.5, "intent_type": "react",
                    "recommendations": [], "is_personalized": True,
                    "react_steps": step + 1,
                    "skill_version": sm.get_version(user_id),
                }

            used_tools.append(tool_name)
            result = self.tool_executor.execute(tool_name, tool_input)
            observations.append({
                "step":    step + 1,
                "thought": thought,
                "tool":    tool_name,
                "input":   tool_input,
                "result":  result,
            })

        # MAX_STEPS 用完 → LLM 總結
        logger.warning("[ReAct] Exceeded MAX_STEPS, summarizing observations")
        all_results = "\n".join(
            f"From {o['tool']}: {o['result']}"
            for o in observations
            if o["result"] and len(o["result"]) > 5
        )
        if all_results:
            summary_prompt = (
                f'User said: "{query}"\n\n'
                f"Information found:\n{all_results}\n\n"
                f"Write ONE natural friendly sentence answering the user. "
                f"Use the same language as the user. Do not copy raw data.\n"
                f"Also pick the most relevant furniture label for navigation.\n"
                f'Reply JSON only: {{"answer": "...", "nav_target": "table or unknown"}}'
            )
            summary = _call_llm_json(
                self.ollama_url, self.model_name,
                "You are a home robot. Summarize search results into a natural reply.",
                summary_prompt
            )
            if summary and summary.get("answer"):
                ans = summary["answer"]
                nav = summary.get("nav_target", "unknown")
                nav_pos = self._resolve_pos(nav)
                return {
                    "status": "Success", "answer": ans,
                    "nav_target": nav_pos or nav, "nav_label": nav,
                    "options": self._build_options(nav_pos, nav, query),
                    "confidence": 0.5, "intent_type": "react",
                    "recommendations": [], "is_personalized": True,
                    "react_steps": MAX_STEPS,
                    "skill_version": sm.get_version(user_id),
                }

        return None

    # ── 固定 pipeline fallback ─────────────────────────────────────────────────
    def _pipeline_process(self, query, user_id="Unknown", robot_pos=None,
                          user_pos=None, room=""):
        intent_result  = self._llm_analyze_intent(query, user_id)
        intent_type    = intent_result.get("intent_type", "direct")
        search_kw      = intent_result.get("search_keywords", [query])
        expanded_query = " ".join(search_kw)

        print(f"[Pipeline] intent={intent_type} | kw={search_kw}")

        habit_results    = self.vector.search_habit(expanded_query, user_id=user_id, top_k=5)
        dynamic_personal = self.vector.search_dynamic(expanded_query, top_k=5, user_filter=user_id)

        if len(dynamic_personal) < 2:
            dynamic_all = self.vector.search_dynamic(expanded_query, top_k=5)
            seen_labels = {r["label"] for r in dynamic_personal}
            for r in dynamic_all:
                if r["label"] not in seen_labels:
                    dynamic_personal.append(r)
                    seen_labels.add(r["label"])
        dynamic_results = dynamic_personal[:5]

        recommendations          = self._cross_match(habit_results, dynamic_results, user_id)
        nav_target, nav_label    = self._resolve_nav(recommendations, dynamic_results, habit_results)
        answer                   = self._generate_answer(
            query=query, intent_type=intent_type, user_id=user_id, room=room,
            habit_results=habit_results, dynamic_results=dynamic_results,
            recommendations=recommendations, nav_label=nav_label,
        )
        options    = self._build_options(nav_target, nav_label, query=query)
        confidence = (
            recommendations[0]["score"]    if recommendations else
            dynamic_results[0]["similarity"] if dynamic_results else
            habit_results[0]["similarity"]   if habit_results else 0.0
        )

        self._log_conversation(
            query=query, expanded_query=expanded_query,
            intent_type=intent_type, user_id=user_id, answer=answer,
            nav_target=nav_target, nav_label=nav_label, room=room,
            recommendations=recommendations,
            is_personalized=len(dynamic_personal) > 0,
        )

        return {
            "status": "Success", "answer": answer,
            "nav_target": nav_target, "nav_label": nav_label,
            "options": options, "confidence": round(confidence, 3),
            "intent_type": intent_type, "recommendations": recommendations,
            "is_personalized": len(dynamic_personal) > 0,
        }

    # ── 輔助 ──────────────────────────────────────────────────────────────────
    def _resolve_pos(self, nav_label):
        if not nav_label or nav_label == "unknown":
            return None
        doc = self.db.scene_snapshots.find_one({"label": nav_label})
        return doc["pos"] if doc and doc.get("pos") else None

    def _llm_analyze_intent(self, query, user_id):
        prompt = f"""Analyze the user's intent.
User ID: {user_id}
User said: "{query}"
Reply ONLY valid JSON:
{{"intent_type":"fuzzy_need|object_search|habit_query|direct","search_keywords":["kw1","kw2"],"need_category":"food|drink|rest|entertainment|medicine|object|person|other"}}"""
        try:
            resp = requests.post(
                f"{self.ollama_url}/api/generate",
                json={"model": self.model_name, "prompt": prompt, "stream": False,
                      "options": {"temperature": 0.1, "num_predict": 128}},
                timeout=30
            )
            if resp.status_code == 200:
                raw   = resp.json().get("response", "").strip()
                clean = re.sub(r'```(?:json)?\s*', '', raw).strip()
                match = re.search(r'\{.*\}', clean, re.DOTALL)
                if match:
                    return json.loads(match.group(0))
        except Exception as e:
            print(f"[Intent LLM Error] {e}")
        return {"intent_type": "direct", "search_keywords": [query], "need_category": "other"}

    def _cross_match(self, habit_results, dynamic_results, user_id):
        habit_items = {}
        for h in habit_results:
            fpos = h.get("furniture_pos")
            for item in h.get("interacting_items", []):
                item = item.lower()
                if item not in habit_items:
                    habit_items[item] = {"count": 0, "instance": h.get("instance", ""),
                                         "furniture_pos": fpos, "similarity": h.get("similarity", 0.0)}
                habit_items[item]["count"] += 1

        recommendations, seen = [], set()
        for d in dynamic_results:
            label = d.get("label", "").lower()
            if label in seen:
                continue
            seen.add(label)
            is_habitual    = label in habit_items
            habit_count    = habit_items[label]["count"] if is_habitual else 0
            interact_count = d.get("interact_count", 0)
            score = (min(interact_count/10.0,1.0)*0.4 +
                     min(habit_count/5.0,1.0)*0.4 +
                     d.get("similarity",0.0)*0.2)
            recommendations.append({
                "label": label, "room": d.get("room",""),
                "last_seen_on": d.get("last_seen_on",""),
                "furniture_pos": d.get("furniture_pos"),
                "interact_count": interact_count, "habit_count": habit_count,
                "is_habitual": is_habitual,
                "is_mine": user_id in d.get("interacted_by",[]),
                "score": round(score, 3),
            })

        recommendations.sort(key=lambda x: x["score"], reverse=True)
        for label, info in habit_items.items():
            if label not in seen:
                recommendations.append({
                    "label": label, "room": "", "last_seen_on": info["instance"],
                    "furniture_pos": info["furniture_pos"],
                    "interact_count": 0, "habit_count": info["count"],
                    "is_habitual": True, "is_mine": True,
                    "score": round(info["count"]/10.0*0.4, 3),
                    "note": "not_currently_visible",
                })
        return recommendations[:5]

    def _resolve_nav(self, recommendations, dynamic_results, habit_results):
        for source in [recommendations, dynamic_results, habit_results]:
            for r in source:
                label = r.get("last_seen_on") or r.get("instance") or r.get("label")
                pos   = r.get("furniture_pos")
                if label:
                    doc = self.db.scene_snapshots.find_one({"label": label})
                    if doc and doc.get("pos"):
                        return doc["pos"], label
                if pos and label:
                    return pos, label
        return None, None

    def _generate_answer(self, query, intent_type, user_id, room,
                         habit_results, dynamic_results, recommendations, nav_label):
        is_chinese = any('\u4e00' <= c <= '\u9fff' for c in query)
        parts = []
        if recommendations:
            parts.append("=== Recommended ===")
            for r in recommendations[:3]:
                parts.append(f"- {r['label']} at {r['last_seen_on']} ({r['room']}), "
                             f"interacted {r['interact_count']} times")
        if habit_results:
            parts.append("=== Habits ===")
            for h in habit_results[:2]:
                items = ", ".join(h.get("interacting_items",[])) or "nothing"
                parts.append(f"- {user_id} {h['action']} near {h['instance']}, using {items}")
        context = "\n".join(parts) if parts else "No relevant memory."
        lang    = "Traditional Chinese" if is_chinese else "English"
        space   = ""
        if not recommendations and dynamic_results:
            space = "Items in home: " + ", ".join(
                f"{r['label']} on {r['last_seen_on']}" for r in dynamic_results[:3])
        prompt = (f"Home robot for {user_id}. Reply in {lang}. 1 sentence MAX.\n"
                  f"Memory:\n{context}\n{space}\nUser: \"{query}\"")
        try:
            resp = requests.post(
                f"{self.ollama_url}/api/generate",
                json={"model": self.model_name, "prompt": prompt, "stream": False,
                      "options": {"temperature": 0.3, "num_predict": 80}},
                timeout=30
            )
            if resp.status_code == 200:
                answer = resp.json().get("response", "").strip()
                if answer:
                    return answer
        except Exception as e:
            print(f"[LLM Error] {e}")
        if recommendations:
            top = recommendations[0]
            if is_chinese:
                return f"建議您去「{top['last_seen_on']}」找「{top['label']}」。"
            return f"I suggest '{top['last_seen_on']}' where '{top['label']}' is."
        return "抱歉，我目前沒有找到相關記憶。" if is_chinese else "Sorry, no relevant memory found."

    def confirm(self, choice, nav_target, nav_label, user_id, query):
        try:
            self.conv_logs.find_one_and_update(
                {"user_id": user_id, "query": query},
                {"$set": {"user_choice": choice, "confirmed_at": datetime.datetime.now()}},
                sort=[("timestamp", -1)]
            )
        except Exception as e:
            print(f"[Confirm] skipped: {e}")
        if choice == 1:
            return {"status": "navigate", "nav_target": nav_target,
                    "nav_label": nav_label, "message": f"好的，導航到「{nav_label}」。"}
        elif choice == 2:
            pos_str = f"[{nav_target[0]:.1f}, {nav_target[1]:.1f}]" if nav_target else "未知"
            return {"status": "info_only", "message": f"「{nav_label}」的座標是 {pos_str}。"}
        return {"status": "cancelled", "message": "已取消。"}

    def _build_options(self, nav_target, nav_label, query=""):
        is_chinese = any('\u4e00' <= c <= '\u9fff' for c in query)
        if nav_target and nav_label:
            if is_chinese:
                return [{"id":1,"label":f"導航到「{nav_label}」"},
                        {"id":2,"label":"只告訴我位置"},{"id":3,"label":"取消"}]
            return [{"id":1,"label":f"Navigate to '{nav_label}'"},
                    {"id":2,"label":"Just tell me the location"},{"id":3,"label":"Cancel"}]
        return [{"id":3,"label":"關閉" if is_chinese else "Close"}]

    def _log_conversation(self, query, expanded_query, intent_type,
                           user_id, answer, nav_target, nav_label,
                           room, recommendations, is_personalized):
        self.conv_logs.insert_one({
            "user_id": user_id, "query": query,
            "expanded_query": expanded_query, "intent_type": intent_type,
            "answer": answer, "nav_label": nav_label, "nav_target": nav_target,
            "room": room, "recommendations": recommendations,
            "is_personalized": is_personalized,
            "timestamp": datetime.datetime.now(),
        })