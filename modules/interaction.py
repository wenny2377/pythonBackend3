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

# ── ReAct system prompt ───────────────────────────────────────────────────────
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
   Always use this to confirm something is actually available right now.

3. finish(answer: str, nav_target: str, nav_label: str)
   Give the final answer combining habit + current availability.
   - answer: natural sentence in the SAME language as the user
   - nav_target: furniture label where the recommended object is (e.g. "table") or "unknown"
   - nav_label: same as nav_target

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
        三層搜尋，MongoDB 是 ground truth：
        1. FAISS 語意搜尋（快速候選）
        2. MongoDB 最近看到的物件（TTL 過濾，確保還在）
        3. 合併去重，回傳給 LLM
        """
        from datetime import datetime, timedelta
        TTL_HOURS = 2  # 超過 2 小時未看到視為可能不在

        # Layer 1: FAISS 語意搜尋
        faiss_results = self.vector.search_dynamic(query, top_k=5)
        seen_labels   = {r.get("label","") for r in faiss_results}

        # Layer 2: MongoDB — 最近有看到的物件（TTL 過濾）
        cutoff = datetime.utcnow() - timedelta(hours=TTL_HOURS)
        recent_docs = list(self.db.dynamic_objects.find(
            {"last_seen": {"$gte": cutoff}},
            {"label":1,"room":1,"last_seen_on":1,
             "furniture_pos":1,"interact_count":1,"last_seen":1}
        ).sort("interact_count", -1).limit(12))

        # 如果沒有 last_seen 欄位的 doc，撈全部（舊資料相容）
        if not recent_docs:
            recent_docs = list(self.db.dynamic_objects.find(
                {}, {"label":1,"room":1,"last_seen_on":1,
                     "furniture_pos":1,"interact_count":1}
            ).sort("interact_count", -1).limit(12))

        # 合併：FAISS 結果優先，MongoDB 補充
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
                    "similarity":    0.0,  # MongoDB 補充，非 FAISS 命中
                })
                seen_labels.add(label)

        if not combined:
            return "No objects currently visible in the home."

        # 最多傳 8 個給 LLM，格式包含 Camera/Room 讓 LLM 做空間推理
        lines = []
        for r in combined[:8]:
            interact = r.get("interact_count", 0)
            freq     = f", used {interact}x" if interact > 0 else ""
            room     = r.get("room","?")
            lines.append(
                f"- {r.get('label','?')}: "
                f"on {r.get('last_seen_on','?')} "
                f"in {room}{freq}"
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

        # SkillManager（動態載入避免循環 import）
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

    # ── 快速意圖分類（關鍵字，不呼叫 LLM）────────────────────────────────────
    CHAT_KEYWORDS = {
        "hate", "love", "feel", "feeling", "miss", "sad", "happy", "angry",
        "tired", "stressed", "bored", "lonely", "excited", "scared", "worry",
        "thank", "thanks", "hello", "hi", "hey", "bye", "goodbye", "sorry",
        "joke", "chat", "talk", "tell me", "how are you", "what do you think",
        "i think", "i feel", "my boss", "my friend", "my life", "i hate",
        "i love", "i miss", "can we", "let's", "lol", "haha",
    }
    LIST_KEYWORDS = {
        "what can i eat", "what can i drink", "what's available", "what do we have",
        "what is available", "show me", "list", "options", "choices",
        "what food", "what drink", "what snack",
    }

    def _classify_intent(self, query: str) -> str:
        """
        回傳: 'chat' | 'list' | 'service'
        純關鍵字比對，不呼叫 LLM，速度快。
        """
        q = query.lower().strip()
        if any(kw in q for kw in self.LIST_KEYWORDS):
            return "list"
        if any(kw in q for kw in self.CHAT_KEYWORDS):
            return "chat"
        return "service"

    # ── 主入口 ────────────────────────────────────────────────────────────────
    def process(self, query, user_id="Unknown", robot_pos=None,
                user_pos=None, room=""):

        print(f"\n[Interact] user={user_id} | query='{query}' | room={room}")

        intent = self._classify_intent(query)
        print(f"[Classify] intent={intent}")

        # 純聊天 → 不進 ReAct，直接用 LLM 回應
        if intent == "chat":
            return self._chat_response(query, user_id)

        # 列表查詢 → 直接從 MongoDB 撈所有選項
        if intent == "list":
            return self._list_response(query, user_id)

        # ReAct 主流程（service 類）
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

        # Fallback：固定 pipeline
        logger.info("[Interact] Using fixed pipeline fallback")
        return self._pipeline_process(query, user_id, robot_pos, user_pos, room)

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

    # ── 列表回應（所有可用選項）──────────────────────────────────────────────
    def _list_response(self, query: str, user_id: str) -> dict:
        print(f"[List] Fetching available options from MongoDB")
        from datetime import datetime, timedelta

        # 從 MongoDB 取最近看到的物件
        cutoff = datetime.utcnow() - timedelta(hours=2)
        docs = list(self.db.dynamic_objects.find(
            {"last_seen": {"$gte": cutoff}},
            {"label":1,"room":1,"last_seen_on":1,"furniture_pos":1,"interact_count":1}
        ).sort("interact_count", -1))

        if not docs:
            docs = list(self.db.dynamic_objects.find(
                {}, {"label":1,"room":1,"last_seen_on":1,"furniture_pos":1,"interact_count":1}
            ).sort("interact_count", -1).limit(15))

        # 排除人物類
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
                "confidence": 0.5, "intent_type": "list",
                "recommendations": [], "is_personalized": False,
            }

        # 用 LLM 從清單裡選出和 query 相關的物件
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

        # 用過濾後的 label 取對應 doc
        label_to_doc = {d["label"].lower(): d for d in docs}
        filtered_docs = [label_to_doc[l] for l in relevant_labels if l in label_to_doc]

        # 如果 LLM 過濾失敗，fallback 到全部
        if not filtered_docs:
            filtered_docs = docs[:6]

        # 用習慣記憶排序
        habit_results = self.vector.search_habit(query, user_id=user_id, top_k=5)
        habit_items   = set()
        for h in habit_results:
            habit_items.update([i.lower() for i in h.get("interacting_items", [])])

        filtered_docs.sort(key=lambda d: (
            0 if d.get("label","").lower() in habit_items else 1,
            -d.get("interact_count", 0)
        ))

        # 組回答
        items_str = ", ".join(
            f"{d['label']} ({d.get('last_seen_on','?')} in {d.get('room','?')})"
            for d in filtered_docs[:5]
        )
        if is_chinese:
            answer = f"目前可以吃/喝的有：{items_str}。您想要哪一個？"
        else:
            answer = f"Currently available: {items_str}. Which one would you like?"

        # 選項：每個相關物件一個
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
            "confidence": 0.9, "intent_type": "list",
            "recommendations": [
                {"label": d["label"], "last_seen_on": d.get("last_seen_on"),
                 "room": d.get("room"), "score": 0.8}
                for d in filtered_docs[:4]
            ],
            "is_personalized": len(habit_items) > 0,
        }

    # ── ReAct 流程 ────────────────────────────────────────────────────────────
    def _react_process(self, query, user_id, room):
        sm = self.skill_manager

        # 1. 讀取或生成 SKILL.md
        skill_md = sm.get_skill(user_id)
        if not skill_md:
            skill_md = sm.generate(user_id)

        # 2. GapDetector
        has_gap, missing = sm.detect_gap(user_id, query)
        if has_gap and missing:
            print(f"[GapDetector] Missing: {missing} → fill_gap()")
            skill_md = sm.fill_gap(user_id, query, missing)

        # 3. 更新 last_used
        self.db.user_skills.update_one(
            {"user_id": user_id},
            {"$set": {"last_used": datetime.datetime.utcnow()}}
        )

        # 4. system prompt = ReAct + SKILL.md
        system = REACT_SYSTEM_PROMPT
        if skill_md:
            system += f"\n\n## This user's skill profile\n{skill_md}"

        # 5. 取得已知家具
        furniture_labels = [
            d["label"] for d in
            self.db.scene_snapshots.find({}, {"label": 1})
            if "label" in d
        ]
        furniture_str = ", ".join(furniture_labels) or "unknown"

        # 6. ReAct loop
        observations = []
        used_tools   = []   # 記錄已使用的工具，防止重複

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

            # 告訴 LLM 已用過的工具 + 剩餘步數
            remaining = MAX_STEPS - step
            used_str  = f"Already used: {used_tools}" if used_tools else "No tools used yet"
            next_hint = "You MUST call finish now." if remaining == 1 else \
                        f"Remaining steps: {remaining} (including this one)"

            user_prompt = (
                f"User: {query}\n"
                f"User ID: {user_id}\n"
                f"Room: {room}\n"
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

            # tool 是空字串 → LLM 沒有選工具，直接跳出讓 summarize 處理
            if not tool_name:
                logger.warning(f"[ReAct] Step {step+1}: empty tool name, breaking to summarize")
                break

            # finish → 結束
            if tool_name == "finish":
                answer     = tool_input.get("answer", "")
                nav_target = tool_input.get("nav_target", "unknown")
                nav_label  = tool_input.get("nav_label", nav_target)

                # 對話型查詢：沒有用任何搜尋工具 → 不顯示導航
                # 例如 "can you help me", "how are you", "chat with me"
                if not observations:
                    nav_target = None
                    nav_label  = None

                nav_pos = self._resolve_pos(nav_target) if nav_target and nav_target != "unknown" else None

                # RelevanceGate → 更新 SKILL.md
                if sm.should_update(user_id, query, answer, observations):
                    print(f"[RelevanceGate] Updating SKILL.md...")
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

            # 執行工具
            # 如果重複呼叫同一個工具 → 強制用 LLM 總結後 finish
            if tool_name in used_tools:
                print(f"[ReAct] Step {step+1}: '{tool_name}' already used → force finish")

                # 收集所有 observations 的結果
                all_results = "\n".join(
                    f"From {o['tool']}: {o['result']}"
                    for o in observations
                    if o["result"] and "No " not in o["result"][:3]
                )

                # 用 LLM 把結果轉成自然語言
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

        # MAX_STEPS 用完 → 用 LLM 總結已有的 observations
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

        # 完全沒有結果才 fallback 到 pipeline
        return None

    # ── 固定 pipeline fallback（v2 完整保留）─────────────────────────────────
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

        print(f"[Pipeline] answer={answer[:60]}...")
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