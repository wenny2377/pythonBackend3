"""
InteractionEngine v2
完全移除 rule base INTENT_MAP
改用 LLM 做意圖理解 + 個人化推薦

流程：
  1. LLM 分析意圖 → 輸出搜尋關鍵字 + 意圖類型（JSON）
  2. FAISS Layer 1 查習慣行為記憶
  3. FAISS Layer 2 查相關動態物件
  4. 交叉比對：習慣物件 ∩ 現有物件 → 個人化推薦
  5. LLM 生成個人化回答（含推薦理由）
"""

import datetime
import requests
import json
import re


class InteractionEngine:

    def __init__(self, mongo_client, vector_memory, ollama_url, model_name):
        self.db         = mongo_client["robot_rag_db"]
        self.vector     = vector_memory
        self.ollama_url = ollama_url
        self.model_name = model_name
        self.conv_logs  = self.db["conversation_logs"]

    # ─────────────────────────────────────────────
    # 主入口
    # ─────────────────────────────────────────────
    def process(self, query, user_id="Unknown", robot_pos=None,
                user_pos=None, room=""):

        # Step 1: LLM 分析意圖（完全取代 rule base）
        intent_result = self._llm_analyze_intent(query, user_id)
        intent_type   = intent_result.get("intent_type", "direct")
        search_kw     = intent_result.get("search_keywords", [query])
        expanded_query= " ".join(search_kw)

        print(f"[Intent] '{query}' → type={intent_type} | kw={search_kw}")

        # Step 2: FAISS 查習慣行為記憶（個人化）
        habit_results = self.vector.search_habit(
            expanded_query, user_id=user_id, top_k=5
        )

        # Step 3: FAISS 查相關動態物件
        # 先查此用戶互動過的（個人化優先）
        dynamic_personal = self.vector.search_dynamic(
            expanded_query, top_k=5, user_filter=user_id
        )
        # 不足時補全體結果
        if len(dynamic_personal) < 2:
            dynamic_all  = self.vector.search_dynamic(expanded_query, top_k=5)
            seen_labels  = {r["label"] for r in dynamic_personal}
            for r in dynamic_all:
                if r["label"] not in seen_labels:
                    dynamic_personal.append(r)
                    seen_labels.add(r["label"])
        dynamic_results = dynamic_personal[:5]

        # Step 4: 交叉比對（習慣物件 ∩ 現有物件）
        recommendations = self._cross_match(
            habit_results   = habit_results,
            dynamic_results = dynamic_results,
            user_id         = user_id,
        )

        # Step 5: 決定導航目標
        nav_target, nav_label = self._resolve_nav(
            recommendations = recommendations,
            dynamic_results = dynamic_results,
            habit_results   = habit_results,
        )

        # Step 6: LLM 生成個人化回答
        answer = self._generate_answer(
            query           = query,
            intent_type     = intent_type,
            user_id         = user_id,
            room            = room,
            habit_results   = habit_results,
            dynamic_results = dynamic_results,
            recommendations = recommendations,
            nav_label       = nav_label,
        )

        # Step 7: 組選項
        options = self._build_options(nav_target, nav_label, query=query)

        # Step 8: 記錄對話
        self._log_conversation(
            query           = query,
            expanded_query  = expanded_query,
            intent_type     = intent_type,
            user_id         = user_id,
            answer          = answer,
            nav_target      = nav_target,
            nav_label       = nav_label,
            room            = room,
            recommendations = recommendations,
            is_personalized = len(dynamic_personal) > 0,
        )

        confidence = recommendations[0]["score"] if recommendations else (
            dynamic_results[0]["similarity"] if dynamic_results else (
                habit_results[0]["similarity"] if habit_results else 0.0
            )
        )

        print(f"[Interact] answer={answer[:60]}...")
        print(f"[Interact] nav={nav_label} @ {nav_target} | conf={confidence:.2f}")

        return {
            "status":          "Success",
            "answer":          answer,
            "nav_target":      nav_target,
            "nav_label":       nav_label,
            "options":         options,
            "confidence":      round(confidence, 3),
            "intent_type":     intent_type,
            "recommendations": recommendations,
            "is_personalized": len(dynamic_personal) > 0,
        }

    # ─────────────────────────────────────────────
    # Step 1: LLM 意圖分析（取代 rule base）
    # ─────────────────────────────────────────────
    def _llm_analyze_intent(self, query: str, user_id: str) -> dict:
        """
        讓 LLM 分析使用者意圖，輸出：
        {
          "intent_type": "fuzzy_need" / "object_search" / "habit_query" / "direct",
          "search_keywords": ["keyword1", "keyword2", ...],
          "need_category": "food" / "drink" / "rest" / "entertainment" / "object" / "other"
        }
        """
        prompt = f"""Analyze the user's intent from their message.

User ID: {user_id}
User said: "{query}"

Reply ONLY with valid JSON, no markdown, no extra text:
{{
  "intent_type": "fuzzy_need|object_search|habit_query|direct",
  "search_keywords": ["english", "keywords", "for", "faiss", "search"],
  "need_category": "food|drink|rest|entertainment|medicine|object|person|other"
}}

intent_type definitions:
- fuzzy_need: user expresses a feeling or need (hungry, tired, thirsty, bored, cold)
- object_search: user wants to find a specific object (where is my glasses)
- habit_query: user asks about someone's habits or routine
- direct: direct factual question

search_keywords: 5-8 English keywords relevant to FAISS memory search.
For fuzzy_need, include related actions and objects (e.g. hungry → eating, food, kitchen, apple, banana).
"""
        try:
            resp = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model":   self.model_name,
                    "prompt":  prompt,
                    "stream":  False,
                    "options": {"temperature": 0.1, "num_predict": 128}
                },
                timeout=30
            )
            if resp.status_code == 200:
                raw   = resp.json().get("response", "").strip()
                clean = re.sub(r'```(?:json)?\s*', '', raw).strip()
                match = re.search(r'\{.*\}', clean, re.DOTALL)
                if match:
                    result = json.loads(match.group(0))
                    print(f"   🧠 [Intent] {result}")
                    return result
        except Exception as e:
            print(f"[Intent LLM Error] {e}")

        # Fallback：把原始 query 當關鍵字
        return {
            "intent_type":    "direct",
            "search_keywords": [query],
            "need_category":  "other"
        }

    # ─────────────────────────────────────────────
    # Step 4: 交叉比對（核心推薦邏輯）
    # 習慣物件（高 weight）∩ 現有物件（dynamic_objects）
    # ─────────────────────────────────────────────
    def _cross_match(self, habit_results, dynamic_results, user_id) -> list:
        """
        推薦邏輯（類推薦系統）：
        1. 從習慣記憶取出 interacting_items（這個人常用的物件）
        2. 對照 dynamic_results（現在家裡有的物件）
        3. 有交集的物件 → 高分推薦
        4. 只有動態物件有、習慣沒有 → 低分推薦
        """
        recommendations = []

        # 建立習慣物件集合（含 weight 資訊）
        habit_items = {}  # label → {count, instance, furniture_pos}
        for h in habit_results:
            instance    = h.get("instance", "")
            fpos        = h.get("furniture_pos")
            for item in h.get("interacting_items", []):
                item = item.lower()
                if item not in habit_items:
                    habit_items[item] = {
                        "count":         0,
                        "instance":      instance,
                        "furniture_pos": fpos,
                        "similarity":    h.get("similarity", 0.0),
                    }
                habit_items[item]["count"] += 1

        # 對照動態物件
        seen = set()
        for d in dynamic_results:
            label = d.get("label", "").lower()
            if label in seen:
                continue
            seen.add(label)

            is_habitual      = label in habit_items
            habit_count      = habit_items[label]["count"] if is_habitual else 0
            interact_count   = d.get("interact_count", 0)
            is_mine          = user_id in d.get("interacted_by", [])

            # 計算推薦分數
            # 個人化互動次數 × 0.4 + 習慣出現次數 × 0.4 + FAISS 相似度 × 0.2
            score = (
                min(interact_count / 10.0, 1.0) * 0.4 +
                min(habit_count    / 5.0,  1.0) * 0.4 +
                d.get("similarity", 0.0)         * 0.2
            )

            recommendations.append({
                "label":         label,
                "room":          d.get("room", ""),
                "last_seen_on":  d.get("last_seen_on", ""),
                "furniture_pos": d.get("furniture_pos"),
                "interact_count":interact_count,
                "habit_count":   habit_count,
                "is_habitual":   is_habitual,
                "is_mine":       is_mine,
                "score":         round(score, 3),
            })

        # 依分數排序
        recommendations.sort(key=lambda x: x["score"], reverse=True)

        # 補充：習慣裡有但 dynamic 沒出現的（可能被移走了）
        for label, info in habit_items.items():
            if label not in seen:
                recommendations.append({
                    "label":         label,
                    "room":          "",
                    "last_seen_on":  info["instance"],
                    "furniture_pos": info["furniture_pos"],
                    "interact_count":0,
                    "habit_count":   info["count"],
                    "is_habitual":   True,
                    "is_mine":       True,
                    "score":         round(info["count"] / 10.0 * 0.4, 3),
                    "note":          "not_currently_visible",
                })

        print(f"   🎯 [CrossMatch] {len(recommendations)} recommendations")
        for r in recommendations[:3]:
            print(f"      '{r['label']}' score={r['score']} "
                  f"habit={r['habit_count']} interact={r['interact_count']}")

        return recommendations[:5]

    # ─────────────────────────────────────────────
    # Step 5: 決定導航目標
    # ─────────────────────────────────────────────
    def _resolve_nav(self, recommendations, dynamic_results, habit_results):
        """優先推薦分數最高且有座標的目標"""
        for r in recommendations:
            pos = r.get("furniture_pos")
            if pos:
                # 從 scene_snapshots 補最新座標
                label = r.get("last_seen_on") or r.get("label")
                doc   = self.db.scene_snapshots.find_one({"label": label})
                if doc and doc.get("pos"):
                    return doc["pos"], label
                return pos, label

        # fallback：dynamic_results
        for d in dynamic_results:
            pos   = d.get("furniture_pos")
            label = d.get("last_seen_on")
            if pos and label:
                return pos, label
            if label:
                doc = self.db.scene_snapshots.find_one({"label": label})
                if doc and doc.get("pos"):
                    return doc["pos"], label

        # fallback：habit_results
        for h in habit_results:
            label = h.get("instance")
            pos   = h.get("furniture_pos")
            if pos and label:
                return pos, label
            if label:
                doc = self.db.scene_snapshots.find_one({"label": label})
                if doc and doc.get("pos"):
                    return doc["pos"], label

        return None, None

    # ─────────────────────────────────────────────
    # Step 6: LLM 個人化回答生成
    # ─────────────────────────────────────────────
    def _generate_answer(self, query, intent_type, user_id, room,
                         habit_results, dynamic_results,
                         recommendations, nav_label):

        is_chinese = any('\u4e00' <= c <= '\u9fff' for c in query)

        # 組合個人化 context
        context_parts = []

        # 推薦物件（最多3個）
        if recommendations:
            context_parts.append("=== Recommended items ===")
            for r in recommendations[:3]:
                status = "✓ habitual" if r["is_habitual"] else ""
                mine   = "✓ personally used" if r["is_mine"] else ""
                context_parts.append(
                    f"- {r['label']}: located at {r['last_seen_on']} "
                    f"({r['room']}), "
                    f"interacted {r['interact_count']} times. {status} {mine}"
                )

        # 習慣記憶（最多2個）
        if habit_results:
            context_parts.append("=== Personal habits ===")
            for h in habit_results[:2]:
                items = ", ".join(h.get("interacting_items", [])) or "nothing specific"
                context_parts.append(
                    f"- {user_id} usually {h['action']} near {h['instance']}, "
                    f"using: {items} (similarity={h['similarity']:.2f})"
                )

        context = "\n".join(context_parts) if context_parts else "No relevant memory found."

        intent_hint = {
            "fuzzy_need":    "Suggest ONLY items/places found in the personal memory below. Do NOT invent or assume anything not in the memory.",
            "object_search": "Tell the user where the object was last seen based on memory only.",
            "habit_query":   "Summarize the user's habits based on memory only.",
            "direct":        "Answer directly and concisely.",
        }.get(intent_type, "")

        lang = "Traditional Chinese (繁體中文)" if is_chinese else "English"

        # 空間現有物件（dynamic_objects，不限用戶）
        space_items_str = ""
        if not recommendations and dynamic_results:
            space_items = [
                f"{r['label']} (in {r['room']}, on {r['last_seen_on']})"
                for r in dynamic_results[:3]
            ]
            space_items_str = "Items currently in the home: " + ", ".join(space_items)

        no_memory_hint = ""
        if not recommendations and not habit_results:
            if space_items_str:
                no_memory_hint = (
                    "No personal habit memory yet for this user. "
                    "Instead, tell them what relevant items are currently available in the space."
                )
            else:
                no_memory_hint = (
                    "No personal memory and no relevant items found. "
                    "Tell the user honestly you need more time to observe."
                )

        prompt = f"""You are a home service robot for {user_id}.
{intent_hint}
{no_memory_hint}

Rules:
- Reply in {lang}
- 1 sentence MAX
- If no habit memory, mention what is currently available in the space
- Only use information from the memory or space items below
- No lists, no preamble, no invented items

Personal memory for {user_id}:
{context}
{space_items_str}

User said: "{query}"
"""
        try:
            resp = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model":   self.model_name,
                    "prompt":  prompt,
                    "stream":  False,
                    "options": {"temperature": 0.3, "num_predict": 80}
                },
                timeout=30
            )
            if resp.status_code == 200:
                answer = resp.json().get("response", "").strip()
                if answer:
                    return answer
        except Exception as e:
            print(f"[LLM Error] {e}")

        # Fallback
        if recommendations:
            top = recommendations[0]
            if is_chinese:
                reason = f"您之前用過 {top['interact_count']} 次" if top["is_mine"] else "家裡現在有"
                return f"建議您去「{top['last_seen_on']}」，{reason}「{top['label']}」。"
            else:
                return f"I suggest going to '{top['last_seen_on']}' where '{top['label']}' is located."
        if nav_label:
            return f"您要找的應該在「{nav_label}」附近。" if is_chinese \
                   else f"What you need should be near '{nav_label}'."
        return "抱歉，我目前沒有找到相關記憶。" if is_chinese \
               else "Sorry, I couldn't find any relevant memory."

    # ─────────────────────────────────────────────
    # 確認導航
    # ─────────────────────────────────────────────
    def confirm(self, choice, nav_target, nav_label, user_id, query):
        self.conv_logs.update_one(
            {"user_id": user_id, "query": query},
            {"$set": {
                "user_choice":  choice,
                "confirmed_at": datetime.datetime.now()
            }},
            sort=[("timestamp", -1)]
        )
        if choice == 1:
            return {
                "status":     "navigate",
                "nav_target": nav_target,
                "nav_label":  nav_label,
                "message":    f"好的，導航到「{nav_label}」。"
            }
        elif choice == 2:
            pos_str = f"[{nav_target[0]:.1f}, {nav_target[1]:.1f}]" if nav_target else "未知"
            return {
                "status":  "info_only",
                "message": f"「{nav_label}」的座標是 {pos_str}。"
            }
        else:
            return {"status": "cancelled", "message": "已取消。"}

    # ─────────────────────────────────────────────
    # 組選項
    # ─────────────────────────────────────────────
    def _build_options(self, nav_target, nav_label, query=""):
        is_chinese = any('\u4e00' <= c <= '\u9fff' for c in query)
        if nav_target and nav_label:
            if is_chinese:
                return [
                    {"id": 1, "label": f"導航到「{nav_label}」"},
                    {"id": 2, "label": "只告訴我位置"},
                    {"id": 3, "label": "取消"},
                ]
            else:
                return [
                    {"id": 1, "label": f"Navigate to '{nav_label}'"},
                    {"id": 2, "label": "Just tell me the location"},
                    {"id": 3, "label": "Cancel"},
                ]
        return [{"id": 3, "label": "關閉" if is_chinese else "Close"}]

    # ─────────────────────────────────────────────
    # 記錄對話
    # ─────────────────────────────────────────────
    def _log_conversation(self, query, expanded_query, intent_type,
                           user_id, answer, nav_target, nav_label,
                           room, recommendations, is_personalized):
        self.conv_logs.insert_one({
            "user_id":         user_id,
            "query":           query,
            "expanded_query":  expanded_query,
            "intent_type":     intent_type,
            "answer":          answer,
            "nav_label":       nav_label,
            "nav_target":      nav_target,
            "room":            room,
            "recommendations": recommendations,
            "is_personalized": is_personalized,
            "timestamp":       datetime.datetime.now(),
        })
