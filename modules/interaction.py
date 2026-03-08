import datetime
import requests
import json


class InteractionEngine:
    def __init__(self, mongo_client, vector_memory, ollama_url, model_name):
        self.db           = mongo_client["robot_rag_db"]
        self.vector       = vector_memory
        self.ollama_url   = ollama_url
        self.model_name   = model_name
        self.conv_logs    = self.db["conversation_logs"]

    # ─────────────────────────────────────────────
    # 主入口
    # ─────────────────────────────────────────────
    def process(self, query, user_id="Unknown", robot_pos=None,
                user_pos=None, room=""):

        # Step 1: 搜尋行為記憶
        habit_results = self.vector.search_habit(
            query, user_id=user_id, top_k=3
        )

        # Step 2: 搜尋動態物件
        dynamic_results = self.vector.search_dynamic(query, top_k=3)

        # Step 3: 補 MongoDB 最新座標
        nav_target    = None
        nav_label     = None
        best_dynamic  = None

        if dynamic_results:
            best_dynamic = dynamic_results[0]
            nav_label    = best_dynamic.get("last_seen_on")
            nav_target   = best_dynamic.get("furniture_pos")

            if nav_label and not nav_target:
                doc = self.db.scene_snapshots.find_one({"label": nav_label})
                if doc:
                    nav_target = doc.get("pos")

        elif habit_results:
            best_habit = habit_results[0]
            nav_label  = best_habit.get("instance")
            nav_target = best_habit.get("furniture_pos")

            if nav_label and not nav_target:
                doc = self.db.scene_snapshots.find_one({"label": nav_label})
                if doc:
                    nav_target = doc.get("pos")

        # Step 4: Gemma3 生成中文回答
        answer = self._generate_answer(
            query=query,
            user_id=user_id,
            room=room,
            habit_results=habit_results,
            dynamic_results=dynamic_results,
            nav_label=nav_label,
        )

        # Step 5: 組選項
        options = self._build_options(nav_target, nav_label, query=query)

        # Step 6: 記錄對話
        self._log_conversation(
            query=query,
            user_id=user_id,
            answer=answer,
            nav_target=nav_target,
            nav_label=nav_label,
            room=room,
        )

        confidence = dynamic_results[0]["similarity"] if dynamic_results else (
            habit_results[0]["similarity"] if habit_results else 0.0
        )

        print(f"[Interact] answer={answer[:60]}...")
        print(f"[Interact] nav={nav_label} @ {nav_target} | conf={confidence:.2f}")

        return {
            "status":     "Success",
            "answer":     answer,
            "nav_target": nav_target,
            "nav_label":  nav_label,
            "options":    options,
            "confidence": round(confidence, 3),
        }

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
            return {
                "status":  "cancelled",
                "message": "已取消。"
            }

    # ─────────────────────────────────────────────
    # Gemma3 生成回答
    # ─────────────────────────────────────────────
    def _generate_answer(self, query, user_id, room,
                         habit_results, dynamic_results, nav_label):

        # 組 context 給 LLM
        context_parts = []

        if dynamic_results:
            for r in dynamic_results[:2]:
                context_parts.append(
                    f"物品「{r['label']}」{r['spatial_rel']} {r['last_seen_on']}，"
                    f"在 {r['room']}，出現 {r['seen_count']} 次，"
                    f"互動 {r['interact_count']} 次。"
                )

        if habit_results:
            for r in habit_results[:2]:
                items = "、".join(r.get("interacting_items", [])) or "無特定物品"
                context_parts.append(
                    f"{r['user']} 通常在 {r['instance']} {r['action']}，"
                    f"使用 {items}。"
                )

        context = "\n".join(context_parts) if context_parts else "目前沒有相關記憶。"

        prompt = f"""You are a home service robot assistant. Answer the user's question based on the observation memory below.
Reply in the same language as the user's question (Chinese if asked in Chinese, English if asked in English).
Keep the answer to 2 sentences maximum. Reply directly without preamble like "based on memory" or "observations show".

Observation memory:
{context}

User question: {query}"""

        try:
            resp = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model":       self.model_name,
                    "prompt":      prompt,
                    "stream":      False,
                    "options":     {"temperature": 0.3, "num_predict": 128}
                },
                timeout=30
            )
            if resp.status_code == 200:
                answer = resp.json().get("response", "").strip()
                if answer:
                    return answer
        except Exception as e:
            print(f"[LLM Error] {e}")

        # fallback template（簡單語言偵測）
        is_chinese = any('\u4e00' <= c <= '\u9fff' for c in query)
        if nav_label:
            return f"你要找的東西應該在「{nav_label}」附近。" if is_chinese \
                   else f"What you're looking for should be near '{nav_label}'."
        return "抱歉，我目前沒有找到相關記憶。" if is_chinese \
               else "Sorry, I couldn't find any relevant memory."

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
    def _log_conversation(self, query, user_id, answer,
                           nav_target, nav_label, room):
        self.conv_logs.insert_one({
            "user_id":    user_id,
            "query":      query,
            "answer":     answer,
            "nav_label":  nav_label,
            "nav_target": nav_target,
            "room":       room,
            "timestamp":  datetime.datetime.now(),
        })