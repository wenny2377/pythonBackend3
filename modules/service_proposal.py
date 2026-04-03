import datetime
import threading
import requests
import json
import re
from collections import deque


ANTI_SPAM_MINUTES = 10
MIN_CONFIDENCE    = 0.60
MIN_SUCCESS_RATE  = 0.20
MIN_SAMPLES_GATE  = 5


class ServiceProposalEngine:

    def __init__(self, db, ollama_url: str, llm_model: str):
        self.db          = db
        self.ollama_url  = ollama_url
        self.llm_model   = llm_model
        self._queue      = deque()
        self._lock       = threading.Lock()
        self._last_proposed: dict = {}

        print("✅ [ServiceProposalEngine] 初始化完成")

    def evaluate(self, user_id: str, intent_prediction: dict,
                 manifold_point_id: str, user_pos: dict,
                 dynamic_results: list = None) -> dict:

        if not intent_prediction.get("trigger"):
            return {"has_proposal": False, "proposal_id": None}

        intent     = intent_prediction.get("intent", "unknown")
        confidence = intent_prediction.get("confidence", 0.0)

        if confidence < MIN_CONFIDENCE:
            return {"has_proposal": False, "proposal_id": None}

        if self._recently_proposed(user_id, intent):
            print(f"   🚫 [Proposal] anti-spam: {user_id} {intent}")
            return {"has_proposal": False, "proposal_id": None}

        stats = self.db.intent_stats.find_one(
            {"user_id": user_id, "intent": intent}
        )
        if stats:
            total    = stats.get("accepted", 0) + stats.get("rejected", 0) + stats.get("ignored", 0)
            accepted = stats.get("accepted", 0)
            if total >= MIN_SAMPLES_GATE:
                rate = accepted / total
                if rate < MIN_SUCCESS_RATE:
                    print(f"   🚫 [Proposal] low success rate {rate:.2f} for {intent}")
                    return {"has_proposal": False, "proposal_id": None}

        nav_target, nav_label = self._resolve_service_target(intent, dynamic_results or [])
        message = self._generate_question(user_id, intent, confidence, nav_label)

        doc = {
            "user_id":           user_id,
            "intent":            intent,
            "confidence":        round(confidence, 3),
            "manifold_point_id": manifold_point_id,
            "message":           message,
            "nav_target":        nav_target,
            "nav_label":         nav_label,
            "user_pos":          user_pos,
            "status":            "pending",
            "created_at":        datetime.datetime.utcnow(),
        }
        result    = self.db.service_proposals.insert_one(doc)
        proposal_id = str(result.inserted_id)

        with self._lock:
            self._queue.append({
                "proposal_id":       proposal_id,
                "user_id":           user_id,
                "intent":            intent,
                "confidence":        round(confidence, 3),
                "manifold_point_id": manifold_point_id,
                "message":           message,
                "nav_target":        nav_target,
                "nav_label":         nav_label,
            })

        self._last_proposed[(user_id, intent)] = datetime.datetime.utcnow()

        print(f"   💡 [Proposal] queued: {user_id} → {intent} (conf={confidence:.2f})")
        return {"has_proposal": True, "proposal_id": proposal_id}

    def get_next_proposal(self) -> dict:
        with self._lock:
            if self._queue:
                return self._queue.popleft()
        return {}

    def handle_response(self, proposal_id: str, user_id: str,
                        result: str, manifold_engine) -> dict:
        try:
            from bson import ObjectId
            self.db.service_proposals.update_one(
                {"_id": ObjectId(proposal_id)},
                {"$set": {"status": result, "responded_at": datetime.datetime.utcnow()}}
            )
            proposal = self.db.service_proposals.find_one({"_id": ObjectId(proposal_id)})
            if proposal and proposal.get("manifold_point_id"):
                manifold_engine.update_service_result(
                    proposal["manifold_point_id"], result
                )
            print(f"   📝 [Proposal] {proposal_id} → {result}")
            return {"status": "ok", "result": result}
        except Exception as e:
            print(f"[Proposal] handle_response error: {e}")
            return {"status": "error", "message": str(e)}

    def _recently_proposed(self, user_id: str, intent: str) -> bool:
        key      = (user_id, intent)
        last     = self._last_proposed.get(key)
        if last is None:
            return False
        elapsed = (datetime.datetime.utcnow() - last).total_seconds() / 60
        return elapsed < ANTI_SPAM_MINUTES

    def _resolve_service_target(self, intent: str,
                                 dynamic_results: list) -> tuple:
        intent_item_map = {
            "drinking": ["water", "cup", "bottle", "drink", "beverage", "glass"],
            "eating":   ["food", "apple", "banana", "snack", "fruit", "plate"],
            "typing":   ["laptop", "computer", "keyboard", "desk"],
            "sleeping": ["bed", "pillow", "blanket", "bedroom"],
            "sitting":  ["sofa", "chair", "couch", "seat"],
        }
        keywords = intent_item_map.get(intent.lower(), [intent])

        for kw in keywords:
            for d in dynamic_results:
                if kw.lower() in d.get("label", "").lower():
                    return d.get("furniture_pos"), d.get("last_seen_on", kw)

        return None, intent

    def _generate_question(self, user_id: str, intent: str,
                            confidence: float, nav_label: str) -> str:
        fallback_map = {
            "drinking": f"你想喝點什麼嗎？",
            "eating":   f"要我幫你拿點吃的嗎？",
            "typing":   f"需要我幫你準備工作環境嗎？",
            "sleeping": f"要休息了嗎？需要我幫你準備嗎？",
            "sitting":  f"要坐下休息一下嗎？",
        }
        fallback = fallback_map.get(intent.lower(), f"需要我幫你什麼忙嗎？")

        prompt = f"""You are a home service robot assistant.
The user ({user_id}) appears to need: {intent}
Confidence: {confidence:.0%}
Relevant location: {nav_label or 'nearby'}

Generate a SHORT, natural, friendly question in Traditional Chinese (繁體中文).
Rules:
- 1 sentence only
- Sound natural, not robotic
- Include the relevant item or location if available
- End with 「嗎？」or 「呢？」

Reply with ONLY the question, no explanation."""

        try:
            resp = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model":   self.llm_model,
                    "prompt":  prompt,
                    "stream":  False,
                    "options": {"temperature": 0.4, "num_predict": 60},
                },
                timeout=20,
            )
            if resp.status_code == 200:
                msg = resp.json().get("response", "").strip()
                msg = msg.split("\n")[0].strip()
                if msg and len(msg) > 3:
                    return msg
        except Exception as e:
            print(f"[Proposal] LLM question error: {e}")

        return fallback