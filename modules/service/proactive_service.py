import datetime
import logging
import requests

from config import Config

logger = logging.getLogger(__name__)

MIN_CONFIDENCE       = 0.25
MIN_INTERVAL_MINUTES = 30
TTL_HOURS            = Config.SNAPSHOT_TTL_HOURS
LLM_TIMEOUT          = Config.LLM_TIMEOUT
LLM_TEMP             = Config.LLM_TEMPERATURE

NO_INTERRUPT_ACTIONS = {
    "Laying",
    "Typing",
    "PhoneUse",
}

GOOD_TRIGGER_ACTIONS = {
    "Standing",
    "Sitting",
    "Watching",
}


class ProactiveService:

    def __init__(self, db, habit_learner, manifold_engine,
                 proposal_manager, ollama_url: str, llm_model: str):
        self.db               = db
        self.habit_learner    = habit_learner
        self.manifold_engine  = manifold_engine
        self.proposal_manager = proposal_manager
        self.ollama_url       = ollama_url
        self.llm_model        = llm_model

    def evaluate(self, user_id: str, current_action: str,
                 prev_action: str, time_slot: str,
                 user_pos: dict = None) -> dict | None:

        if current_action != "Standing":
            return None
        if not prev_action or prev_action in ("Standing", "Walking"):
            return None
        if prev_action in NO_INTERRUPT_ACTIONS:
            return None

        lookahead = self.habit_learner.get_2step_lookahead(
            user_id=user_id,
            current_action=prev_action,
            time_slot=time_slot,
        )

        if not lookahead["actionable"]:
            return None

        need = lookahead["need"]

        available_item = self._find_available_item(need)
        if not available_item:
            return None

        if not self._check_interval(user_id):
            return None

        manifold_ok = self._validate_with_manifold(
            user_id=user_id,
            predicted_action=lookahead["step1"]["action"] if lookahead["step1"] else "",
            virtual_hour=self._current_virtual_hour(),
            user_pos=user_pos,
            prev_action=prev_action,
        )
        if not manifold_ok:
            return None

        proposal = self._generate_proposal(
            user_id=user_id,
            lookahead=lookahead,
            available_item=available_item,
            time_slot=time_slot,
            prev_action=prev_action,
        )

        print(
            f"[ProactiveService] Proposal generated for {user_id}: "
            f"{available_item['label']} | confidence={lookahead['confidence']:.2f}"
        )

        return proposal

    def _find_available_item(self, need: str) -> dict | None:
        if not need:
            return None
        cutoff = datetime.datetime.utcnow() - datetime.timedelta(hours=TTL_HOURS)
        return self.db.dynamic_objects.find_one(
            {"category": need, "last_seen": {"$gte": cutoff}},
            sort=[("interact_count", -1)]
        )

    def _check_interval(self, user_id: str) -> bool:
        last = self.db.service_proposals.find_one(
            {"user_id": user_id},
            sort=[("created_at", -1)],
        )
        if not last:
            return True
        elapsed = (datetime.datetime.utcnow() - last["created_at"]).seconds
        return elapsed >= MIN_INTERVAL_MINUTES * 60

    def _validate_with_manifold(self, user_id: str, predicted_action: str,
                                 virtual_hour, user_pos: dict,
                                 prev_action: str) -> bool:
        try:
            result      = self.manifold_engine.predict_intent(
                user_id=user_id,
                virtual_hour=virtual_hour,
                user_pos=user_pos,
                prev_action=prev_action,
            )
            probs       = result.get("probs", {})
            target_prob = probs.get(predicted_action, 0.0)
            return target_prob >= 0.35
        except Exception as e:
            logger.warning(f"[ProactiveService] ManifoldEngine validation error: {e}")
            return True

    def _generate_proposal(self, user_id: str, lookahead: dict,
                            available_item: dict, time_slot: str,
                            prev_action: str = "") -> dict:
        step1_action = lookahead["step1"]["action"] if lookahead["step1"] else ""
        step2_action = lookahead["step2"]["action"] if lookahead["step2"] else ""
        item_label   = available_item["label"]
        item_loc     = available_item.get("last_seen_on", "nearby")

        message = self._call_llm(
            system=(
                "You are a proactive home robot assistant. "
                "Generate a short, natural proactive offer. "
                "Max 1-2 sentences. Be friendly but not intrusive. "
                "Only refer to what the user just did, not what they will do next."
            ),
            user=(
                f"The user just finished: {prev_action or 'an activity'}.\n"
                f"Time of day: {time_slot}\n"
                f"Item to offer: {item_label} (located at {item_loc})\n"
                f"Generate a proactive offer to bring the user {item_label}:"
            ),
            max_tokens=60,
        ) or f"Would you like me to get you {item_label}?"

        return {
            "user_id":    user_id,
            "message":    message,
            "item":       item_label,
            "item_loc":   item_loc,
            "need":       lookahead["need"],
            "step1":      step1_action,
            "step2":      step2_action,
            "confidence": lookahead["confidence"],
            "time_slot":  time_slot,
            "created_at": datetime.datetime.utcnow(),
        }

    def _current_virtual_hour(self) -> float:
        try:
            doc = self.db.system_config.find_one({"key": "virtual_hour"})
            if doc:
                return float(doc.get("value", datetime.datetime.now().hour))
        except Exception:
            pass
        return float(datetime.datetime.now().hour)

    def _call_llm(self, system: str, user: str,
                  max_tokens: int = 100) -> str | None:
        try:
            resp = requests.post(
                f"{self.ollama_url}/api/chat",
                json={
                    "model":   self.llm_model,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user",   "content": user},
                    ],
                    "stream":  False,
                    "options": {
                        "temperature": LLM_TEMP,
                        "num_predict": max_tokens,
                    },
                },
                timeout=LLM_TIMEOUT,
            )
            resp.raise_for_status()
            return resp.json()["message"]["content"].strip()
        except Exception as e:
            logger.error(f"[ProactiveService] LLM error: {e}")
            return None