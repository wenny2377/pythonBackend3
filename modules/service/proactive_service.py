import datetime
import logging
import requests

from config import Config
from modules.memory.skill_manager import preferred_item_from_skill_md

logger = logging.getLogger(__name__)

MIN_CONFIDENCE       = 0.25
MIN_INTERVAL_MINUTES = 30
TTL_HOURS            = Config.SNAPSHOT_TTL_HOURS
LLM_TIMEOUT          = Config.LLM_TIMEOUT
LLM_TEMP             = Config.LLM_TEMPERATURE

NO_INTERRUPT_ACTIONS = {"Laying", "Typing", "UsingPhone"}


class ProactiveService:

    def __init__(self, db, habit_learner, proposal_manager,
                 ollama_url: str, llm_model: str, skill_manager=None):
        self.db               = db
        self.habit_learner    = habit_learner
        self.proposal_manager = proposal_manager
        self.ollama_url       = ollama_url
        self.llm_model        = llm_model
        self.skill_manager    = skill_manager

    def evaluate(self, user_id: str, current_action: str, prev_action: str,
                 time_slot: str, user_pos: dict = None) -> dict | None:
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
        if not lookahead["actionable"] or lookahead["confidence"] < MIN_CONFIDENCE:
            return None

        need           = lookahead["need"]
        available_item = self._find_available_item(user_id, need)
        if not available_item:
            return None

        if not self._check_interval(user_id):
            return None

        proposal = self._generate_proposal(
            user_id=user_id,
            lookahead=lookahead,
            available_item=available_item,
            time_slot=time_slot,
            prev_action=prev_action,
        )
        print(f"[ProactiveService] {user_id}: {available_item['label']} "
              f"| conf={lookahead['confidence']:.2f}")
        return proposal

    def _find_available_item(self, user_id: str, need: str) -> dict | None:
        if not need:
            return None
        cutoff = datetime.datetime.utcnow() - datetime.timedelta(hours=TTL_HOURS)

        candidates = list(self.db.dynamic_objects.find(
            {"category": need, "last_seen": {"$gte": cutoff}},
            sort=[("interact_count", -1)],
        ))
        if not candidates:
            candidates = list(self.db.dynamic_objects.find(
                {"category": need}, sort=[("interact_count", -1)],
            ))
        if not candidates:
            return None

        preferred_label = self._preferred_label(user_id, candidates, need)
        if preferred_label:
            for c in candidates:
                if c["label"].lower() == preferred_label.lower():
                    return c

        return candidates[0]

    def _preferred_label(self, user_id: str, candidates: list, need: str) -> str:
        if self.skill_manager is None:
            return ""
        try:
            skill_md = self.skill_manager.get_skill(user_id)
        except Exception:
            return ""
        available_labels = {c["label"].lower() for c in candidates}
        return preferred_item_from_skill_md(skill_md, available_labels, need)

    def _check_interval(self, user_id: str) -> bool:
        last_time = self.proposal_manager.get_last_proposal_time(user_id)
        if not last_time:
            return True
        elapsed = (datetime.datetime.utcnow() - last_time).total_seconds()
        return elapsed >= MIN_INTERVAL_MINUTES * 60

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

    def _call_llm(self, system: str, user: str, max_tokens: int = 100) -> str | None:
        try:
            resp = requests.post(
                f"{self.ollama_url}/api/chat",
                json={
                    "model":    self.llm_model,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user",   "content": user},
                    ],
                    "stream":  False,
                    "options": {"temperature": LLM_TEMP, "num_predict": max_tokens},
                },
                timeout=LLM_TIMEOUT,
            )
            resp.raise_for_status()
            return resp.json()["message"]["content"].strip()
        except Exception as e:
            logger.error(f"[ProactiveService] LLM error: {e}")
            return None