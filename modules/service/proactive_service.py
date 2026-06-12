"""
proactive_service.py
────────────────────
Service Layer — Proactive (Robot-Initiated)

Single responsibility:
  Anticipate user needs and generate proactive service proposals.

Design principle — "Predict the user's prediction":
  Instead of waiting for the user to realize they need something,
  the robot observes the current action, performs a 2-step lookahead
  using the learned transition matrix, and prepares service in advance.

  Example:
    User is Eating (Evening)
    → Step 1: Eating → Watching (prob 0.65)
    → Step 2: Watching → SittingDrink (prob 0.58)
    → Inferred need: drink
    → Check dynamic_objects: cola available
    → Trigger: "Would you like a drink while watching TV?"
    → Timing: when user finishes Eating (ReturnToStanding)

Trigger conditions (ALL must be met):
  1. prev_action just ended (transition to Standing)
  2. 2-step lookahead confidence >= MIN_CONFIDENCE
  3. Inferred need exists (food/drink)
  4. Required object available in dynamic_objects (TTL check)
  5. Current action NOT in NO_INTERRUPT list
  6. At least MIN_INTERVAL_MINUTES since last proposal to this user
"""

import datetime
import logging
import requests

from config import Config

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

MIN_CONFIDENCE        = 0.25   # Minimum 2-step lookahead confidence
MIN_INTERVAL_MINUTES  = 30     # Minimum minutes between proposals
TTL_HOURS             = Config.SNAPSHOT_TTL_HOURS
LLM_TIMEOUT           = Config.LLM_TIMEOUT
LLM_TEMP              = Config.LLM_TEMPERATURE

# Actions where robot should NOT interrupt
NO_INTERRUPT_ACTIONS = {
    "Laying",    # sleeping
    "Typing",    # focused work
    "PhoneUse",  # on the phone
}

# Actions that signal a good moment to propose
GOOD_TRIGGER_ACTIONS = {
    "Standing",     # just finished something
    "Sitting",      # relaxing
    "Watching",     # watching TV
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

    # ── Public API ────────────────────────────────────────────────────────────

    def evaluate(self, user_id: str, current_action: str,
                 prev_action: str, time_slot: str,
                 user_pos: dict = None) -> dict | None:
        """
        Evaluate whether to generate a proactive proposal.

        Called from app.py after each HAR episode,
        specifically when current_action == 'Standing'
        and prev_action is a meaningful action.

        Returns proposal dict or None.
        """
        # Condition 1: Only trigger when transitioning to Standing
        # (user just finished an action)
        if current_action != "Standing":
            return None
        if not prev_action or prev_action in ("Standing", "Walking"):
            return None

        # Condition 2: Previous action must not be in NO_INTERRUPT
        if prev_action in NO_INTERRUPT_ACTIONS:
            return None

        # Condition 3: 2-step lookahead
        lookahead = self.habit_learner.get_2step_lookahead(
            user_id=user_id,
            current_action=prev_action,
            time_slot=time_slot,
        )

        if not lookahead["actionable"]:
            return None

        need = lookahead["need"]

        # Condition 4: Required object available
        available_item = self._find_available_item(need)
        if not available_item:
            return None

        # Condition 5: Minimum interval since last proposal
        if not self._check_interval(user_id):
            return None

        # Condition 6: ManifoldEngine secondary validation
        manifold_ok = self._validate_with_manifold(
            user_id=user_id,
            predicted_action=lookahead["step1"]["action"] if lookahead["step1"] else "",
            virtual_hour=self._current_virtual_hour(),
            user_pos=user_pos,
            prev_action=prev_action,
        )
        if not manifold_ok:
            return None

        # Generate proposal
        proposal = self._generate_proposal(
            user_id=user_id,
            lookahead=lookahead,
            available_item=available_item,
            time_slot=time_slot,
        )

        print(
            f"[ProactiveService] Proposal generated for {user_id}: "
            f"{available_item['label']} | confidence={lookahead['confidence']:.2f}"
        )

        return proposal

    # ── Trigger conditions ────────────────────────────────────────────────────

    def _find_available_item(self, need: str) -> dict | None:
        """Find an available item of the required category."""
        if not need:
            return None

        cutoff = datetime.datetime.utcnow() - datetime.timedelta(hours=TTL_HOURS)
        item   = self.db.dynamic_objects.find_one({
            "category":  need,
            "last_seen": {"$gte": cutoff},
        }, sort=[("interact_count", -1)])

        return item

    def _check_interval(self, user_id: str) -> bool:
        """Return True if enough time has passed since last proposal."""
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
        """
        Use ManifoldEngine to cross-validate the prediction.
        Returns True if ManifoldEngine also predicts a service-relevant action.
        """
        try:
            result = self.manifold_engine.predict_intent(
                user_id=user_id,
                virtual_hour=virtual_hour,
                user_pos=user_pos,
                prev_action=prev_action,
            )
            # ManifoldEngine confidence >= 0.4 for the same predicted action
            probs = result.get("probs", {})
            target_prob = probs.get(predicted_action, 0.0)
            return target_prob >= 0.35
        except Exception as e:
            logger.warning(f"[ProactiveService] ManifoldEngine validation error: {e}")
            return True  # Don't block if ManifoldEngine is unavailable

    # ── Proposal generation ───────────────────────────────────────────────────

    def _generate_proposal(self, user_id: str, lookahead: dict,
                             available_item: dict,
                             time_slot: str) -> dict:
        """Generate the proposal message using LLM."""
        step1_action = lookahead["step1"]["action"] if lookahead["step1"] else ""
        step2_action = lookahead["step2"]["action"] if lookahead["step2"] else ""
        item_label   = available_item["label"]
        item_loc     = available_item.get("last_seen_on", "nearby")

        # Generate natural language proposal
        message = self._call_llm(
            system=(
                "You are a proactive home robot assistant. "
                "Generate a short, natural proactive offer. "
                "Max 1-2 sentences. Be friendly but not intrusive."
            ),
            user=(
                f"User just finished an activity.\n"
                f"Predicted next activities: {step1_action} → {step2_action}\n"
                f"Available item: {item_label} (at {item_loc})\n"
                f"Time of day: {time_slot}\n"
                f"Generate a proactive offer:"
            ),
            max_tokens=60,
        ) or f"Would you like me to get you {item_label}?"

        return {
            "user_id":     user_id,
            "message":     message,
            "item":        item_label,
            "item_loc":    item_loc,
            "need":        lookahead["need"],
            "step1":       step1_action,
            "step2":       step2_action,
            "confidence":  lookahead["confidence"],
            "time_slot":   time_slot,
            "created_at":  datetime.datetime.utcnow(),
        }

    # ── Utility helpers ───────────────────────────────────────────────────────

    def _current_virtual_hour(self) -> float:
        """Get current virtual hour from system_config, fallback to real hour."""
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
                    "model": self.llm_model,
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