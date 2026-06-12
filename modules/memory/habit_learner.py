"""
habit_learner.py
────────────────
Memory Layer — Layer 2

Single responsibility:
  Learn user habits from observations.
  - Update transition_counts (incremental, recency-weighted)
  - Trigger SKILL.md update via SkillManager
  - Provide 2-step lookahead for ProactiveService

Collections written:
  - transition_counts    (per-user, per-time-slot transition matrix)

Collections read:
  - observation_logs     (to build SKILL.md)
  - transition_counts    (for lookahead queries)
"""

import datetime
import math

# Minimum observation count before a transition is considered reliable
MIN_TRANSITION_COUNT  = 3

# Minimum combined probability for 2-step lookahead to be actionable
MIN_LOOKAHEAD_CONFIDENCE = 0.25

# Recency decay: observations from N days ago have weight = exp(-decay * N)
RECENCY_DECAY = 0.05

# Service needs inferred from action labels
ACTION_SERVICE_NEEDS = {
    "SittingDrink": "drink",
    "Drinking":     "drink",
    "Eating":       "food",
    "Watching":     "drink",   # TV watching often involves a drink
}

# Actions that should NOT be service targets
NO_SERVICE_ACTIONS = {
    "Laying", "Typing", "PhoneUse", "Walking",
    "Standing", "StandUp", "PickingUp", "PuttingDown",
    "Opening", "Cleaning", "Cooking",
}


class HabitLearner:

    def __init__(self, db, skill_manager):
        self.db             = db
        self.skill_manager  = skill_manager
        self.col_transitions = db.transition_counts
        self.col_obs         = db.observation_logs

    # ── Public API ────────────────────────────────────────────────────────────

    def on_new_observation(self, user_id: str, action: str,
                            prev_action: str, time_slot: str,
                            zone_name: str):
        """
        Called after each observation is stored.
        Updates transition_counts and checks if SKILL.md needs updating.
        """
        if not prev_action or not action:
            return
        if action == prev_action:
            return

        self._update_transition(
            user_id=user_id,
            from_action=prev_action,
            to_action=action,
            time_slot=time_slot,
        )

        self._maybe_update_skill(user_id)

    def get_top_transitions(self, user_id: str, from_action: str,
                             time_slot: str = None,
                             top_k: int = 3) -> list:
        """
        Return top-K most likely next actions from from_action.

        Returns list of dicts:
          [{"action": "Watching", "prob": 0.65, "count": 8}, ...]
        """
        query = {"user_id": user_id, "from_action": from_action}
        if time_slot:
            query["time_slot"] = time_slot

        docs = list(self.col_transitions.find(
            query, {"to_action": 1, "weight": 1, "count": 1}
        ))

        if not docs:
            return []

        total_weight = sum(d.get("weight", 0) for d in docs)
        if total_weight == 0:
            return []

        results = []
        for d in docs:
            if d.get("count", 0) < MIN_TRANSITION_COUNT:
                continue
            prob = d.get("weight", 0) / total_weight
            results.append({
                "action": d["to_action"],
                "prob":   round(prob, 3),
                "count":  d.get("count", 0),
            })

        results.sort(key=lambda x: x["prob"], reverse=True)
        return results[:top_k]

    def get_2step_lookahead(self, user_id: str, current_action: str,
                             time_slot: str = None) -> dict:
        """
        2-step lookahead: predict what comes 2 steps after current_action.

        Returns:
          {
            "step1":      {"action": "Watching",     "prob": 0.65},
            "step2":      {"action": "SittingDrink", "prob": 0.58},
            "need":       "drink",
            "confidence": 0.377,   # step1.prob * step2.prob
            "actionable": True,    # confidence >= MIN_LOOKAHEAD_CONFIDENCE
          }
        """
        empty = {
            "step1": None, "step2": None,
            "need": None, "confidence": 0.0, "actionable": False,
        }

        step1_list = self.get_top_transitions(
            user_id, current_action, time_slot, top_k=1)
        if not step1_list:
            return empty

        step1 = step1_list[0]
        if step1["action"] in NO_SERVICE_ACTIONS:
            return empty

        step2_list = self.get_top_transitions(
            user_id, step1["action"], time_slot, top_k=1)

        step2       = step2_list[0] if step2_list else None
        step2_prob  = step2["prob"] if step2 else 0.0
        step2_action = step2["action"] if step2 else None

        # Infer service need from step2 first, then step1
        need = (ACTION_SERVICE_NEEDS.get(step2_action) or
                ACTION_SERVICE_NEEDS.get(step1["action"]))

        confidence = step1["prob"] * step2_prob if step2 else step1["prob"] * 0.5

        return {
            "step1":      step1,
            "step2":      step2,
            "need":       need,
            "confidence": round(confidence, 3),
            "actionable": confidence >= MIN_LOOKAHEAD_CONFIDENCE and need is not None,
        }

    # ── Private methods ───────────────────────────────────────────────────────

    def _update_transition(self, user_id: str, from_action: str,
                            to_action: str, time_slot: str):
        """
        Incrementally update transition_counts with recency weighting.
        Recent observations get higher weight than old ones.
        """
        # Recency weight: today = 1.0, yesterday = exp(-0.05) ≈ 0.95
        # Observations fade over ~20 days
        recency_weight = 1.0  # Current observation always has weight 1.0

        try:
            self.col_transitions.update_one(
                {
                    "user_id":     user_id,
                    "from_action": from_action,
                    "to_action":   to_action,
                    "time_slot":   time_slot or "Unknown",
                },
                {
                    "$inc": {
                        "count":  1,
                        "weight": recency_weight,
                    },
                    "$set":          {"last_updated": datetime.datetime.utcnow()},
                    "$setOnInsert":  {
                        "user_id":     user_id,
                        "from_action": from_action,
                        "to_action":   to_action,
                        "time_slot":   time_slot or "Unknown",
                        "created_at":  datetime.datetime.utcnow(),
                    },
                },
                upsert=True,
            )
        except Exception as e:
            print(f"[HabitLearner] transition update error: {e}")

    def _apply_recency_decay(self):
        """
        Decay old transition weights daily.
        Should be called by nightly maintenance in app.py.
        """
        try:
            # Multiply all weights by decay factor (0.95)
            decay = math.exp(-RECENCY_DECAY)
            self.col_transitions.update_many(
                {},
                {"$mul": {"weight": decay}}
            )
            # Remove transitions with very low weight (effectively forgotten)
            self.col_transitions.delete_many({"weight": {"$lt": 0.1}})
            print("[HabitLearner] Recency decay applied")
        except Exception as e:
            print(f"[HabitLearner] decay error: {e}")

    def _maybe_update_skill(self, user_id: str):
        """
        Check if SKILL.md needs updating based on observation thresholds.
        Delegates to SkillManager.
        """
        SKILL_UPDATE_THRESHOLD = 5

        try:
            habits = list(self.col_obs.find({
                "user":   user_id,
                "weight": {"$gte": SKILL_UPDATE_THRESHOLD},
            }))
            if not habits:
                return

            for h in habits:
                action    = h.get("action", "")
                instance  = h.get("zone_name") or h.get("instance", "")
                weight    = int(h.get("weight", 0))
                items     = h.get("interacting_items", [])
                time_slot = h.get("time_slot", "")

                if not action or not instance:
                    continue

                item_str = f" with {', '.join(items)}" if items else ""
                slot_str = (f" in {time_slot}"
                            if time_slot and time_slot != "Unknown" else "")
                bullet   = (f"- {action} near {instance}"
                            f"{item_str}{slot_str} ({weight} times)")

                self.skill_manager._insert_if_new(
                    user_id, "## Behavior Patterns", bullet)

                for item in items:
                    pref = (f"- User frequently uses {item} during "
                            f"{action}{slot_str}")
                    self.skill_manager._insert_if_new(
                        user_id, "## Preferences", pref)

        except Exception as e:
            print(f"[HabitLearner] skill update error: {e}")