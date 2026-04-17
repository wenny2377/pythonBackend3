import threading
from datetime import datetime
from pymongo import MongoClient
from config import Config

HABIT_THRESHOLD    = 5    # observations needed to write to SKILL.md
REJECTION_PENALTY  = -3   # weight deduction on user rejection


class HabitLearner:

    def __init__(self, db_client, skill_manager):
        self.db            = db_client[Config.DB_NAME]
        self.skill_manager = skill_manager

    # ─────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────

    def check_and_update(self, user_id: str):
        """
        Check observation_logs for habits that have reached the threshold.
        Called in background after each /predict episode.
        """
        def _bg():
            try:
                self._run_habit_update(user_id)
            except Exception as e:
                print(f"[HabitLearner] Error: {e}", flush=True)
                import traceback; traceback.print_exc()

        threading.Thread(target=_bg, daemon=True).start()

    def handle_rejection(self, user_id: str, intent: str, item: str):
        """
        Called when user rejects a service proposal.
        Penalizes the relevant habit and writes a What NOT to do rule.
        """
        if not item:
            return

        # Penalize weight
        result = self.db.observation_logs.update_many(
            {
                "user":               user_id,
                "interacting_items":  item,
            },
            {"$inc": {"weight": REJECTION_PENALTY}},
        )
        print(f"[HabitLearner] Rejection penalty applied to '{item}' "
              f"for {user_id} ({result.modified_count} entries)", flush=True)

        # Write prohibition rule
        bullet = f"- Do not proactively suggest {item} to this user"
        self._insert_if_new(user_id, "## What NOT to do", bullet)

    def handle_acceptance(self, user_id: str, intent: str, item: str):
        """
        Called when user accepts a service proposal.
        Reinforces the relevant habit.
        """
        if not item:
            return

        self.db.observation_logs.update_many(
            {
                "user":              user_id,
                "interacting_items": item,
            },
            {"$inc": {"weight": 1}},
        )
        print(f"[HabitLearner] Acceptance reinforcement for '{item}' "
              f"for {user_id}", flush=True)

    # ─────────────────────────────────────────────
    # Internal
    # ─────────────────────────────────────────────

    def _run_habit_update(self, user_id: str):
        habits = list(self.db.observation_logs.find({
            "user":   user_id,
            "weight": {"$gte": HABIT_THRESHOLD},
        }))

        if not habits:
            return

        updated = False
        for h in habits:
            action   = h.get("action", "")
            instance = h.get("instance", "")
            weight   = int(h.get("weight", 0))
            items    = h.get("interacting_items", [])

            if not action or not instance:
                continue

            # Write to Behavior Patterns
            item_str = f" with {', '.join(items)}" if items else ""
            bp_bullet = f"- {action} near {instance}{item_str} ({weight} times)"
            changed = self._insert_if_new(user_id, "## Behavior Patterns", bp_bullet)
            if changed:
                updated = True

            # Infer preference from frequently used items
            for item in items:
                pref_bullet = (
                    f"- User frequently uses {item} during {action} "
                    f"(inferred from {weight} observations)"
                )
                changed = self._insert_if_new(user_id, "## Preferences", pref_bullet)
                if changed:
                    updated = True

        if updated:
            print(f"[HabitLearner] SKILL.md auto-updated for {user_id}", flush=True)

    def _insert_if_new(self, user_id: str, section: str, bullet: str) -> bool:
        """
        Insert bullet into SKILL.md section only if semantically new.
        Uses FAISS similarity to detect near-duplicates.
        Returns True if inserted.
        """
        sm  = self.skill_manager
        doc = self.db.user_skills.find_one({"user_id": user_id})
        if not doc:
            return False

        current = doc.get("skill_md", "")

        # Check for semantic duplicate via FAISS
        if self._is_duplicate(bullet, current, section):
            print(f"[HabitLearner] Duplicate skipped: {bullet[:60]}", flush=True)
            return False

        from modules.skill_manager import _insert_bullet, _normalize_bullets, validate_skill
        updated = _insert_bullet(current, section, bullet)
        updated = _normalize_bullets(updated)

        valid, reason = validate_skill(updated)
        if not valid:
            print(f"[HabitLearner] Validate failed: {reason}", flush=True)
            return False

        sm._save(user_id, updated)
        sm._chunk_skill_md(updated, user_id)
        print(f"[HabitLearner] Inserted into {section}: {bullet[:60]}", flush=True)
        return True

    def _is_duplicate(self, new_bullet: str, skill_md: str, section: str,
                      threshold: float = 0.92) -> bool:
        """
        Check if new_bullet is semantically similar to existing bullets
        in the target section using SBERT cosine similarity.
        """
        try:
            import numpy as np
            from sentence_transformers import SentenceTransformer

            # Extract existing bullets in the section
            idx   = skill_md.find(section)
            if idx == -1:
                return False

            after = skill_md[idx:]
            end   = len(after)
            for s in ["## Behavior Patterns", "## Preferences",
                      "## How to Handle Requests", "## What NOT to do"]:
                if s == section:
                    continue
                i = after.find(s, len(section))
                if i != -1 and i < end:
                    end = i

            block   = after[:end]
            bullets = [l.strip() for l in block.split('\n')
                       if l.strip().startswith('-')]

            if not bullets:
                return False

            model   = SentenceTransformer("paraphrase-MiniLM-L6-v2")
            new_vec = model.encode([new_bullet], normalize_embeddings=True)[0]
            old_vecs = model.encode(bullets, normalize_embeddings=True)

            sims = np.dot(old_vecs, new_vec)
            return float(sims.max()) >= threshold

        except Exception as e:
            print(f"[HabitLearner] FAISS duplicate check failed: {e}", flush=True)
            # Fall back to exact match
            return new_bullet.lower() in skill_md.lower()