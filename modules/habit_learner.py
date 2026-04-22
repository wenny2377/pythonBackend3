import threading
from datetime import datetime
from pymongo import MongoClient
from config import Config

HABIT_THRESHOLD   = 5
REJECTION_PENALTY = -3


class HabitLearner:

    def __init__(self, db_client, skill_manager):
        self.db            = db_client[Config.DB_NAME]
        self.skill_manager = skill_manager

    def check_and_update(self, user_id: str):
        def _bg():
            try:
                self._run_habit_update(user_id)
            except Exception as e:
                print(f"[HabitLearner] Error: {e}", flush=True)
                import traceback; traceback.print_exc()

        threading.Thread(target=_bg, daemon=True).start()

    def handle_rejection(self, user_id: str, intent: str, item: str):
        if not item:
            return

        result = self.db.observation_logs.update_many(
            {"user": user_id, "interacting_items": item},
            {"$inc": {"weight": REJECTION_PENALTY}},
        )
        print(f"[HabitLearner] Rejection penalty applied to '{item}' "
              f"for {user_id} ({result.modified_count} entries)", flush=True)

        bullet = f"- Do not proactively suggest {item} to this user"
        self._insert_if_new(user_id, "## What NOT to do", bullet)

    def handle_acceptance(self, user_id: str, intent: str, item: str):
        if not item:
            return

        self.db.observation_logs.update_many(
            {"user": user_id, "interacting_items": item},
            {"$inc": {"weight": 1}},
        )
        print(f"[HabitLearner] Acceptance reinforcement for '{item}' "
              f"for {user_id}", flush=True)

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

            item_str  = f" with {', '.join(items)}" if items else ""
            bp_bullet = f"- {action} near {instance}{item_str} ({weight} times)"
            changed   = self._insert_if_new(user_id, "## Behavior Patterns", bp_bullet)
            if changed:
                updated = True

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
        sm  = self.skill_manager
        doc = self.db.user_skills.find_one({"user_id": user_id})
        if not doc:
            return False

        current = doc.get("skill_md", "")

        print(f"\n[SkillEvolution] Candidate: \"{bullet[:60]}\"", flush=True)
        print(f"[SkillEvolution] Section  : {section}", flush=True)

        if self._is_duplicate(bullet, current, section):
            print(f"[SkillEvolution] x Dedup check FAILED: too similar to existing",
                  flush=True)
            return False
        print(f"[SkillEvolution] v Dedup check PASSED", flush=True)

        from modules.skill_manager import _insert_bullet, _normalize_bullets, validate_skill
        updated = _insert_bullet(current, section, bullet)
        updated = _normalize_bullets(updated)

        valid, reason = validate_skill(updated)
        if not valid:
            print(f"[SkillEvolution] x Validate FAILED: {reason}", flush=True)
            return False
        print(f"[SkillEvolution] v Validate PASSED", flush=True)
        print(f"[SkillEvolution] -> WRITTEN to {section}", flush=True)

        sm._save(user_id, updated)
        sm._chunk_skill_md(updated, user_id)
        return True

    def _is_duplicate(self, new_bullet: str, skill_md: str, section: str,
                      threshold: float = 0.78) -> bool:
        try:
            import numpy as np
            from sentence_transformers import SentenceTransformer

            idx = skill_md.find(section)
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

            model    = SentenceTransformer("paraphrase-MiniLM-L6-v2")
            new_vec  = model.encode([new_bullet], normalize_embeddings=True)[0]
            old_vecs = model.encode(bullets, normalize_embeddings=True)

            sims     = np.dot(old_vecs, new_vec)
            max_sim  = float(sims.max())
            print(f"[SkillEvolution]   max similarity: {max_sim:.3f} "
                  f"(threshold: {threshold})", flush=True)
            return max_sim >= threshold

        except Exception as e:
            print(f"[HabitLearner] duplicate check failed: {e}", flush=True)
            return new_bullet.lower() in skill_md.lower()