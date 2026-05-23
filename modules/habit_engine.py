import threading
import datetime
from pymongo import ReturnDocument

HABIT_THRESHOLD   = 5
REJECTION_PENALTY = -3
DEDUP_THRESHOLD   = 0.78

NO_WEIGHT_ACTIONS = {"PickingUp", "PuttingDown", "Walking", "Standing"}


class HabitEngine:

    def __init__(self, db, skill_manager, manifold_engine=None,
                 vector_memory=None, fat_threshold: int = HABIT_THRESHOLD):
        self.db              = db
        self.skill_manager   = skill_manager
        self.manifold_engine = manifold_engine
        self.vector_memory   = vector_memory
        self.fat_threshold   = fat_threshold

        self.col_obs    = db.observation_logs
        self.col_snap   = db.habit_snapshots
        self.col_seq    = db.activity_sequences
        self.col_skills = db.user_skills

    def record(self, user_id: str, action: str, zone_name: str,
               pos: list, virtual_hour: float, time_slot: str,
               interacting_items: list, raw_desc: str,
               room: str, instance: str, spatial_relations: dict,
               experiment_mode: str = "habit"):
        if not zone_name:
            print(f"[HabitEngine] Skipping — zone_name empty")
            return

        if action in NO_WEIGHT_ACTIONS:
            return

        today  = datetime.datetime.utcnow().strftime("%Y-%m-%d")
        pos_xy = [pos[0] / 10.0, pos[1] / 10.0] if pos else [0.0, 0.0]

        self._update_observation_log(
            user=user_id, action=action, zone_name=zone_name,
            instance=instance, time_slot=time_slot,
            interacting_items=interacting_items,
            spatial_relations=spatial_relations,
            pos_xy=pos_xy, room=room,
            raw_desc=raw_desc, today=today,
        )

        self._write_habit_snapshot(
            user=user_id, action=action, zone_name=zone_name,
            time_slot=time_slot, today=today,
        )

        self._update_activity_sequence(
            user=user_id, action=action,
            instance=zone_name, today=today,
        )

        if self.vector_memory and action not in NO_WEIGHT_ACTIONS:
            try:
                memory_text = (
                    f"{user_id} {time_slot} {room} "
                    f"{action} near {zone_name} "
                    f"with {' '.join(interacting_items)}"
                ).strip()
                self.vector_memory.add_memory(
                    user_id         = user_id,
                    action          = action,
                    furniture_label = zone_name,
                    vlm_description = memory_text,
                    detected_items  = interacting_items,
                    all_items       = interacting_items,
                )
            except Exception as e:
                print(f"[HabitEngine] FAISS write error: {e}")

        if self.manifold_engine is not None and experiment_mode != "recognition":
            try:
                self.manifold_engine.record_training_sample(
                    user_id        = user_id,
                    current_action = action,
                    virtual_hour   = virtual_hour,
                    user_pos       = {"x": pos[0] * 10, "z": pos[1] * 10}
                                     if pos else {},
                    prev_action    = action,
                )
            except Exception as e:
                print(f"[Manifold] {e}")

        threading.Thread(
            target=self._check_and_update_skill,
            args=(user_id,), daemon=True,
        ).start()

    def handle_rejection(self, user_id: str, intent: str, item: str):
        if not item:
            return
        result = self.col_obs.update_many(
            {"user": user_id, "interacting_items": item},
            {"$inc": {"weight": REJECTION_PENALTY}},
        )
        print(f"[HabitEngine] Rejection: '{item}' for {user_id} "
              f"({result.modified_count} entries)")
        bullet = f"- Do not proactively suggest {item} to this user"
        self._insert_if_new(user_id, "## What NOT to do", bullet)

    def handle_acceptance(self, user_id: str, intent: str, item: str):
        if not item:
            return
        self.col_obs.update_many(
            {"user": user_id, "interacting_items": item},
            {"$inc": {"weight": 1}},
        )
        print(f"[HabitEngine] Acceptance: '{item}' for {user_id}")

    def _update_observation_log(self, user, action, zone_name,
                                 instance, time_slot, interacting_items,
                                 spatial_relations, pos_xy, room,
                                 raw_desc, today):
        try:
            self.col_obs.find_one_and_update(
                {"user": user, "zone_name": zone_name,
                 "action": action, "time_slot": time_slot},
                {
                    "$inc":      {"weight": 1},
                    "$addToSet": {"interacting_items":
                                  {"$each": interacting_items}},
                    "$set": {
                        "observed_relations": spatial_relations,
                        "pos":               pos_xy,
                        "room":              room,
                        "instance":          instance,
                        "last_seen":         datetime.datetime.utcnow(),
                        "last_date":         today,
                        "raw_vlm_desc":      raw_desc,
                    },
                    "$setOnInsert": {
                        "user":      user,
                        "zone_name": zone_name,
                        "action":    action,
                        "time_slot": time_slot,
                    },
                },
                upsert=True,
                return_document=ReturnDocument.AFTER,
            )
        except Exception as e:
            print(f"[HabitEngine] obs_log write error: {e}")

    def _write_habit_snapshot(self, user, action, zone_name,
                               time_slot, today):
        try:
            self.col_snap.update_one(
                {"user": user, "action": action,
                 "canonical_key": zone_name, "date": today},
                {
                    "$inc": {"daily_count": 1},
                    "$setOnInsert": {
                        "user":          user,
                        "action":        action,
                        "canonical_key": zone_name,
                        "date":          today,
                        "time_slot":     time_slot,
                    },
                },
                upsert=True,
            )
        except Exception as e:
            print(f"[HabitEngine] habit_snapshot write error: {e}")

    def _update_activity_sequence(self, user, action, instance, today):
        try:
            self.col_seq.update_one(
                {"user": user, "date": today},
                {
                    "$push": {
                        "sequence": {
                            "action":    action,
                            "instance":  instance,
                            "timestamp": datetime.datetime.utcnow(),
                        }
                    },
                    "$setOnInsert": {"user": user, "date": today},
                },
                upsert=True,
            )
        except Exception as e:
            print(f"[HabitEngine] activity_seq write error: {e}")

    def _check_and_update_skill(self, user_id: str):
        try:
            habits = list(self.col_obs.find({
                "user":   user_id,
                "weight": {"$gte": self.fat_threshold},
            }))
            if not habits:
                return

            updated = False
            for h in habits:
                action    = h.get("action", "")
                instance  = h.get("zone_name") or h.get("instance", "")
                weight    = int(h.get("weight", 0))
                items     = h.get("interacting_items", [])
                time_slot = h.get("time_slot", "")

                if not action or not instance:
                    continue

                item_str  = f" with {', '.join(items)}" if items else ""
                slot_str  = (f" in {time_slot}"
                             if time_slot and time_slot != "Unknown" else "")
                bp_bullet = (f"- {action} near {instance}"
                             f"{item_str}{slot_str} ({weight} times)")

                if self._insert_if_new(user_id, "## Behavior Patterns", bp_bullet):
                    updated = True

                for item in items:
                    pref = (f"- User frequently uses {item} during "
                            f"{action}{slot_str} "
                            f"(inferred from {weight} observations)")
                    if self._insert_if_new(user_id, "## Preferences", pref):
                        updated = True

            if updated:
                print(f"[HabitEngine] SKILL.md updated for {user_id}")

        except Exception as e:
            print(f"[HabitEngine] skill update error: {e}")

    def _insert_if_new(self, user_id: str, section: str,
                        bullet: str) -> bool:
        doc = self.col_skills.find_one({"user_id": user_id})
        if not doc:
            return False
        current = doc.get("skill_md", "")

        if self._is_duplicate(bullet, current, section):
            return False

        from modules.skill_manager import (
            _insert_bullet, _normalize_bullets, validate_skill)
        updated = _insert_bullet(current, section, bullet)
        updated = _normalize_bullets(updated)
        valid, reason = validate_skill(updated)
        if not valid:
            return False

        self.skill_manager._save(user_id, updated)
        self.skill_manager._chunk_skill_md(updated, user_id)
        print(f"[HabitEngine] Written to {section}: {bullet[:60]}")
        return True

    def _is_duplicate(self, new_bullet: str, skill_md: str,
                       section: str) -> bool:
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
            model = SentenceTransformer("paraphrase-MiniLM-L6-v2")
            new_v = model.encode([new_bullet], normalize_embeddings=True)[0]
            old_v = model.encode(bullets, normalize_embeddings=True)
            return float((old_v @ new_v).max()) >= DEDUP_THRESHOLD
        except Exception:
            return new_bullet.lower() in skill_md.lower()