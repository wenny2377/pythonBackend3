"""
observation_store.py
────────────────────
Memory Layer — Layer 1

Single responsibility:
  Write raw observations to observation_logs and habit_snapshots.
  Does NOT update SKILL.md, does NOT update transition_counts.
  Those are handled by HabitLearner.

Collections written:
  - observation_logs     (TTL 14 days, weighted by frequency)
  - habit_snapshots      (daily counts per action/zone/user)
  - activity_sequences   (chronological action sequence per user per day)
"""

import datetime
from pymongo import ReturnDocument

NO_RECORD_ACTIONS = {
    "PickingUp", "PuttingDown", "Walking", "Standing", "StandUp"
}


class ObservationStore:

    def __init__(self, db):
        self.db      = db
        self.col_obs  = db.observation_logs
        self.col_snap = db.habit_snapshots
        self.col_seq  = db.activity_sequences

    # ── Public API ────────────────────────────────────────────────────────────

    def record(self,
               user_id: str,
               action: str,
               zone_name: str,
               instance: str,
               time_slot: str,
               interacting_items: list,
               spatial_relations: dict,
               pos_xy: list,
               room: str,
               raw_desc: str,
               today: str = None):
        """
        Record a single observation.
        Called once per HAR episode from _process_predict in app.py.
        """
        if action in NO_RECORD_ACTIONS:
            return
        if not zone_name:
            return

        today = today or datetime.datetime.utcnow().strftime("%Y-%m-%d")

        self._write_observation_log(
            user_id=user_id, action=action, zone_name=zone_name,
            instance=instance, time_slot=time_slot,
            interacting_items=interacting_items,
            spatial_relations=spatial_relations,
            pos_xy=pos_xy, room=room, raw_desc=raw_desc, today=today,
        )
        self._write_habit_snapshot(
            user_id=user_id, action=action, zone_name=zone_name,
            time_slot=time_slot, today=today,
        )
        self._write_activity_sequence(
            user_id=user_id, action=action,
            instance=zone_name, today=today,
        )

    def get_recent_sequence(self, user_id: str, limit: int = 2) -> list:
        """
        Return the last N actions from activity_sequences.
        Used by HabitLearner and ProactiveService.
        """
        doc = self.col_seq.find_one(
            {"user": user_id}, sort=[("date", -1)])
        if not doc:
            return []
        seq = doc.get("sequence", [])
        return [s.get("action", "") for s in seq[-limit:]]

    def get_observation_weight(self, user_id: str, action: str,
                                zone_name: str, time_slot: str) -> int:
        """Return the weight of a specific observation pattern."""
        doc = self.col_obs.find_one({
            "user": user_id, "action": action,
            "zone_name": zone_name, "time_slot": time_slot,
        })
        return int(doc.get("weight", 0)) if doc else 0

    # ── Private writers ───────────────────────────────────────────────────────

    def _write_observation_log(self, user_id, action, zone_name,
                                instance, time_slot, interacting_items,
                                spatial_relations, pos_xy, room,
                                raw_desc, today):
        try:
            self.col_obs.find_one_and_update(
                {"user": user_id, "zone_name": zone_name,
                 "action": action, "time_slot": time_slot},
                {
                    "$inc":      {"weight": 1},
                    "$addToSet": {
                        "interacting_items": {"$each": interacting_items}
                    },
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
                        "user": user_id, "zone_name": zone_name,
                        "action": action, "time_slot": time_slot,
                    },
                },
                upsert=True,
                return_document=ReturnDocument.AFTER,
            )
        except Exception as e:
            print(f"[ObservationStore] obs_log write error: {e}")

    def _write_habit_snapshot(self, user_id, action, zone_name,
                               time_slot, today):
        try:
            self.col_snap.update_one(
                {"user": user_id, "action": action,
                 "canonical_key": zone_name, "date": today},
                {
                    "$inc": {"daily_count": 1},
                    "$setOnInsert": {
                        "user": user_id, "action": action,
                        "canonical_key": zone_name,
                        "date": today, "time_slot": time_slot,
                    },
                },
                upsert=True,
            )
        except Exception as e:
            print(f"[ObservationStore] habit_snapshot write error: {e}")

    def _write_activity_sequence(self, user_id, action, instance, today):
        try:
            self.col_seq.update_one(
                {"user": user_id, "date": today},
                {
                    "$push": {
                        "sequence": {
                            "action":    action,
                            "instance":  instance,
                            "timestamp": datetime.datetime.utcnow(),
                        }
                    },
                    "$setOnInsert": {"user": user_id, "date": today},
                },
                upsert=True,
            )
        except Exception as e:
            print(f"[ObservationStore] activity_seq write error: {e}")