import datetime

SKILL_UPDATE_THRESHOLD = 5


class PatternAnalyzer:

    def __init__(self, db):
        self.db = db
        self.col_obs = db.observation_logs
        self.col_patterns = db.behavior_patterns

    def analyze_user(self, user_id: str):
        habits = list(self.col_obs.find({
            "user": user_id,
            "weight": {"$gte": SKILL_UPDATE_THRESHOLD},
        }))
        if not habits:
            return []

        updated = []
        for h in habits:
            action = h.get("action", "")
            zone_name = h.get("zone_name") or h.get("instance", "")
            time_slot = h.get("time_slot", "")
            weight = h.get("weight", 0)
            items = h.get("interacting_items", [])

            if not action or not zone_name:
                continue

            pattern = self._write_pattern(
                user_id=user_id, action=action, zone_name=zone_name,
                time_slot=time_slot, sample_count=weight, items=items)
            updated.append(pattern)

        return updated

    def _write_pattern(self, user_id, action, zone_name, time_slot,
                        sample_count, items):
        item_counts = {}
        for item in items:
            item_counts[item] = item_counts.get(item, 0) + 1

        common_items = sorted(
            item_counts.keys(), key=lambda k: item_counts[k], reverse=True)

        confidence = min(1.0, sample_count / 10.0)

        doc = {
            "user_id": user_id,
            "action": action,
            "zone_name": zone_name,
            "time_slot": time_slot or "Unknown",
            "common_items": common_items,
            "item_counts": item_counts,
            "sample_count": sample_count,
            "confidence": round(confidence, 3),
            "last_analyzed": datetime.datetime.utcnow(),
        }

        self.col_patterns.update_one(
            {
                "user_id": user_id,
                "action": action,
                "zone_name": zone_name,
                "time_slot": time_slot or "Unknown",
            },
            {"$set": doc},
            upsert=True,
        )
        return doc

    def get_patterns(self, user_id: str, action: str = None,
                      time_slot: str = None) -> list:
        query = {"user_id": user_id}
        if action:
            query["action"] = action
        if time_slot:
            query["time_slot"] = time_slot
        return list(self.col_patterns.find(query).sort("confidence", -1))

    def preferred_item(self, user_id: str, available_labels: set,
                        need_type_actions: list) -> str:
        patterns = list(self.col_patterns.find(
            {"user_id": user_id, "action": {"$in": need_type_actions}}
        ).sort("confidence", -1))

        for p in patterns:
            for item in p.get("common_items", []):
                if item.lower() in available_labels:
                    return item
        return ""