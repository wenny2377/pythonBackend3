import math
import datetime
from pymongo import MongoClient


OBJECT_CATEGORIES = {
    "bottle":   "drink", "cola":     "drink", "cup":      "drink",
    "juice":    "drink", "water":    "drink", "beer":     "drink",
    "coffee":   "drink", "tea":      "drink",
    "apple":    "food",  "banana":   "food",  "plate":    "food",
    "bowl":     "food",  "food":     "food",  "bread":    "food",
    "sandwich": "food",  "pizza":    "food",
    "broom":    "tool",  "mop":      "tool",  "pan":      "tool",
    "spatula":  "tool",  "knife":    "tool",
    "phone":    "device","laptop":   "device","keyboard": "device",
    "remote":   "device","book":     "media", "magazine": "media",
}


def _skeleton_to_semantic(skel_body: str, head_pitch: float,
                           hand_to_head: float = -1,
                           spine_angle: float = -1,
                           arm_elevation: float = -1,
                           wrist_height: float = -999) -> str:
    hints = []

    if hand_to_head >= 0:
        if hand_to_head < 0.38:
            hints.append("hand near mouth (eating or drinking)")
        elif hand_to_head < 0.50:
            hints.append("hand near face (possible phone use)")

    if spine_angle >= 0 and skel_body == "standing":
        if spine_angle > 25:
            hints.append("leaning forward significantly (cleaning or cooking)")
        elif spine_angle > 12:
            hints.append("leaning forward slightly")

    if arm_elevation >= 0 and skel_body == "standing":
        if arm_elevation > 130:
            hints.append("arms reaching downward (cleaning posture)")
        elif arm_elevation < 55:
            hints.append("arms raised upward (drinking or phone use)")
        elif 70 <= arm_elevation <= 110:
            hints.append("arms horizontal (typing posture)")

    if wrist_height > -999 and skel_body == "standing":
        if wrist_height > 0.10:
            hints.append("wrist above hip level (raised arm activity)")
        elif wrist_height < -0.08:
            hints.append("wrist below hip level (cleaning posture)")

    if head_pitch > -999 and skel_body:
        if skel_body == "sitting":
            if head_pitch < -15:
                hints.append("head tilted back (drinking posture)")
            elif 15 <= head_pitch <= 28:
                hints.append("head slightly down (eating posture)")
            elif head_pitch > 60:
                hints.append("head bent far forward (reading posture)")
            elif -5 <= head_pitch <= 10:
                hints.append("head neutral (resting or watching)")
        elif skel_body == "standing":
            if head_pitch > 55 and not any("cleaning" in h for h in hints):
                hints.append("head bent far forward (cleaning or working posture)")
            elif 20 <= head_pitch <= 35 and not any("cleaning" in h or "cooking" in h for h in hints):
                hints.append("head slightly down (cooking posture)")

    return ", ".join(hints) if hints else ""


def _dist(pos_a, pos_b):
    if not pos_a or not pos_b:
        return 999.0
    try:
        ax = float(pos_a.get("x", 0)) if isinstance(pos_a, dict) else float(pos_a[0])
        az = float(pos_a.get("z", 0)) if isinstance(pos_a, dict) else float(pos_a[1])
        bx = float(pos_b[0]) if isinstance(pos_b, list) else float(pos_b.get("x", 0))
        bz = float(pos_b[1]) if isinstance(pos_b, list) else float(pos_b.get("z", 0))
        return math.sqrt((ax - bx) ** 2 + (az - bz) ** 2)
    except Exception:
        return 999.0


def _get_facing_target(user_pos, user_forward, db, max_dist=6.0):
    if not user_pos or not user_forward:
        return "unknown"
    try:
        ux = float(user_pos.get("x", 0))
        uz = float(user_pos.get("z", 0))
        fx = float(user_forward.get("x", 0))
        fz = float(user_forward.get("z", 0))
        flen = math.sqrt(fx ** 2 + fz ** 2)
        if flen < 0.01:
            return "unknown"
        fx /= flen
        fz /= flen

        best_label = "unknown"
        best_score = 0.3

        for doc in db.scene_snapshots.find({}, {"label": 1, "pos": 1}):
            pos = doc.get("pos")
            if not isinstance(pos, list) or len(pos) < 2:
                continue
            dx = pos[0] - ux
            dz = pos[1] - uz
            dist = math.sqrt(dx ** 2 + dz ** 2)
            if dist < 0.1 or dist > max_dist:
                continue
            cos_a = (fx * dx / dist) + (fz * dz / dist)
            if cos_a > best_score:
                best_score = cos_a
                best_label = doc.get("label", "unknown")

        return best_label
    except Exception:
        return "unknown"


def build_scene_text(user_pos, user_forward, room_name,
                     skel_body, head_pitch, held_object, db,
                     user_id="", virtual_hour=None,
                     spine_angle=-1, arm_elevation=-1,
                     hand_to_head=-1, wrist_height=-999):
    lines = []

    lines.append(f"=== Scene Graph ===")
    lines.append(f"Room: {room_name or 'Unknown'}")

    if virtual_hour is not None:
        try:
            h = float(virtual_hour)
            slot = ("Morning" if h < 10 else
                    "Noon" if h < 13 else
                    "Afternoon" if h < 18 else
                    "Evening" if h < 22 else "Night")
            lines.append(f"Time: {h:.0f}:00 ({slot})")
        except Exception:
            pass

    body_str  = skel_body or "unknown"
    pitch_str = f"{head_pitch:.0f}°" if head_pitch and head_pitch > -999 else "unknown"
    lines.append(f"Person: body={body_str}, head_pitch={pitch_str}")

    posture = _skeleton_to_semantic(
        skel_body, head_pitch,
        hand_to_head=hand_to_head,
        spine_angle=spine_angle,
        arm_elevation=arm_elevation,
        wrist_height=wrist_height,
    )
    if posture:
        lines.append(f"Posture: {posture}")

    if held_object and held_object not in ("none", "unknown", ""):
        cat = OBJECT_CATEGORIES.get(held_object.lower(), "object")
        lines.append(f"Holding: {held_object} [{cat}]")
    else:
        lines.append(f"Holding: nothing")

    facing = _get_facing_target(user_pos, user_forward, db)
    tv_doc = None
    try:
        tv_doc = db.device_states.find_one({"label": "tv"})
    except Exception:
        pass
    tv_state = tv_doc.get("state", "off") if tv_doc else "unknown"
    if facing in ("tv", "television"):
        lines.append(f"Facing: {facing} (currently {tv_state})")
    else:
        lines.append(f"Facing: {facing}")

    if user_pos:
        try:
            ux = float(user_pos.get("x", 0))
            uz = float(user_pos.get("z", 0))
            nearby_furniture = []
            for doc in db.scene_snapshots.find(
                    {"room": {"$regex": room_name, "$options": "i"}} if room_name else {},
                    {"label": 1, "pos": 1}):
                pos = doc.get("pos")
                if not isinstance(pos, list) or len(pos) < 2:
                    continue
                d = math.sqrt((ux - pos[0]) ** 2 + (uz - pos[1]) ** 2)
                if d <= 2.0:
                    label = doc.get("label", "")
                    raw_contents = [
                        obj["label"] for obj in
                        db.dynamic_objects.find(
                            {"last_seen_on": label},
                            {"label": 1}
                        )
                    ]
                    tagged = []
                    for item in raw_contents:
                        cat = OBJECT_CATEGORIES.get(item.lower(), "")
                        tagged.append(f"{item} [{cat}]" if cat else item)
                    entry = f"{label} ({d:.1f}m away)"
                    if tagged:
                        entry += f", contains: {', '.join(tagged)}"
                    nearby_furniture.append((d, entry))

            if nearby_furniture:
                nearby_furniture.sort(key=lambda x: x[0])
                lines.append("Nearby furniture:")
                for _, entry in nearby_furniture[:4]:
                    lines.append(f"  - {entry}")
        except Exception:
            pass

    if tv_doc and facing not in ("tv", "television"):
        lines.append(f"TV: {tv_state}")

    if user_id:
        try:
            prev_doc = db.activity_sequences.find_one(
                {"user": user_id}, sort=[("date", -1)])
            if prev_doc and prev_doc.get("sequence"):
                seq = prev_doc["sequence"]
                if seq:
                    recent = [e.get("action", "") for e in seq[-3:] if e.get("action")]
                    if recent:
                        lines.append(f"Recent actions: {' → '.join(recent)}")
        except Exception:
            pass

    lines.append("=== End Scene ===")
    return "\n".join(lines)