import math
import datetime
from pymongo import MongoClient


OBJECT_CATEGORIES = {
    "bottle":     "drink", "cola":       "drink", "cup":        "drink",
    "juice":      "drink", "water":      "drink", "beer":       "drink",
    "coffee":     "drink", "tea":        "drink",
    "apple":      "food",  "banana":     "food",  "plate":      "food",
    "bowl":       "food",  "food":       "food",  "bread":      "food",
    "sandwich":   "food",  "pizza":      "food",
    "spoon":      "food",  "fork":       "food",  "chopsticks": "food",
    "broom":      "tool",  "mop":        "tool",  "pan":        "tool",
    "spatula":    "tool",  "knife":      "tool",
    "phone":      "device","laptop":     "device","keyboard":   "device",
    "remote":     "device","tablet":     "device","ipad":       "device",
    "book":       "media", "magazine":   "media", "newspaper":  "media",
    "notebook":   "media",
}


def _skeleton_to_semantic(skel_body: str, head_pitch: float,
                           hand_to_head: float = -1,
                           left_hand_to_head: float = -1,
                           spine_angle: float = -1,
                           arm_elevation: float = -1,
                           wrist_height: float = -999,
                           left_wrist_height: float = -999,
                           wrist_x: float = -999,
                           wrist_z: float = -999,
                           left_wrist_x: float = -999,
                           left_wrist_z: float = -999) -> str:
    hints = []

    best_h2h = -1
    if hand_to_head >= 0 and left_hand_to_head >= 0:
        best_h2h = min(hand_to_head, left_hand_to_head)
    elif hand_to_head >= 0:
        best_h2h = hand_to_head
    elif left_hand_to_head >= 0:
        best_h2h = left_hand_to_head

    # Hand proximity
    if best_h2h >= 0:
        if best_h2h < 0.38:
            if skel_body == "standing" and head_pitch > -999 and head_pitch > 20:
                hints.append("hand very close to face while looking down")
            else:
                hints.append("hand very close to face")
        elif best_h2h < 0.50:
            hints.append("hand close to face")

    # Wrist XZ: both hands forward at desk level (typing)
    if skel_body == "sitting":
        r_valid = wrist_x > -999 and wrist_z > -999
        l_valid = left_wrist_x > -999 and left_wrist_z > -999
        if r_valid and l_valid:
            if wrist_z > 0.05 and left_wrist_z > 0.05:
                if abs(wrist_height) < 0.15 and abs(left_wrist_height) < 0.15:
                    hints.append("both hands extended forward at desk level")

    # Spine (standing only)
    if spine_angle >= 0 and skel_body == "standing":
        if spine_angle > 25:
            hints.append("trunk leaning forward significantly")
        elif spine_angle > 12:
            hints.append("trunk leaning forward slightly")

    # Arm (standing only)
    if arm_elevation >= 0 and skel_body == "standing":
        if arm_elevation > 130:
            hints.append("arms pointing downward")
        elif arm_elevation < 55:
            hints.append("arms raised upward")
        elif 70 <= arm_elevation <= 110:
            hints.append("arms horizontal")

    # Wrist height (standing only)
    if wrist_height > -999 and skel_body == "standing":
        if wrist_height > 0.10:
            hints.append("wrist above hip level")
        elif wrist_height < -0.08:
            hints.append("wrist below hip level")

    # Head direction with context
    if head_pitch > -999 and skel_body:
        if skel_body == "sitting":
            if head_pitch < -15:
                hints.append("head tilting back (consistent with drinking)")
            elif 15 <= head_pitch <= 35:
                hints.append("head slightly down")
            elif head_pitch > 60:
                hints.append("head bent far forward (consistent with reading)")
            elif -5 <= head_pitch <= 10:
                hints.append("head facing forward")
        elif skel_body == "standing":
            if head_pitch < -15:
                hints.append("head tilting back (consistent with drinking)")
            elif head_pitch > 55:
                hints.append("head bent far forward")
            elif 20 <= head_pitch <= 55:
                if not any("looking down" in h for h in hints):
                    hints.append("head looking down")
            elif 20 <= head_pitch <= 35:
                hints.append("head slightly down")

    return ", ".join(hints) if hints else ""


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
        best_score = 0.65

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
                     hand_to_head=-1, wrist_height=-999,
                     left_hand_to_head=-1, left_wrist_height=-999,
                     wrist_x=-999, wrist_z=-999,
                     left_wrist_x=-999, left_wrist_z=-999):
    lines = []

    lines.append("=== Scene Graph ===")
    lines.append(f"Room: {room_name or 'Unknown'}")

    if virtual_hour is not None:
        try:
            h = float(virtual_hour)
            if h < 6:
                slot = "Dawn"
            elif h < 12:
                slot = "Morning"
            elif h < 14:
                slot = "LunchTime"
            elif h < 18:
                slot = "Afternoon"
            elif h < 22:
                slot = "Evening"
            else:
                slot = "Night"
            lines.append(f"Time: {h:.0f}:00 ({slot})")
        except Exception:
            pass

    body_str  = skel_body or "unknown"
    pitch_str = f"{head_pitch:.0f}°" if head_pitch and head_pitch > -999 else "unknown"
    lines.append(f"Person: body={body_str}, head_pitch={pitch_str}")

    posture = _skeleton_to_semantic(
        skel_body, head_pitch,
        hand_to_head=hand_to_head,
        left_hand_to_head=left_hand_to_head,
        spine_angle=spine_angle,
        arm_elevation=arm_elevation,
        wrist_height=wrist_height,
        left_wrist_height=left_wrist_height,
        wrist_x=wrist_x,
        wrist_z=wrist_z,
        left_wrist_x=left_wrist_x,
        left_wrist_z=left_wrist_z,
    )
    if posture:
        lines.append(f"Posture: {posture}")

    if held_object and held_object not in ("none", "unknown", ""):
        cat = OBJECT_CATEGORIES.get(held_object.lower(), "object")
        lines.append(f"Holding: {held_object} [{cat}]")
    else:
        lines.append("Holding: nothing")

    facing = _get_facing_target(user_pos, user_forward, db)
    tv_doc   = None
    tv_scene = None
    try:
        tv_doc   = db.device_states.find_one({"label": "tv"})
        tv_scene = db.scene_snapshots.find_one(
            {"label": {"$in": ["tv", "television"]}}, {"pos": 1})
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

    if tv_scene and user_pos:
        try:
            tv_pos = tv_scene.get("pos", [])
            if len(tv_pos) >= 2:
                ux2 = float(user_pos.get("x", 0))
                uz2 = float(user_pos.get("z", 0))
                tv_dist = math.sqrt((ux2 - tv_pos[0])**2 + (uz2 - tv_pos[1])**2)
                if tv_dist < 6.0:
                    lines.append(f"TV: {tv_dist:.1f}m away, currently {tv_state}")
                else:
                    lines.append(f"TV: {tv_state}")
            else:
                lines.append(f"TV: {tv_state}")
        except Exception:
            lines.append(f"TV: {tv_state}")
    elif tv_doc and facing not in ("tv", "television"):
        lines.append(f"TV: {tv_state}")



    lines.append("=== End Scene ===")
    return "\n".join(lines)