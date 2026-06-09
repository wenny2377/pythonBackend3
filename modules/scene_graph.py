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
    "mouse":      "device",
    "book":       "media", "magazine":   "media", "newspaper":  "media",
    "notebook":   "media",
}

HELD_OBJECT_TO_ACTION = {
    "bottle":   "Drinking",
    "cola":     "Drinking",
    "cup":      "Drinking",
    "bowl":     "Eating",
    "plate":    "Eating",
    "apple":    "Eating",
    "banana":   "Eating",
    "pan":      "Cooking",
    "broom":    "Cleaning",
    "book":     "Reading",
    "phone":    "PhoneUse",
    "remote":   "Watching",
    "keyboard": "Typing",
    "mouse":    "Typing",
}


# Stores previous wrist positions for movement trend detection
# key: user_id, value: (wrist_height, left_wrist_height, timestamp)
_wrist_history: dict = {}


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
                           left_wrist_z: float = -999,
                           user_id: str = "",
                           prev_wrist_height: float = -999,
                           prev_hand_to_head: float = -1) -> str:
    hints = []

    # ── Hand to face distance ─────────────────────────────────────────────
    best_h2h = -1
    if hand_to_head >= 0 and left_hand_to_head >= 0:
        best_h2h = min(hand_to_head, left_hand_to_head)
    elif hand_to_head >= 0:
        best_h2h = hand_to_head
    elif left_hand_to_head >= 0:
        best_h2h = left_hand_to_head

    if best_h2h >= 0:
        if best_h2h < 0.35:
            hints.append("hand very close to face")
        elif best_h2h < 0.50:
            hints.append("hand close to face")

        # L2: Hand movement trend (approaching vs moving away from face)
        if prev_hand_to_head >= 0 and best_h2h >= 0:
            delta = best_h2h - prev_hand_to_head
            if delta < -0.05:
                hints.append("hand moving toward face")
            elif delta > 0.05:
                hints.append("hand moving away from face")

    # ── Wrist position (Typing detection) ────────────────────────────────
    if skel_body == "sitting":
        r_valid = wrist_x > -999 and wrist_z > -999
        l_valid = left_wrist_x > -999 and left_wrist_z > -999
        if r_valid and l_valid:
            if wrist_z > 0.05 and left_wrist_z > 0.05:
                if abs(wrist_height) < 0.15 and abs(left_wrist_height) < 0.15:
                    hints.append("both hands extended forward at desk level")

    # ── Wrist height trend ───────────────────────────────────────────────
    if wrist_height > -999 and prev_wrist_height > -999:
        delta_h = wrist_height - prev_wrist_height
        if delta_h > 0.08:
            hints.append("wrist rising (hand lifting up)")
        elif delta_h < -0.08:
            hints.append("wrist lowering (hand coming down)")

    # ── Head pitch ───────────────────────────────────────────────────────
    if head_pitch > -999:
        if head_pitch < -55:
            hints.append("head strongly tilted back (consistent with lying down)")
        elif head_pitch < -18:
            hints.append("head tilted back (consistent with drinking)")
        elif head_pitch > 65:
            hints.append("head bent far forward (consistent with reading)")
        elif head_pitch > 45:
            hints.append("head looking down significantly")
        elif head_pitch > 20:
            hints.append("head looking slightly down")
        elif -10 <= head_pitch <= 10:
            hints.append("head facing forward")

    # ── Arm elevation ────────────────────────────────────────────────────
    if arm_elevation >= 0:
        if arm_elevation > 165:
            hints.append("arm raised very high (consistent with opening/reaching)")
        elif arm_elevation > 130:
            hints.append("arm raised")
        elif arm_elevation < 60:
            hints.append("arm lowered (consistent with sitting drink or resting)")

    return ", ".join(hints) if hints else ""


def _infer_body_position(head_pitch: float,
                          hand_to_head: float,
                          arm_elevation: float) -> str:
    if head_pitch > -999 and head_pitch < -55:
        return "lying"

    if head_pitch > -999 and head_pitch < -18:
        if hand_to_head >= 0 and hand_to_head < 0.35:
            return "sitting"

    if hand_to_head >= 0 and hand_to_head < 0.35 and arm_elevation >= 0 and arm_elevation > 130:
        return "standing"

    if head_pitch > -999 and head_pitch > 65:
        return "sitting"

    if head_pitch > -999 and 10 < head_pitch < 25:
        if hand_to_head >= 0 and hand_to_head < 0.40:
            return "sitting"

    if arm_elevation >= 0 and arm_elevation > 165:
        return "standing"

    return "unknown"


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
                     left_wrist_x=-999, left_wrist_z=-999,
                     prev_wrist_height=-999, prev_hand_to_head=-1,
                     held_age=0.0):
    lines = []

    lines.append("=== Scene Graph ===")
    lines.append(f"Room: {room_name or 'Unknown'}")

    if virtual_hour is not None:
        try:
            h = float(virtual_hour)
            if h < 6:
                slot = "Dawn"
            elif h < 10:
                slot = "Morning"
            elif h < 13:
                slot = "Noon"
            elif h < 18:
                slot = "Afternoon"
            elif h < 22:
                slot = "Evening"
            else:
                slot = "Night"
            lines.append(f"Time: {h:.0f}:00 ({slot})")
        except Exception:
            pass

    inferred_body = _infer_body_position(
        head_pitch    = head_pitch    if head_pitch > -999 else -999,
        hand_to_head  = hand_to_head  if hand_to_head >= 0 else -1,
        arm_elevation = arm_elevation if arm_elevation >= 0 else -1,
    )
    body_str  = inferred_body if inferred_body != "unknown" else (skel_body or "unknown")
    pitch_str = f"{head_pitch:.0f}" if head_pitch and head_pitch > -999 else "unknown"
    lines.append(f"Person: body={body_str}, head_pitch={pitch_str}")

    posture = _skeleton_to_semantic(
        skel_body          = body_str,
        head_pitch         = head_pitch,
        hand_to_head       = hand_to_head,
        left_hand_to_head  = left_hand_to_head,
        spine_angle        = spine_angle,
        arm_elevation      = arm_elevation,
        wrist_height       = wrist_height,
        left_wrist_height  = left_wrist_height,
        wrist_x            = wrist_x,
        wrist_z            = wrist_z,
        left_wrist_x       = left_wrist_x,
        left_wrist_z       = left_wrist_z,
        user_id            = user_id,
        prev_wrist_height  = prev_wrist_height,
        prev_hand_to_head  = prev_hand_to_head,
    )
    if posture:
        lines.append(f"Posture cues: {posture}")

    if held_object and held_object not in ("none", "unknown", ""):
        cat = OBJECT_CATEGORIES.get(held_object.lower(), "object")
        # Natural language description of held object + duration
        if held_age and held_age > 0:
            if held_age < 10:
                duration_str = "just picked up"
            elif held_age < 60:
                duration_str = f"holding for {int(held_age)} seconds"
            else:
                duration_str = f"holding for over a minute"
        else:
            duration_str = "holding"
        lines.append(f"Holding: {held_object} ({cat}, {duration_str})")
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
