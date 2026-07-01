import math

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

SKELETON_THRESHOLDS = {
    "body_tilted":       20.0,
    "sitting_ratio":     0.65,
    "possibly_sitting":  0.50,
    "head_down":         16.0,
    "hand_face_close":   0.40,
    "hand_face_mid":     0.50,
    "arm_raised":       130.0,
    "arm_lowered":      155.0,
}


def _skeleton_to_semantic(
    body_axis_angle:    float,
    head_pitch:         float,
    hand_to_head:       float,
    left_hand_to_head:  float,
    knee_hip_ratio:     float,
    arm_elevation:      float,
    left_arm_elevation: float,
    t: dict = SKELETON_THRESHOLDS,
) -> str:
    hints = []

    if body_axis_angle >= 0:
        if body_axis_angle > t["body_tilted"]:
            hints.append("body is significantly tilted or near-horizontal")
        else:
            if knee_hip_ratio >= 0:
                if knee_hip_ratio > t["sitting_ratio"]:
                    hints.append("body is upright, person is sitting")
                elif knee_hip_ratio > t["possibly_sitting"]:
                    hints.append("body is upright, possibly seated")
                else:
                    hints.append("body is upright, person is standing")
            else:
                hints.append("body is upright")

    if head_pitch >= 0:
        if head_pitch > t["head_down"]:
            hints.append("head is looking down")
        else:
            hints.append("head is facing forward")

    best_h2h = -1.0
    if hand_to_head >= 0 and left_hand_to_head >= 0:
        best_h2h = min(hand_to_head, left_hand_to_head)
    elif hand_to_head >= 0:
        best_h2h = hand_to_head
    elif left_hand_to_head >= 0:
        best_h2h = left_hand_to_head

    if best_h2h >= 0:
        if best_h2h < t["hand_face_close"]:
            hints.append("hand is very close to face or mouth")
        elif best_h2h < t["hand_face_mid"]:
            hints.append("hand is near chest level")
        else:
            hints.append("hand is extended away from face")

    for side, elev in [("right", arm_elevation), ("left", left_arm_elevation)]:
        if elev < 0:
            continue
        if elev < t["arm_raised"]:
            hints.append(f"{side} arm is raised toward interaction level")
        elif elev > t["arm_lowered"]:
            hints.append(f"{side} arm is hanging down naturally")
        else:
            hints.append(f"{side} arm is at mid level")

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
            dx   = pos[0] - ux
            dz   = pos[1] - uz
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


def build_scene_text(
    user_pos,
    user_forward,
    room_name,
    db,
    user_id:            str   = "",
    virtual_hour:       float = None,
    tv_on:              bool  = None,
    held_event:         str   = None,
    held_age:           float = 0.0,
    body_axis_angle:    float = -1.0,
    head_pitch:         float = -1.0,
    hand_to_head:       float = -1.0,
    left_hand_to_head:  float = -1.0,
    knee_hip_ratio:     float = -1.0,
    arm_elevation:      float = -1.0,
    left_arm_elevation: float = -1.0,
    skel_body:          str   = None,
) -> dict:

    time_str = ""
    if virtual_hour is not None:
        try:
            h = float(virtual_hour)
            if h >= 0:
                if h < 6:       slot = "Dawn"
                elif h < 10:    slot = "Morning"
                elif h < 13:    slot = "Noon"
                elif h < 18:    slot = "Afternoon"
                elif h < 22:    slot = "Evening"
                else:           slot = "Night"
                time_str = f"{h:.0f}:00 ({slot})"
        except Exception:
            pass

    posture = _skeleton_to_semantic(
        body_axis_angle=body_axis_angle,
        head_pitch=head_pitch,
        hand_to_head=hand_to_head,
        left_hand_to_head=left_hand_to_head,
        knee_hip_ratio=knee_hip_ratio,
        arm_elevation=arm_elevation,
        left_arm_elevation=left_arm_elevation,
    )

    held_str = ""
    if held_event and held_event not in ("none", "unknown", ""):
        held_str = held_event

    facing   = _get_facing_target(user_pos, user_forward, db)
    tv_scene = None
    try:
        tv_scene = db.scene_snapshots.find_one(
            {"label": {"$in": ["tv", "television"]}}, {"pos": 1})
    except Exception:
        pass

    if tv_on is None:
        try:
            tv_doc = db.device_states.find_one({"label": "tv"})
            tv_on  = tv_doc.get("state", "off") == "on" if tv_doc else False
        except Exception:
            tv_on = False

    tv_state_str = "ON" if tv_on else "off"

    nearby_furniture_entries = []
    if user_pos:
        try:
            ux = float(user_pos.get("x", 0))
            uz = float(user_pos.get("z", 0))
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
                        db.dynamic_objects.find({"last_seen_on": label}, {"label": 1})
                    ]
                    tagged = []
                    for item in raw_contents:
                        cat = OBJECT_CATEGORIES.get(item.lower(), "")
                        tagged.append(f"{item} [{cat}]" if cat else item)
                    entry = f"{label} ({d:.1f}m away)"
                    if tagged:
                        entry += f", contains: {', '.join(tagged)}"
                    nearby_furniture_entries.append((d, entry))
            nearby_furniture_entries.sort(key=lambda x: x[0])
            nearby_furniture_entries = nearby_furniture_entries[:4]
        except Exception:
            pass

    tv_dist_str = ""
    if tv_scene and user_pos:
        try:
            tv_pos = tv_scene.get("pos", [])
            if len(tv_pos) >= 2:
                ux2     = float(user_pos.get("x", 0))
                uz2     = float(user_pos.get("z", 0))
                tv_dist = math.sqrt((ux2 - tv_pos[0])**2 + (uz2 - tv_pos[1])**2)
                tv_dist_str = f"{tv_dist:.1f}m" if tv_dist < 6.0 else ""
        except Exception:
            pass

    lines = ["=== Scene Graph ==="]
    lines.append(f"Room: {room_name or 'Unknown'}")
    if time_str:
        lines.append(f"Time: {time_str}")
    if posture:
        lines.append(f"Posture cues: {posture}")
    if held_str:
        lines.append(f"Object event: {held_str}")
    else:
        lines.append("No recent object pickups")
    if facing in ("tv", "television"):
        lines.append(f"Facing: {facing} (TV is {tv_state_str})")
    else:
        lines.append(f"Facing: {facing}")
    if nearby_furniture_entries:
        lines.append("Nearby furniture:")
        for _, entry in nearby_furniture_entries:
            lines.append(f"  - {entry}")
    if tv_dist_str:
        lines.append(f"TV: {tv_state_str}, {tv_dist_str} away")
    else:
        lines.append(f"TV: {tv_state_str}")
    lines.append("=== End Scene ===")

    return {
        "room":    room_name or "Unknown",
        "time":    time_str,
        "posture": posture,
        "facing":  facing,
        "tv_on":   bool(tv_on),
        "held":    held_str,
        "nearby":  [entry for _, entry in nearby_furniture_entries],
        "text":    "\n".join(lines),
    }