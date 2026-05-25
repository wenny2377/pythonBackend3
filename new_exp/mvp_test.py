import os
import json
import math
import requests
from collections import defaultdict, Counter

import cv2
import numpy as np
from pymongo import MongoClient
from ultralytics import YOLO

MONGO_URI  = "mongodb://127.0.0.1:27017/"
DB_NAME    = "robot_rag_db"
OLLAMA_URL = "http://127.0.0.1:11434"
LLM_MODEL  = "llama3.1:8b-instruct-q4_K_M"
IMAGE_DIR  = "/home/wenny/db/robotBrain/debug_images"

NEARBY_RADIUS    = 2.5
INTERACTION_DIST = 1.5
HIGH_SPEC_DELTA  = 0.40

BEHAVIOR_LABELS = [
    "Eating","Drinking","SittingDrink","Cooking","Opening",
    "Laying","Watching","Reading","Cleaning","PhoneUse","Typing",
    "Standing","Walking",
]

STRUCTURAL_BLACKLIST = {
    "wall","floor","ceiling","wooden floor","white wall",
    "window","door","ground","concrete floor","tile floor",
    "carpet","baseboard",
}

NOSE=0; L_EYE=1; R_EYE=2; L_EAR=3; R_EAR=4
L_SHOULDER=5; R_SHOULDER=6; L_ELBOW=7; R_ELBOW=8
L_WRIST=9; R_WRIST=10; L_HIP=11; R_HIP=12
L_KNEE=13; R_KNEE=14; L_ANKLE=15; R_ANKLE=16

ROOM_NAME_MAP = {
    "kitchen":    "Kitchen",
    "livingroom": "LivingRoom",
    "dadroom":    "BedRoom(dad)",
}

OBJECT_TO_ACTION = {
    "cup":"Drinking","bottle":"Drinking","mug":"Drinking",
    "glass":"Drinking","juice":"Drinking","cola":"Drinking",
    "fork":"Eating","spoon":"Eating","bowl":"Eating",
    "plate":"Eating","food":"Eating","apple":"Eating",
    "banana":"Eating","saladbowl":"Eating",
    "pan":"Cooking","pot":"Cooking","spatula":"Cooking",
    "book":"Reading","magazine":"Reading",
    "phone":"PhoneUse","smartphone":"PhoneUse",
    "laptop":"Typing","keyboard":"Typing",
    "broom":"Cleaning","mop":"Cleaning",
    "remote":"Watching",
}

HAND_HINT_ACTIONS = {
    "eating_or_drinking": ["Eating","Drinking","SittingDrink"],
    "phone_use":          ["PhoneUse"],
    "cooking_or_typing":  ["Cooking","Typing"],
    "reading":            ["Reading"],
    "opening":            ["Opening"],
    "resting":            ["Laying"],
    "watching_relaxed":   ["Watching","SittingDrink"],
    "sitting_relaxed":    ["Watching","SittingDrink","Eating"],
}

FRIDGE_FURNITURE = {"refrigerator","fridge","cabinet","cabinet2"}


def connect():
    return MongoClient(MONGO_URI)[DB_NAME]


def parse_filename(fname):
    base  = os.path.basename(fname).replace(".jpg","")
    parts = base.split("_")
    return {
        "timestamp":    parts[0],
        "user_id":      "_".join(parts[1:3]),
        "ground_truth": parts[3],
        "room_name":    parts[4],
        "cam_id":       parts[5],
        "burst_idx":    parts[6],
    }


def group_bursts(image_dir):
    files  = sorted(f for f in os.listdir(image_dir) if f.endswith(".jpg"))
    groups = defaultdict(list)
    for f in files:
        info = parse_filename(f)
        key  = f"{info['timestamp']}_{info['user_id']}_{info['ground_truth']}_{info['room_name']}"
        groups[key].append({**info, "path": os.path.join(image_dir, f)})
    return dict(groups)


def _angle(a, b, c):
    v1 = a - b
    v2 = c - b
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return 180.0
    return float(np.degrees(np.arccos(np.clip(np.dot(v1,v2)/(n1*n2),-1,1))))


def extract_features(kp, kp_conf, box):
    def pt(i):
        return np.array(kp[i][:2], dtype=float)

    def conf(i):
        return float(kp_conf[i]) if i < len(kp_conf) else 0.0

    bw = float(box[2]-box[0])
    bh = float(box[3]-box[1])
    aspect = bh / (bw + 1e-6)

    shoulder_mid = (pt(L_SHOULDER)+pt(R_SHOULDER))/2
    hip_mid      = (pt(L_HIP)+pt(R_HIP))/2
    knee_mid     = (pt(L_KNEE)+pt(R_KNEE))/2
    ankle_mid    = (pt(L_ANKLE)+pt(R_ANKLE))/2
    wrist_mid    = (pt(L_WRIST)+pt(R_WRIST))/2

    body_vec = hip_mid - shoulder_mid
    body_len = np.linalg.norm(body_vec)
    if body_len < 1:
        return None

    def nd(a, b):
        return np.linalg.norm(a-b)/body_len

    langle = _angle(pt(L_SHOULDER), pt(L_ELBOW), pt(L_WRIST))
    rangle = _angle(pt(R_SHOULDER), pt(R_ELBOW), pt(R_WRIST))

    knee_conf_avg  = (conf(L_KNEE) + conf(R_KNEE)) / 2
    ankle_conf_avg = (conf(L_ANKLE) + conf(R_ANKLE)) / 2
    hip_conf_avg   = (conf(L_HIP) + conf(R_HIP)) / 2
    hip_sh_ratio   = abs(hip_mid[1]-shoulder_mid[1]) / (bh + 1e-6)

    lwrist_above = (shoulder_mid[1]-pt(L_WRIST)[1])/body_len
    rwrist_above = (shoulder_mid[1]-pt(R_WRIST)[1])/body_len
    dominant_wrist_above = max(lwrist_above, rwrist_above)
    weaker_wrist_above   = min(lwrist_above, rwrist_above)
    wrist_height_asym    = abs(pt(L_WRIST)[1]-pt(R_WRIST)[1]) / (body_len + 1e-6)

    return {
        "aspect":               aspect,
        "body_len":             body_len,
        "body_horizontal":      abs(body_vec[0])/(abs(body_vec[1])+1e-6),
        "knee_hip_ratio":       nd(knee_mid, hip_mid),
        "wrist_to_nose":        nd(wrist_mid, pt(NOSE)),
        "lwrist_to_nose":       nd(pt(L_WRIST), pt(NOSE)),
        "rwrist_to_nose":       nd(pt(R_WRIST), pt(NOSE)),
        "wrist_to_hip":         nd(wrist_mid, hip_mid),
        "wrist_to_shoulder":    nd(wrist_mid, shoulder_mid),
        "wrist_spread":         nd(pt(L_WRIST), pt(R_WRIST)),
        "dominant_wrist_above": dominant_wrist_above,
        "weaker_wrist_above":   weaker_wrist_above,
        "wrist_height_asym":    wrist_height_asym,
        "nose_y_rel_sh":        (shoulder_mid[1]-pt(NOSE)[1])/body_len,
        "lelbow_angle":         langle,
        "relbow_angle":         rangle,
        "min_elbow_angle":      min(langle, rangle),
        "wrist_below_hip":      (pt(L_WRIST)[1]>hip_mid[1] or pt(R_WRIST)[1]>hip_mid[1]),
        "arm_extended":         nd(wrist_mid, shoulder_mid) > 0.75,
        "knee_conf":            knee_conf_avg,
        "ankle_conf":           ankle_conf_avg,
        "hip_conf":             hip_conf_avg,
        "hip_sh_ratio":         hip_sh_ratio,
    }


def infer_pose(features):
    if features is None:
        return "unknown", "unknown", []

    f      = features
    hints  = []
    aspect = f["aspect"]

    lower_body_hidden = (f["knee_conf"] < 0.35 and f["ankle_conf"] < 0.35)
    compact_upper     = f["hip_sh_ratio"] < 0.28

    if f["body_horizontal"] > 0.85 or aspect < 0.6:
        body_pos = "lying"
    elif lower_body_hidden or compact_upper:
        body_pos = "bent_or_sitting"
        hints.append(
            f"knee_conf:{f['knee_conf']:.2f} ankle:{f['ankle_conf']:.2f} "
            f"hip_sh:{f['hip_sh_ratio']:.2f} -> sitting")
    elif aspect < 1.5:
        body_pos = "bent_or_sitting"
        hints.append(f"aspect:{aspect:.2f} -> sitting")
    else:
        body_pos = "standing"

    wrist_near_face   = f["wrist_to_nose"] < 0.55
    min_wrist_nose    = min(f["lwrist_to_nose"], f["rwrist_to_nose"])
    one_wrist_at_face = min_wrist_nose < 0.45
    wrist_at_chest    = 0.25 < f["wrist_to_hip"] < 0.90
    wrist_low         = f["wrist_to_hip"] < 0.40
    arms_bent         = f["min_elbow_angle"] < 130
    head_down         = f["nose_y_rel_sh"] < 0.10
    dominant_raised   = f["dominant_wrist_above"] > 0.15
    weaker_raised     = f["weaker_wrist_above"] > 0.10
    one_hand_up       = dominant_raised and not weaker_raised
    wrist_spread_wide = f["wrist_spread"] > 0.55
    high_asym         = f["wrist_height_asym"] > 0.30
    arm_extended      = f["arm_extended"]
    wrist_below_hip   = f["wrist_below_hip"]

    if body_pos == "lying":
        hand_hint = "resting"

    elif body_pos == "bent_or_sitting":
        if arm_extended and wrist_below_hip and not wrist_near_face:
            hand_hint = "opening"
            hints.append("arm_extended+wrist_below -> opening")
        elif head_down and arm_extended and not wrist_near_face:
            hand_hint = "opening"
            hints.append("head_down+arm_extended -> opening")
        elif one_wrist_at_face and arms_bent:
            hand_hint = "eating_or_drinking"
            hints.append(f"one_wrist->face:{min_wrist_nose:.2f} -> eat/drink")
        elif wrist_at_chest and head_down and high_asym and not wrist_near_face:
            hand_hint = "phone_use"
            hints.append(f"chest+head_down+asym:{f['wrist_height_asym']:.2f} -> phone")
        elif wrist_at_chest and wrist_spread_wide and not wrist_near_face and not high_asym:
            hand_hint = "reading"
            hints.append(f"chest+spread:{f['wrist_spread']:.2f}+sym -> reading")
        elif not dominant_raised and not wrist_near_face:
            hand_hint = "watching_relaxed"
            hints.append("bent+arms_low -> watching_relaxed")
        else:
            hand_hint = "sitting_relaxed"
            hints.append("sitting, no clear action")

    else:  # standing
        if arm_extended and wrist_below_hip and aspect < 1.6:
            hand_hint = "opening"
            hints.append(f"arm_down+aspect:{aspect:.2f} -> opening")
        elif one_wrist_at_face and arms_bent and dominant_raised:
            hand_hint = "eating_or_drinking"
            hints.append(f"one_wrist->face:{min_wrist_nose:.2f} raised -> eat/drink")
        elif wrist_near_face and arms_bent and dominant_raised:
            hand_hint = "eating_or_drinking"
            hints.append(f"wrist->face:{f['wrist_to_nose']:.2f} -> eat/drink")
        elif one_hand_up and high_asym and not wrist_near_face and wrist_at_chest:
            hand_hint = "phone_use"
            hints.append(f"one_hand_up+asym:{f['wrist_height_asym']:.2f} -> phone")
        elif wrist_at_chest and arms_bent and head_down and high_asym and not wrist_near_face:
            hand_hint = "phone_use"
            hints.append(f"chest+head_down+asym -> phone")
        elif wrist_low and wrist_spread_wide and not high_asym:
            hand_hint = "cooking_or_typing"
            hints.append(f"wrist_low:{f['wrist_to_hip']:.2f} spread:{f['wrist_spread']:.2f}")
        elif wrist_at_chest and wrist_spread_wide and not wrist_near_face and not high_asym:
            hand_hint = "reading"
            hints.append("chest+spread+sym -> reading")
        elif not arms_bent and not dominant_raised:
            hand_hint = "resting"
            hints.append("arms relaxed")
        else:
            hand_hint = "unknown"

    return body_pos, hand_hint, hints


def run_yolo(image_path, yolo_model):
    img     = cv2.imread(image_path)
    results = yolo_model(img, verbose=False)
    persons = []
    for r in results:
        if r.keypoints is None:
            continue
        kps      = r.keypoints.xy.cpu().numpy()
        kps_conf = r.keypoints.conf.cpu().numpy() if r.keypoints.conf is not None \
                   else np.ones((len(kps), 17))
        for j, kp in enumerate(kps):
            conf = float(r.boxes.conf[j]) if r.boxes is not None else 0.0
            if conf < 0.25:
                continue
            box  = r.boxes.xyxy[j].cpu().numpy()
            kc   = kps_conf[j]
            feat = extract_features(kp, kc, box)
            body_pos, hand_hint, hints = infer_pose(feat)
            persons.append({
                "conf": conf, "body_pos": body_pos,
                "hand_hint": hand_hint, "hints": hints,
                "feat": feat,
            })
    persons.sort(key=lambda x: -x["conf"])
    return persons


def get_db_room_name(room_name_raw):
    return ROOM_NAME_MAP.get(room_name_raw.lower().strip(), room_name_raw)


def get_furniture_in_room(db, db_room_name):
    room_tgt = db_room_name.strip()
    docs = list(db.scene_snapshots.find(
        {"room": {"$regex": f"^{room_tgt}$", "$options": "i"}},
        {"label": 1, "pos": 1}
    ))
    return docs


def get_objects_in_room(db, db_room_name):
    room_tgt = db_room_name.strip()
    docs = list(db.dynamic_objects.find(
        {"room": {"$regex": f"^{room_tgt}$", "$options": "i"}},
        {"label": 1}
    ))
    return [d.get("label", "").lower().strip() for d in docs
            if d.get("label", "").lower().strip() not in STRUCTURAL_BLACKLIST]


def get_top_affinity_per_furniture(db, furniture_docs):
    result = {}
    for doc in furniture_docs:
        label = doc.get("label","").lower().strip()
        aff   = list(db.affinity_matrix.find(
            {"furniture": label}, {"behavior":1,"score":1}))
        if aff:
            best = max(aff, key=lambda x: x.get("score",0))
            result[label] = {
                "top_action": best.get("behavior"),
                "top_score":  best.get("score",0),
            }
    return result


def rule_based_inference(body_pos, hand_hint, nearby_objects,
                          furniture_affinity,
                          has_fridge, has_stove, has_bed, has_tv, has_desk):
    scores = defaultdict(float)

    if body_pos == "lying":
        scores["Laying"] += 3.0

    if hand_hint == "opening" and has_fridge:
        scores["Opening"] += 3.0
    elif hand_hint == "opening":
        scores["Opening"] += 1.5

    if hand_hint == "eating_or_drinking":
        scores["Eating"] += 1.5
        scores["Drinking"] += 1.5
        if any(o in ["cup", "bottle", "mug", "glass"] for o in nearby_objects):
            scores["Drinking"] += 0.5
        if any(o in ["fork", "spoon", "bowl", "plate", "food"] for o in nearby_objects):
            scores["Eating"] += 0.5

    if hand_hint == "phone_use":
        scores["PhoneUse"] += 2.5

    if hand_hint == "reading":
        scores["Reading"] += 2.5

    if hand_hint == "cooking_or_typing":
        if has_stove:
            scores["Cooking"] += 2.5
        if has_desk:
            scores["Typing"] += 2.5

    if hand_hint in ("watching_relaxed", "sitting_relaxed") and has_tv:
        scores["Watching"] += 2.0

    if hand_hint == "unknown":
        if body_pos == "standing":
            if has_stove:
                scores["Cooking"] += 0.8
            else:
                scores["Standing"] += 0.5
        elif body_pos == "bent_or_sitting":
            if has_tv:
                scores["Watching"] += 0.8
            if has_desk:
                scores["Typing"] += 0.8
            scores["Watching"] += 0.2

    if not scores:
        return None, "no_score", 0.0

    best   = max(scores, key=scores.get)
    score  = scores[best]
    reason = (f"rule:{body_pos}+{hand_hint}+"
              f"stove={has_stove},fridge={has_fridge},"
              f"tv={has_tv},desk={has_desk}")
    return best, reason, score


def llm_infer(evidence: dict) -> tuple:
    behavior_list = ", ".join(BEHAVIOR_LABELS)

    prompt = (
        f"You are a smart home activity recognition system.\n"
        f"Analyze the posture and environment evidence to predict the most logical user activity.\n\n"
        f"Evidence Data:\n{json.dumps(evidence, indent=2)}\n\n"
        f"Output EXACTLY in this format:\n"
        f"ACTION: <one activity from allowed list>\n"
        f"REASON: <one brief sentence explaining why>\n\n"
        f"Allowed Activities: {behavior_list}\n\n"
        f"Common Sense Rules:\n"
        f"- If body_position is 'lying', it's almost always Laying.\n"
        f"- If hand_hint is 'unknown', look carefully at the room and available furniture, but do not blind guess Eating.\n"
        f"- If hand_hint matches an action category, prioritize it unless location makes it impossible.\n"
    )

    try:
        resp = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model":   LLM_MODEL,
                "prompt":  prompt,
                "stream":  False,
                "options": {"temperature":0.05,"num_predict":80},
            },
            timeout=30,
        )
        raw    = resp.json().get("response","").strip()
        action = None
        reason = raw

        for line in raw.split("\n"):
            line = line.strip()
            if line.startswith("ACTION:"):
                action = line.replace("ACTION:","").strip()
            elif line.startswith("REASON:"):
                reason = line.replace("REASON:","").strip()

        if action and action in BEHAVIOR_LABELS:
            return action, reason
        for b in BEHAVIOR_LABELS:
            if b.lower() in raw.lower():
                return b, raw
        return None, raw

    except Exception as e:
        return None, f"LLM error: {e}"


def process_group(group_key, images, db, yolo_model):
    info         = images[0]
    ground_truth = info["ground_truth"]
    room_name    = info["room_name"]
    db_room      = get_db_room_name(room_name)

    all_persons = []
    for img_info in images:
        persons = run_yolo(img_info["path"], yolo_model)
        all_persons.extend(persons)

    if all_persons:
        all_persons.sort(key=lambda x: -x["conf"])
        best       = all_persons[0]
        body_pos   = best["body_pos"]
        hand_hint  = best["hand_hint"]
        pose_hints = best["hints"]

        body_votes = Counter(p["body_pos"] for p in all_persons)
        hand_votes = Counter(p["hand_hint"] for p in all_persons)
        body_pos   = body_votes.most_common(1)[0][0]
        hand_hint  = hand_votes.most_common(1)[0][0]
    else:
        body_pos   = "unknown"
        hand_hint  = "unknown"
        pose_hints = []

    furniture_docs     = get_furniture_in_room(db, db_room)
    furniture_affinity = get_top_affinity_per_furniture(db, furniture_docs)
    nearby_objects     = get_objects_in_room(db, db_room)

    # === 【環境常識注入修正點】補足 MongoDB 遺失的關鍵環境空間錨點 ===
    room_lower = room_name.lower().strip()
    has_stove  = False
    has_fridge = False
    has_bed    = False
    has_tv     = False
    has_desk   = False

    if "kitchen" in room_lower:
        has_stove  = True
        has_fridge = True
    elif "living" in room_lower:
        has_tv     = True
    elif "dad" in room_lower or "bedroom" in room_lower:
        has_bed    = True
        has_desk   = True

    # 若 DB 有查到依然保留作為聯集補充
    furniture_labels = set(furniture_affinity.keys())
    if "stove" in furniture_labels: has_stove = True
    if bool(furniture_labels & FRIDGE_FURNITURE): has_fridge = True
    if bool(furniture_labels & {"bed","dad's bed"}): has_bed = True
    if bool(furniture_labels & {"tv","television"}): has_tv = True
    if bool(furniture_labels & {"desk","keyboard","monitor"}): has_desk = True

    top_furnitures = sorted(
        furniture_affinity.items(),
        key=lambda x: x[1]["top_score"], reverse=True)[:3]

    rule_action, rule_reason, rule_score = rule_based_inference(
        body_pos, hand_hint, nearby_objects,
        furniture_affinity, has_fridge, has_stove, has_bed, has_tv, has_desk,
    )

    evidence = {
        "room":           room_name,
        "body_position":  body_pos,
        "hand_hint":      hand_hint,
        "pose_hints":     pose_hints,
        "nearby_objects": nearby_objects[:6],
        "top_furniture":  {k: v["top_action"] for k,v in top_furnitures},
        "has_stove":      has_stove,
        "has_fridge":     has_fridge,
        "has_bed":        has_bed,
        "has_tv":         has_tv,
        "has_desk":       has_desk,
        "rule_suggested_action": rule_action,
        "rule_confidence":       rule_score,
    }

    predicted, reason = llm_infer(evidence)
    correct = (predicted == ground_truth)

    return {
        "group":          group_key,
        "ground_truth":   ground_truth,
        "predicted":      predicted,
        "correct":        correct,
        "body_position":  body_pos,
        "hand_hint":      hand_hint,
        "rule_action":    rule_action,
        "nearby_objects": nearby_objects[:3],
        "reason":         reason,
    }


def main():
    print("Loading YOLOv8 pose model...")
    yolo_model = YOLO("yolov8n-pose.pt")

    print("Connecting to MongoDB...")
    db = connect()

    print(f"Scanning {IMAGE_DIR}...")
    groups = group_bursts(IMAGE_DIR)
    print(f"Found {len(groups)} groups\n")

    print(f"{'GT':15} {'Rule':15} {'Pred':15} {'Body':18} {'Hand':25} OK")
    print("-"*100)

    results = []
    correct = 0
    total   = 0

    for key, images in sorted(groups.items()):
        result = process_group(key, images, db, yolo_model)
        results.append(result)
        total += 1
        if result["correct"]:
            correct += 1

        ok   = "✓" if result["correct"] else "✗"
        rule = str(result["rule_action"] or "-")
        pred = str(result["predicted"] or "-")
        print(f"{ok} {result['ground_truth']:14} {rule:14} {pred:14} "
              f"{result['body_position']:17} {result['hand_hint']:24}")

        if not result["correct"]:
            if result["nearby_objects"]:
                print(f"    nearby : {result['nearby_objects']}")
            print(f"    reason : {str(result['reason'])[:90]}")

    print(f"\n{'='*60}")
    print(f"Total: {total}  Correct: {correct}  Accuracy: {correct/total*100:.1f}%")

    by_class = defaultdict(lambda: {"total":0,"correct":0})
    for r in results:
        gt = r["ground_truth"]
        by_class[gt]["total"] += 1
        if r["correct"]:
            by_class[gt]["correct"] += 1

    print("\nPer-class accuracy:")
    for gt, stat in sorted(by_class.items()):
        acc = stat["correct"]/stat["total"]*100
        bar = "█" * int(acc/5)
        print(f"  {gt:15}: {acc:5.1f}% ({stat['correct']}/{stat['total']}) {bar}")

    rule_correct = sum(1 for r in results if r["rule_action"] == r["ground_truth"])
    rule_total   = len(results)
    print(f"\nRule-based accuracy: {rule_correct}/{rule_total} = {rule_correct/rule_total*100:.1f}%")

    print("\nBody position distribution in results:")
    body_counter = Counter(r["body_position"] for r in results)
    for k, v in body_counter.most_common():
        print(f"  {k:22}: {v}")

    print("\nHand hint distribution in results:")
    hand_counter = Counter(r["hand_hint"] for r in results)
    for k, v in hand_counter.most_common():
        print(f"  {k:25}: {v}")


if __name__ == "__main__":
    main()