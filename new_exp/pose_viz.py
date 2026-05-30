import os
import cv2
import numpy as np
from ultralytics import YOLO
from collections import Counter

IMAGE_DIR  = "/home/wenny/db/robotBrain/debug_images"
OUTPUT_DIR = "/home/wenny/db/robotBrain/new_exp/pose_viz"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SKELETON = [
    (0,1),(0,2),(1,3),(2,4),
    (5,6),(5,7),(7,9),(6,8),(8,10),
    (5,11),(6,12),(11,12),
    (11,13),(13,15),(12,14),(14,16),
]

NOSE=0; L_EYE=1; R_EYE=2; L_EAR=3; R_EAR=4
L_SHOULDER=5; R_SHOULDER=6; L_ELBOW=7; R_ELBOW=8
L_WRIST=9; R_WRIST=10; L_HIP=11; R_HIP=12
L_KNEE=13; R_KNEE=14; L_ANKLE=15; R_ANKLE=16

HAND_HINT_COLOR = {
    "eating_or_drinking": (0,220,80),
    "phone_use":          (255,140,0),
    "cooking_or_typing":  (0,140,255),
    "reading":            (220,0,220),
    "opening":            (0,220,220),
    "resting":            (160,160,160),
    "sitting_relaxed":    (180,180,0),
    "unknown":            (100,100,100),
}

BODY_COLOR = {
    "standing":        (0,200,255),
    "bent_or_sitting": (0,165,255),
    "lying":           (255,100,0),
    "unknown":         (150,150,150),
}


def _angle(a, b, c):
    v1 = a - b
    v2 = c - b
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return 180.0
    return float(np.degrees(np.arccos(np.clip(np.dot(v1,v2)/(n1*n2),-1,1))))


def extract_features(kp, box):
    # ===================================================
    # 相機 10 度傾斜校正區區
    # ===================================================
    # 如果畫面「順時針」歪（人看起來往左倒），請用 -10.0 轉回來
    # 如果畫面「逆時針」歪（人看起來往右倒），請改用 10.0
    ANGLE_DEG = -10.0  
    angle_rad = np.radians(ANGLE_DEG)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    
    # 使用 Bounding Box 的中心點作為旋轉基準點
    cx = float(box[0] + box[2]) / 2.0
    cy = float(box[1] + box[3]) / 2.0
    
    # 複製一份關鍵點，並對有效座標進行逆向旋轉
    corrected_kp = np.copy(kp)
    for i in range(len(kp)):
        if kp[i][0] > 0 and kp[i][1] > 0:
            x_old = kp[i][0] - cx
            y_old = kp[i][1] - cy
            # 旋轉矩陣變換
            x_new = x_old * cos_a - y_old * sin_a + cx
            y_new = x_old * sin_a + y_old * cos_a + cy
            corrected_kp[i][0] = x_new
            corrected_kp[i][1] = y_new
    # ===================================================

    def pt(i):
        # 接下來所有特徵計算都採用校正後的座標 (corrected_kp)
        return np.array(corrected_kp[i][:2], dtype=float)

    bw = float(box[2] - box[0])
    bh = float(box[3] - box[1])
    aspect = bh / (bw + 1e-6)

    shoulder_mid = (pt(L_SHOULDER) + pt(R_SHOULDER)) / 2
    hip_mid      = (pt(L_HIP)      + pt(R_HIP))      / 2
    knee_mid     = (pt(L_KNEE)     + pt(R_KNEE))      / 2
    ankle_mid    = (pt(L_ANKLE)    + pt(R_ANKLE))     / 2
    wrist_mid    = (pt(L_WRIST)    + pt(R_WRIST))     / 2

    body_vec = hip_mid - shoulder_mid
    body_len = np.linalg.norm(body_vec)

    # 判斷點是否被偵測到，仍可用原本的 kp（因為不影響大於 0 的判斷）
    visible_ankles = (kp[L_ANKLE][0] > 0 and kp[L_ANKLE][1] > 0) or \
                     (kp[R_ANKLE][0] > 0 and kp[R_ANKLE][1] > 0)
    visible_knees  = (kp[L_KNEE][0] > 0 and kp[L_KNEE][1] > 0) or \
                     (kp[R_KNEE][0] > 0 and kp[R_KNEE][1] > 0)
    visible_hips   = (kp[L_HIP][0] > 0 and kp[L_HIP][1] > 0) or \
                     (kp[R_HIP][0] > 0 and kp[R_HIP][1] > 0)

    if body_len < 1:
        return None

    def nd(a, b):
        return np.linalg.norm(a - b) / body_len

    langle = _angle(pt(L_SHOULDER), pt(L_ELBOW), pt(L_WRIST))
    rangle = _angle(pt(R_SHOULDER), pt(R_ELBOW), pt(R_WRIST))

    wrist_below_hip = (
        pt(L_WRIST)[1] > hip_mid[1] or
        pt(R_WRIST)[1] > hip_mid[1]
    )

    head_forward = (shoulder_mid[1] - pt(NOSE)[1]) / body_len

    return {
        "aspect":              aspect,
        "body_len":            body_len,
        "body_horizontal":     abs(body_vec[0]) / (abs(body_vec[1]) + 1e-6),
        "knee_hip_ratio":      nd(knee_mid, hip_mid),
        "wrist_to_nose":       nd(wrist_mid, pt(NOSE)),
        "wrist_to_hip":        nd(wrist_mid, hip_mid),
        "wrist_to_shoulder":   nd(wrist_mid, shoulder_mid),
        "wrist_spread":        nd(pt(L_WRIST), pt(R_WRIST)),
        "shoulder_width":      nd(pt(L_SHOULDER), pt(R_SHOULDER)),
        "lwrist_above_sh":     (shoulder_mid[1] - pt(L_WRIST)[1]) / body_len,
        "rwrist_above_sh":     (shoulder_mid[1] - pt(R_WRIST)[1]) / body_len,
        "nose_y_rel_sh":       head_forward,
        "lelbow_angle":        langle,
        "relbow_angle":        rangle,
        "min_elbow_angle":     min(langle, rangle),
        "wrist_below_hip":     wrist_below_hip,
        "visible_ankles":      visible_ankles,
        "visible_knees":       visible_knees,
        "visible_hips":        visible_hips,
        "arm_extended":        nd(wrist_mid, shoulder_mid) > 0.75,
    }


def infer_pose(features):
    if features is None:
        return "unknown", "unknown", []

    f      = features
    hints  = []
    aspect = f["aspect"]

    if f["body_horizontal"] > 0.85 or aspect < 0.6:
        body_pos = "lying"
    elif aspect < 1.5 or (not f["visible_ankles"] and not f["visible_knees"]):
        body_pos = "bent_or_sitting"
    else:
        body_pos = "standing"

    wrist_near_face   = f["wrist_to_nose"] < 0.55
    wrist_at_chest    = 0.25 < f["wrist_to_hip"] < 0.90
    wrist_low         = f["wrist_to_hip"] < 0.40
    arms_bent         = f["min_elbow_angle"] < 130
    head_down         = f["nose_y_rel_sh"] < 0.10
    wrist_raised      = max(f["lwrist_above_sh"], f["rwrist_above_sh"]) > 0.15
    wrist_spread_wide = f["wrist_spread"] > 0.55
    arm_extended      = f["arm_extended"]
    wrist_below_hip   = f["wrist_below_hip"]

    if body_pos == "lying":
        hand_hint = "resting"
        hints.append("body horizontal -> resting")

    elif body_pos == "bent_or_sitting":
        if arm_extended and wrist_below_hip and not wrist_near_face:
            hand_hint = "opening"
            hints.append(f"arm_extended+wrist_below_hip -> opening")
        elif head_down and arm_extended and not wrist_near_face:
            hand_hint = "opening"
            hints.append(f"head_down+arm_extended -> opening")
        elif wrist_near_face and arms_bent:
            hand_hint = "eating_or_drinking"
            hints.append(f"wrist->face:{f['wrist_to_nose']:.2f} bent")
        elif wrist_at_chest and head_down and not wrist_near_face:
            hand_hint = "phone_use"
            hints.append(f"chest+head_down -> phone_use")
        elif wrist_at_chest and wrist_spread_wide:
            hand_hint = "reading"
            hints.append(f"chest+spread:{f['wrist_spread']:.2f} -> reading")
        else:
            hand_hint = "sitting_relaxed"
            hints.append("sitting, no clear hand action")

    else:
        if arm_extended and wrist_below_hip and aspect < 1.6:
            hand_hint = "opening"
            hints.append(f"standing+arm_down+aspect:{aspect:.2f} -> opening")
        elif wrist_near_face and arms_bent and wrist_raised:
            hand_hint = "eating_or_drinking"
            hints.append(f"wrist->face:{f['wrist_to_nose']:.2f} raised+bent")
        elif wrist_at_chest and arms_bent and head_down and not wrist_near_face:
            hand_hint = "phone_use"
            hints.append(f"chest:{f['wrist_to_hip']:.2f} head_down -> phone_use")
        elif wrist_low and wrist_spread_wide:
            hand_hint = "cooking_or_typing"
            hints.append(f"wrist_low:{f['wrist_to_hip']:.2f} spread:{f['wrist_spread']:.2f}")
        elif wrist_at_chest and wrist_spread_wide and not wrist_near_face:
            hand_hint = "reading"
            hints.append(f"chest+spread -> reading")
        elif not arms_bent and not wrist_raised:
            hand_hint = "resting"
            hints.append("arms relaxed")
        else:
            hand_hint = "unknown"

    return body_pos, hand_hint, hints


def draw_pose_on_image(img, kp, body_pos, hand_hint, hints, box, conf):
    # 繪製骨架依然使用 YOLO 原生輸出的 kp，讓標註貼合原始影像畫面
    for i, j in SKELETON:
        x1, y1 = int(kp[i][0]), int(kp[i][1])
        x2, y2 = int(kp[j][0]), int(kp[j][1])
        if x1 > 0 and y1 > 0 and x2 > 0 and y2 > 0:
            cv2.line(img, (x1,y1), (x2,y2), (0,220,0), 2, cv2.LINE_AA)

    for idx, (x, y) in enumerate(kp):
        x, y = int(x), int(y)
        if x > 0 and y > 0:
            cv2.circle(img, (x,y), 4, (0,120,255), -1, cv2.LINE_AA)

    b = box.astype(int)
    cv2.rectangle(img, (b[0],b[1]), (b[2],b[3]), (255,200,0), 2, cv2.LINE_AA)
    cv2.putText(img, f"Person {conf:.2f}", (b[0], b[1]-8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,200,0), 2, cv2.LINE_AA)

    pc = BODY_COLOR.get(body_pos, (150,150,150))
    hc = HAND_HINT_COLOR.get(hand_hint, (100,100,100))

    y0 = 10
    cv2.putText(img, f"Body: {body_pos}", (10, y0+16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, pc, 2, cv2.LINE_AA)
    cv2.putText(img, f"Hand: {hand_hint}", (10, y0+38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, hc, 2, cv2.LINE_AA)
    for k, h in enumerate(hints[:2]):
        cv2.putText(img, h, (10, y0+58+k*17),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, (200,200,200), 1, cv2.LINE_AA)
    return img


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


def main():
    print("Loading YOLOv8 pose model...")
    yolo = YOLO("yolov8n-pose.pt")

    files = sorted(f for f in os.listdir(IMAGE_DIR) if f.endswith(".jpg"))
    print(f"Processing {len(files)} images -> {OUTPUT_DIR}\n")

    body_counter = Counter()
    hand_counter = Counter()
    correct_body = {"standing":0,"bent_or_sitting":0,"lying":0}
    total_body   = {"standing":0,"bent_or_sitting":0,"lying":0}

    GT_BODY = {
        "Cooking":"standing","Drinking":"standing","Eating":"bent_or_sitting",
        "Opening":"bent_or_sitting","Laying":"lying","Reading":"bent_or_sitting",
        "PhoneUse":"standing","Typing":"bent_or_sitting","Watching":"bent_or_sitting",
        "SittingDrink":"bent_or_sitting","Cleaning":"standing",
    }

    for i, fname in enumerate(files):
        path = os.path.join(IMAGE_DIR, fname)
        info = parse_filename(fname)
        img  = cv2.imread(path)
        if img is None:
            continue

        results = yolo(img, verbose=False)
        persons = []

        for r in results:
            if r.keypoints is None:
                continue
            for j, kp in enumerate(r.keypoints.xy.cpu().numpy()):
                conf = float(r.boxes.conf[j]) if r.boxes is not None else 0.0
                if conf < 0.25:
                    continue
                box  = r.boxes.xyxy[j].cpu().numpy()
                feat = extract_features(kp, box)
                body_pos, hand_hint, hints = infer_pose(feat)
                persons.append({
                    "kp": kp, "box": box, "conf": conf,
                    "body_pos": body_pos, "hand_hint": hand_hint,
                    "hints": hints, "feat": feat,
                })

        persons.sort(key=lambda x: -x["conf"])

        for p in persons:
            draw_pose_on_image(img, p["kp"], p["body_pos"],
                               p["hand_hint"], p["hints"], p["box"], p["conf"])

        gt = info["ground_truth"]
        if persons:
            bp = persons[0]["body_pos"]
            hh = persons[0]["hand_hint"]
            body_counter[bp] += 1
            hand_counter[hh] += 1

            expected_body = GT_BODY.get(gt, "")
            if expected_body:
                total_body[expected_body] = total_body.get(expected_body, 0) + 1
                if bp == expected_body:
                    correct_body[expected_body] = correct_body.get(expected_body, 0) + 1

            info_text = (f"GT:{gt}  Room:{info['room_name']}  "
                         f"Cam:{info['cam_id']}  Body:{bp}  Hand:{hh}")
        else:
            info_text = (f"GT:{gt}  Room:{info['room_name']}  "
                         f"Cam:{info['cam_id']}  NO PERSON")

        h2, w2 = img.shape[:2]
        bar = np.zeros((50, w2, 3), dtype=np.uint8)
        bar[:] = (30,30,30)
        cv2.putText(bar, info_text, (8,33),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255,255,255), 1, cv2.LINE_AA)

        out      = np.vstack([img, bar])
        out_name = fname.replace(".jpg","_pose.jpg")
        cv2.imwrite(os.path.join(OUTPUT_DIR, out_name), out)

        bp_str = persons[0]["body_pos"] if persons else "NO_PERSON"
        hh_str = persons[0]["hand_hint"] if persons else "-"
        match  = ""
        if persons and gt in GT_BODY:
            match = "✓" if bp_str == GT_BODY[gt] else "✗"
        print(f"[{i+1:02d}/{len(files)}] {fname}  body={bp_str} {match}  hand={hh_str}")

    print(f"\nDone -> {OUTPUT_DIR}")
    print("\nBody position distribution:")
    for k, v in body_counter.most_common():
        print(f"  {k:20}: {v}")
    print("\nHand hint distribution:")
    for k, v in hand_counter.most_common():
        print(f"  {k:25}: {v}")
    print("\nBody position accuracy (vs expected from GT):")
    for bp in ["standing","bent_or_sitting","lying"]:
        t = total_body.get(bp,0)
        c = correct_body.get(bp,0)
        acc = c/t*100 if t > 0 else 0
        print(f"  {bp:20}: {acc:5.1f}% ({c}/{t})")
    print(f"\nView: eog {OUTPUT_DIR}/*.jpg")


if __name__ == "__main__":
    main()