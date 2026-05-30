import os
import json
import base64
import requests
from collections import defaultdict, Counter
import cv2
from ultralytics import YOLO

OLLAMA_URL = "http://127.0.0.1:11434"
VLM_MODEL  = "gemma3:4b"
IMAGE_DIR  = "/home/wenny/db/robotBrain/debug_images"
OUTPUT_DIR = "./visual_debug_results"

ROOM_NAME_MAP = {
    "kitchen":    "Kitchen",
    "livingroom": "LivingRoom",
    "dadroom":    "BedRoom(dad)",
}

ROOM_LABELS = {
    "bedroom": ["Laying", "Watching", "Reading", "Typing", "PhoneUse"],
    "dad":     ["Laying", "Watching", "Reading", "Typing", "PhoneUse"],
    "kitchen": ["Cooking", "Opening", "Eating", "Drinking", "Cleaning"],
    "livingroom": ["Eating", "Drinking", "SittingDrink", "Laying",
                   "Watching", "Reading", "PhoneUse", "Cleaning"],
}

yolo_model = YOLO("yolov8n-pose.pt")


def parse_filename(fname):
    base  = os.path.basename(fname).replace(".jpg", "").replace(".png", "")
    parts = base.split("_")
    room_raw = parts[4] if len(parts) > 4 else ""
    return {
        "timestamp":    parts[0] if len(parts) > 0 else "",
        "user_id":      parts[2] if len(parts) > 2 else "",
        "ground_truth": parts[3] if len(parts) > 3 else "",
        "room_name":    ROOM_NAME_MAP.get(room_raw.lower().strip(), room_raw),
        "cam_id":       parts[5] if len(parts) > 5 else "",
        "burst_idx":    parts[6] if len(parts) > 6 else "",
    }


def group_bursts(image_dir):
    files  = sorted(f for f in os.listdir(image_dir)
                    if f.endswith(".jpg") or f.endswith(".png"))
    groups = defaultdict(list)
    for f in files:
        info = parse_filename(f)
        key  = (f"{info['timestamp']}_{info['user_id']}_"
                f"{info['ground_truth']}_{info['room_name']}")
        groups[key].append({**info, "path": os.path.join(image_dir, f)})
    return dict(groups)


def get_allowed_labels(room_name):
    room_lower = room_name.lower()
    for key, labels in ROOM_LABELS.items():
        if key in room_lower:
            return labels
    return ["Eating", "Drinking", "SittingDrink", "Watching",
            "Laying", "Reading", "PhoneUse", "Cleaning"]


def crop_upper_body(image_path):
    try:
        img = cv2.imread(image_path)
        if img is None:
            return image_path, False

        h, w = img.shape[:2]
        results = yolo_model(img, verbose=False)

        if not results or results[0].keypoints is None:
            return image_path, False

        kps = results[0].keypoints.data
        if kps is None or len(kps) == 0:
            return image_path, False

        kp = kps[0].cpu().numpy()

        points = []
        for idx in [0, 5, 6, 7, 8, 9, 10]:
            if idx < len(kp) and kp[idx][2] > 0.3:
                points.append((int(kp[idx][0]), int(kp[idx][1])))

        if len(points) < 2:
            if results[0].boxes is not None and len(results[0].boxes) > 0:
                box   = results[0].boxes[0].xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = map(int, box)
                mid_y = (y1 + y2) // 2
                crop  = img[y1:mid_y, x1:x2]
                if crop.size > 0:
                    out = image_path.replace(".jpg", "_crop.jpg").replace(".png", "_crop.png")
                    cv2.imwrite(out, crop)
                    return out, True
            return image_path, False

        xs  = [p[0] for p in points]
        ys  = [p[1] for p in points]
        pad = 40
        x1  = max(0, min(xs) - pad)
        y1  = max(0, min(ys) - pad)
        x2  = min(w, max(xs) + pad)
        y2  = min(h, max(ys) + pad)

        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            return image_path, False

        out = image_path.replace(".jpg", "_crop.jpg").replace(".png", "_crop.png")
        cv2.imwrite(out, crop)
        return out, True

    except Exception as e:
        print(f"[Crop] {e}")
        return image_path, False


def vlm_infer(image_path, room_name, allowed_labels, is_cropped):
    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode()

    allowed_str = "/".join(allowed_labels)
    focus       = "upper body and hands" if is_cropped else "person"

    prompt = (
        f"Image shows a {focus} in {room_name}.\n"
        f"Step 1: What object is the person holding or interacting with?\n"
        f"Step 2: Based on that object and head angle, choose ONE action.\n"
        f"- PhoneUse/Reading: head tilted down, eyes looking down\n"
        f"- Drinking: head tilted back, chin raised\n"
        f"- Watching: head level, eyes forward\n"
        f"- Cooking: at stove, holding pan or spatula\n"
        f"- Eating: bringing food to mouth\n"
        f"Answer with exactly ONE word from: {allowed_str}\n"
        f"Action:"
    )

    try:
        resp = requests.post(
            f"{OLLAMA_URL}/api/chat",
            json={
                "model":    VLM_MODEL,
                "messages": [{"role": "user", "content": prompt,
                               "images": [img_b64]}],
                "stream":   False,
                "options":  {"temperature": 0.0, "num_predict": 20},
            },
            timeout=60,
        )
        raw = resp.json().get("message", {}).get("content", "").strip()
        for label in allowed_labels:
            if label.lower() in raw.lower():
                return label, raw
        return allowed_labels[0], raw
    except Exception as e:
        return allowed_labels[0], f"error: {e}"


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    groups = group_bursts(IMAGE_DIR)

    print(f"YOLO Upper-Body + VLM Pipeline | model={VLM_MODEL}")
    print(f"{'GT':15} {'Pred':18} {'Room':15} OK")
    print("-" * 60)

    correct = total = 0

    for key, images in sorted(groups.items()):
        gt      = images[0]["ground_truth"]
        room    = images[0]["room_name"]
        allowed = get_allowed_labels(room)
        preds   = []

        for info in images:
            crop_path, is_cropped = crop_upper_body(info["path"])
            pred, raw = vlm_infer(crop_path, room, allowed, is_cropped)
            if crop_path != info["path"] and os.path.exists(crop_path):
                os.remove(crop_path)
            preds.append(pred)

        final_pred = Counter(preds).most_common(1)[0][0] if preds else allowed[0]
        total += 1
        ok     = final_pred.lower() == gt.lower()
        if ok:
            correct += 1
        marker = "✓" if ok else "✗"
        print(f"{marker} {gt:14} {final_pred:17} {room:14}")

        rep = cv2.imread(images[0]["path"])
        if rep is not None:
            color = (0, 255, 0) if ok else (0, 0, 255)
            cv2.putText(rep, f"GT:{gt} PRED:{final_pred}",
                        (10, 35), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, color, 2, cv2.LINE_AA)
            cv2.imwrite(os.path.join(OUTPUT_DIR,
                f"{'OK' if ok else 'NG'}_{os.path.basename(images[0]['path'])}"), rep)

    acc = correct / total * 100 if total else 0
    print(f"\n{'='*50}")
    print(f"Accuracy: {correct}/{total} = {acc:.1f}%")


if __name__ == "__main__":
    main()