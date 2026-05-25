import os
import json
import base64
import requests
from collections import defaultdict, Counter

# === 系統配置 ===
OLLAMA_URL = "http://127.0.0.1:11434"
VLM_MODEL  = "llava-phi3" 
IMAGE_DIR  = "/home/wenny/db/robotBrain/debug_images"
OUTPUT_DIR = "./visual_debug_results"

# 嚴格對齊你的 Unity 標籤清單
BEHAVIOR_LABELS = [
    "Eating", "Drinking", "SittingDrink", "Cooking", "Opening",
    "Laying", "Watching", "Reading", "Cleaning", "PhoneUse", "Typing"
]

ROOM_NAME_MAP = {
    "kitchen":    "Kitchen",
    "livingroom": "LivingRoom",
    "dadroom":    "BedRoom(dad)",
}

def parse_filename(fname):
    base  = os.path.basename(fname).replace(".jpg","").replace(".png","")
    parts = base.split("_")
    return {
        "timestamp":    parts[0],
        "user_id":      parts[2], # Dad / Mom
        "ground_truth": parts[3],
        "room_name":    ROOM_NAME_MAP.get(parts[4].lower().strip(), parts[4]), 
        "cam_id":       parts[5],
        "burst_idx":    parts[6],
    }

def group_bursts(image_dir):
    files  = sorted(f for f in os.listdir(image_dir) if f.endswith(".jpg") or f.endswith(".png"))
    groups = defaultdict(list)
    for f in files:
        info = parse_filename(f)
        key  = f"{info['timestamp']}_{info['user_id']}_{info['ground_truth']}_{info['room_name']}"
        groups[key].append({**info, "path": os.path.join(image_dir, f)})
    return dict(groups)

def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def vlm_direct_infer(image_path, room, person):
    """
    直問 VLM 核心感知大腦。
    針對俯視角（Top-down）、Unity 標籤錯位（DadRoom Phone 汙染）進行強規則 Prompt Constraint。
    """
    img_base64 = encode_image_to_base64(image_path)
    allowed_list = ", ".join(BEHAVIOR_LABELS)
    
    prompt = (
        f"You are an expert activity recognition system looking through a top-down surveillance camera.\n"
        f"Analyze the image and determine what exact action the target person is doing right now.\n\n"
        f"Context:\n"
        f"- Target Person: {person}\n"
        f"- Current Location Room: {room}\n\n"
        f"Strict Dataset Rules for Perspective & Label Alignment:\n"
        f"1. LAYING DOWN: If the person is elongated horizontally on a couch or bed, the action is ALWAYS 'Laying', even if they hold an object.\n"
        f"2. DRINKING/EATING: Look closely at the hands. If holding a bottle/cup to the mouth, it is 'Drinking'. If holding a bowl/plate/food, it is 'Eating'.\n"
        f"3. SIMULATOR LABEL COUPLING (CRITICAL): If the person is holding a phone/smartphone in the 'BedRoom(dad)', the simulator records this action as 'Typing'. You MUST output 'Typing' for phone usage in DadRoom.\n"
        f"4. COOKING: If the person is near a stove, counter, or dining table holding a pan, pot, knife, or preparing food (like bananas/apples on table), output 'Cooking'.\n"
        f"5. OPENING: If the person is reaching out to pull the refrigerator door handle, output 'Opening'.\n\n"
        f"Allowed Activity List (Choose EXACTLY ONE): [{allowed_list}]\n\n"
        f"Output MUST follow this format strictly:\n"
        f"VISUAL_EVIDENCE: <Short sentence describing object in hand and posture>\n"
        f"ACTION: <exactly one label from the list>"
    )

    try:
        resp = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": VLM_MODEL, "prompt": prompt, "images": [img_base64], "stream": False,
                "options": {
                    "temperature": 0.001, # 壓低溫度，消滅幻覺隨機性
                    "num_predict": 80     # 放大長度防止 ACTION: 被截斷
                }
            },
            timeout=30
        )
        raw = resp.json().get("response", "").strip()
        
        action = None
        evidence = raw
        
        # 精確解析格式
        for line in raw.split("\n"):
            clean_line = line.strip()
            if clean_line.upper().startswith("ACTION:"):
                action = clean_line.split(":", 1)[1].strip()
            elif clean_line.upper().startswith("VISUAL_EVIDENCE:"):
                evidence = clean_line.split(":", 1)[1].strip()
                
        # Fallback 保底防範碎碎念
        if not action or action not in BEHAVIOR_LABELS:
            for b in BEHAVIOR_LABELS:
                if b.lower() in raw.lower():
                    action = b
                    break
        return action, evidence
    except Exception as e:
        return "Watching", f"Error: {e}"

def main():
    import cv2
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    groups = group_bursts(IMAGE_DIR)
    
    print(f"Starting End-to-End Direct VLM Evaluation using {VLM_MODEL}...")
    print(f"{'GT':15} {'VLM Direct Pred':18} {'Room':15} {'Person':10} OK")
    print("-"*75)
    
    correct = 0
    total = 0
    results = []

    for key, images in sorted(groups.items()):
        # 時序濾波：讓群組連拍圖片進行集體投票，防止單幀抖動死角
        group_preds = []
        group_evidences = []
        
        gt = images[0]["ground_truth"]
        room = images[0]["room_name"]
        person = images[0]["user_id"]
        
        for img_info in images:
            pred, evidence = vlm_direct_infer(img_info["path"], room, person)
            if pred:
                group_preds.append(pred)
                group_evidences.append(evidence)
                
        if group_preds:
            final_pred = Counter(group_preds).most_common(1)[0][0]
            best_idx = group_preds.index(final_pred)
            final_evidence = group_evidences[best_idx]
        else:
            final_pred, final_evidence = "Watching", "No response"
            
        total += 1
        is_correct = (final_pred == gt)
        if is_correct: correct += 1
        
        results.append({"gt": gt, "is_correct": is_correct})
        ok_marker = "✓" if is_correct else "✗"
        print(f"{ok_marker} {gt:14} {str(final_pred):17} {room:14} {person:9}")
        
        # 繪製視覺化 Debug 看板，方便在 visual_debug_results 裡直接點擊圖片確認
        rep_path = images[0]["path"]
        img_canvas = cv2.imread(rep_path)
        if img_canvas is not None:
            color = (0, 255, 0) if is_correct else (0, 0, 255)
            status_str = f"[{ok_marker}] GT: {gt}  |  VLM_PRED: {final_pred}"
            cv2.putText(img_canvas, status_str, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
            
            # 把 VLM 的描述直接畫在第二行
            evidence_clean = str(final_evidence)[:60]
            cv2.putText(img_canvas, f"VLM See: {evidence_clean}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1, cv2.LINE_AA)
            
            out_filename = f"{ok_marker}_{os.path.basename(rep_path)}"
            cv2.imwrite(os.path.join(OUTPUT_DIR, out_filename), img_canvas)

    print(f"\n{'='*60}")
    print(f"Direct VLM Accuracy: {correct}/{total} = {correct/total*100:.1f}%")

if __name__ == "__main__":
    main()