import os
import re
from collections import defaultdict, Counter
import cv2
import torch
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from ultralytics import YOLO  # 引入 YOLO 來當我們的「空間背景過濾器」

# === 系統配置 ===
IMAGE_DIR  = "/home/wenny/db/robotBrain/debug_images"
OUTPUT_DIR = "./clip_debug_results"

# 俯視角學術語意擴寫（此處聚焦於「人體本身」的姿態與手部動作描述，因為背景會被切掉）
BEHAVIOR_PROMPTS = {
    "Eating":      "A close-up surveillance photo of a person holding a bowl, food, or plate near their chest or mouth.",
    "Drinking":    "A photo of a person holding a cup, water bottle, or beverage can, raising their hand directly to their mouth to drink.",
    "SittingDrink": "A person sitting down, holding a small cup or mug in their hand.",
    "Cooking":     "A person holding a frying pan, cooking pot, kitchen utensil, or knife, handling ingredients.",
    "Opening":     "A person extending their arm and reaching out to pull open a handle or refrigerator door.",
    "Laying":      "A person lying flat horizontally, body stretched out resting on a soft couch surface or bed.",
    "Watching":    "A person standing or sitting with completely empty hands, looking forward at something.",
    "Reading":     "A person looking down at an open book, reading papers, or holding a document in hand.",
    "Cleaning":    "A person holding a broom, mop, cleaning cloth, or wiping a counter surface.",
    "PhoneUse":    "A person holding a smartphone or mobile phone in their hands, staring down at the screen.",
    "Typing":      "A person pressing keys on a laptop computer keyboard or typing on a device."
}
BEHAVIOR_LABELS = list(BEHAVIOR_PROMPTS.keys())

print("Loading Models...")
device = "cuda" if torch.cuda.is_available() else "cpu"
yolo_model = YOLO("yolov8n.pt") # 用最輕量級的物件框選器
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# 預先編碼文字向量
text_inputs = processor(text=list(BEHAVIOR_PROMPTS.values()), return_tensors="pt", padding=True).to(device)
with torch.no_grad():
    text_features = model.get_text_features(**text_inputs)
    text_features /= text_features.norm(dim=-1, keepdim=True)

def parse_filename(fname):
    base  = os.path.basename(fname).replace(".jpg","").replace(".png","")
    parts = base.split("_")
    return {"timestamp": parts[0], "user_id": parts[2], "ground_truth": parts[3], "room_name": parts[4]}

def group_bursts(image_dir):
    files  = sorted(f for f in os.listdir(image_dir) if f.endswith(".jpg") or f.endswith(".png"))
    groups = defaultdict(list)
    for f in files:
        info = parse_filename(f)
        key  = f"{info['timestamp']}_{info['user_id']}_{info['ground_truth']}_{info['room_name']}"
        groups[key].append({**info, "path": os.path.join(image_dir, f)})
    return dict(groups)

def clip_predict_cropped_person(image_path):
    """【學術界主流作法】: 利用 YOLO 裁切出人體局部影像，排除大背景干擾，再送交 CLIP 進行對齊"""
    try:
        img_bgr = cv2.imread(image_path)
        if img_bgr is None: return "Watching", 0.0
        
        # 1. 使用 YOLO 偵測人的位置 (class 0 為 person)
        yolo_res = yolo_model(img_bgr, verbose=False)
        box = None
        for r in yolo_res:
            for b in r.boxes:
                if int(b.cls[0]) == 0: # 抓到人體
                    box = b.xyxy[0].cpu().numpy().astype(int)
                    break
        
        # 2. 核心空間解耦：如果抓到人體框，就把人體切片出來；否則用原圖
        if box is not None:
            # 稍微向外擴大 15 像素，確保手持物件（如長平底鍋、可樂罐）沒有被切邊
            h, w, _ = img_bgr.shape
            x1, y1, x2, y2 = max(0, box[0]-15), max(0, box[1]-15), min(w, box[2]+15), min(h, box[3]+15)
            crop_bgr = img_bgr[y1:y2, x1:x2]
            crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
            image_pil = Image.fromarray(crop_rgb)
        else:
            image_pil = Image.open(image_path).convert("RGB")
            
        # 3. 特徵向量匹配
        img_inputs = processor(images=image_pil, return_tensors="pt").to(device)
        with torch.no_grad():
            img_features = model.get_image_features(**img_inputs)
            img_features /= img_features.norm(dim=-1, keepdim=True)
        
        similarity_scores = (img_features @ text_features.T).cpu().numpy()[0]
        best_idx = np.argmax(similarity_scores)
        return BEHAVIOR_LABELS[best_idx], float(similarity_scores[best_idx])
    except:
        return "Watching", 0.0

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    groups = group_bursts(IMAGE_DIR)
    
    print(f"\nEvaluating via Object-Level Semantic Space Alignment...")
    print(f"{'GT':15} {'CLIP Crop Pred':18} {'Room':15} {'Person':10} Score  OK")
    print("-"*82)
    
    correct = 0
    total = 0

    for key, images in sorted(groups.items()):
        group_preds = []
        group_scores = []
        gt = images[0]["ground_truth"]
        room = images[0]["room_name"]
        person = images[0]["user_id"]
        
        for img_info in images:
            pred, score = clip_predict_cropped_person(img_info["path"])
            group_preds.append(pred)
            group_scores.append(score)
            
        final_pred = Counter(group_preds).most_common(1)[0][0]
        best_score = group_scores[group_preds.index(final_pred)]
        
        total += 1
        is_correct = (final_pred == gt)
        if is_correct: correct += 1
        
        ok_marker = "✓" if is_correct else "✗"
        print(f"{ok_marker} {gt:14} {final_pred:17} {room:14} {person:9} {best_score:.3f}")

    print(f"\n{'='*60}")
    print(f"CLIP Cropped Person Accuracy: {correct}/{total} = {correct/total*100:.1f}%")

if __name__ == "__main__":
    main()