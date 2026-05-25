import os
import re
import cv2
import torch
import numpy as np
from PIL import Image
from collections import defaultdict, Counter
from transformers import CLIPProcessor, CLIPModel
from ultralytics import YOLO

# === 系統配置 ===
IMAGE_DIR  = "/home/wenny/db/robotBrain/debug_images"
OUTPUT_DIR = "./clip_crop_visuals"

# 11 類俯視角人體微觀語意描述
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

print("正在初始化系統 (人物互動裁切版：YOLOv8s 80類全開 + 門檻下調 + CLIP)...")
device = "cuda" if torch.cuda.is_available() else "cpu"

# 採用偵測實力較強的 s 版本
yolo_model = YOLO("yolov8s.pt")  

clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

YOLO_NAMES = yolo_model.names

# CLIP 文本特徵向量預編碼
text_inputs = clip_processor(text=list(BEHAVIOR_PROMPTS.values()), return_tensors="pt", padding=True).to(device)
with torch.no_grad():
    text_features = clip_model.get_text_features(**text_inputs)
    text_features /= text_features.norm(dim=-1, keepdim=True)

def parse_filename(fname):
    base  = os.path.basename(fname).replace(".jpg","").replace(".png","")
    parts = base.split("_")
    if len(parts) < 5:
        return {"timestamp": "0", "user_id": "Unknown", "ground_truth": "Unknown", "room_name": "Unknown"}
    return {"timestamp": parts[0], "user_id": parts[2], "ground_truth": parts[3], "room_name": parts[4]}

def process_and_visualize():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    files = sorted(f for f in os.listdir(IMAGE_DIR) if f.endswith(".jpg") or f.endswith(".png"))
    print(f"找到 {len(files)} 張圖片。開始執行人物區域裁切與微觀偵測...")
    
    for fname in files:
        img_path = os.path.join(IMAGE_DIR, fname)
        img_bgr = cv2.imread(img_path)
        if img_bgr is None: continue
        
        info = parse_filename(fname)
        gt = info["ground_truth"]
        
        # 1. 先跑全圖 YOLO 偵測，找出人體位置（conf=0.15 確保不漏抓人或手上小物）
        yolo_res = yolo_model(img_bgr, conf=0.15, verbose=False)
        
        person_box = None
        all_boxes_info = []
        
        for r in yolo_res:
            for b in r.boxes:
                cls_id = int(b.cls[0])
                conf_val = float(b.conf[0])
                box_coord = b.xyxy[0].cpu().numpy().astype(int)
                
                if cls_id == 0:
                    person_box = box_coord  # 鎖定人體大框
                
                obj_label = YOLO_NAMES.get(cls_id, f"class_{cls_id}").upper()
                all_boxes_info.append({"label": obj_label, "conf": conf_val, "box": box_coord})
        
        # 2. 如果有抓到人，就以人體框為核心進行裁切（往外擴展 40 像素確保能包進手上物件）
        if person_box is not None:
            h, w, _ = img_bgr.shape
            pad = 40  # 擴展邊距
            x1 = max(0, person_box[0] - pad)
            y1 = max(0, person_box[1] - pad)
            x2 = min(w, person_box[2] + pad)
            y2 = min(h, person_box[3] + pad)
            
            # 建立裁切後的畫布
            interaction_canvas = img_bgr[y1:y2, x1:x2].copy()
            
            # 3. 將原本落在這個區域內的 YOLO 綠色框，等比例對應畫到裁切圖上
            for obj in all_boxes_info:
                bx1, by1, bx2, by2 = obj["box"]
                
                # 檢查這個物件是不是在我們裁切的視野範圍內
                if bx2 >= x1 and bx1 <= x2 and by2 >= y1 and by1 <= y2:
                    # 轉換為裁切圖的相對座標
                    rx1, ry1 = max(0, bx1 - x1), max(0, by1 - y1)
                    rx2, ry2 = min(x2 - x1, bx2 - x1), min(y2 - y1, by2 - y1)
                    
                    display_txt = f"{obj['label']} {obj['conf']:.2f}"
                    
                    # 畫上綠色框
                    cv2.rectangle(interaction_canvas, (rx1, ry1), (rx2, ry2), (0, 255, 0), 2)
                    
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    label_size = 0.35
                    label_thickness = 1
                    text_size = cv2.getTextSize(display_txt, font, label_size, label_thickness)[0]
                    
                    tx = rx1
                    ty = max(ry1, text_size[1] + 5)
                    
                    cv2.rectangle(interaction_canvas, (tx, ty - text_size[1] - 5), (tx + text_size[0], ty), (0, 255, 0), cv2.FILLED)
                    cv2.putText(interaction_canvas, display_txt, (tx, ty - 2), font, label_size, (0, 0, 0), label_thickness, cv2.LINE_AA)
            
            # 4. CLIP 對這張人物局部裁切圖進行推理（更聚焦於動作本身）
            crop_pil = Image.fromarray(cv2.cvtColor(img_bgr[y1:y2, x1:x2], cv2.COLOR_BGR2RGB))
            img_inputs = clip_processor(images=crop_pil, return_tensors="pt").to(device)
            with torch.no_grad():
                img_features = clip_model.get_image_features(**img_inputs)
                img_features /= img_features.norm(dim=-1, keepdim=True)
                
            similarity_scores = (img_features @ text_features.T).cpu().numpy()[0]
            best_idx = np.argmax(similarity_scores)
            clip_pred = BEHAVIOR_LABELS[best_idx]
            score = float(similarity_scores[best_idx])
            
            # 5. 在裁切圖的頂部繪製輕量資訊面板
            is_correct = (clip_pred == gt)
            color = (0, 255, 0) if is_correct else (0, 0, 255)
            marker = "[✓]" if is_correct else "[✗]"
            
            status_txt = f"{marker} GT: {gt} | Pred: {clip_pred} ({score:.2f})"
            cv2.putText(interaction_canvas, status_txt, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
            
            # 儲存裁切後的人物互動對照圖
            cv2.imwrite(os.path.join(OUTPUT_DIR, f"CROP_VIS_{fname}"), interaction_canvas)
        else:
            # 如果這張圖連 YOLO 都完全找不到人，就維持原圖並上傳警告，方便你排查
            canvas = img_bgr.copy()
            cv2.putText(canvas, "NO PERSON DETECTED", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imwrite(os.path.join(OUTPUT_DIR, f"CROP_VIS_{fname}"), canvas)
            
    print(f"\n人物互動裁切圖已成功生成至 '{OUTPUT_DIR}'！")

if __name__ == "__main__":
    process_and_visualize()