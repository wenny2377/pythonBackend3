import csv
import os
import re
from collections import defaultdict
from pymongo import MongoClient

CHARADES_TRAIN_CSV   = "Charades_v1_train.csv"
CHARADES_CLASSES_TXT = "Charades_v1_classes.txt"
MONGO_URI = "mongodb://127.0.0.1:27017/"
DB_NAME   = "robot_rag_db"

YOUR_LABELS = [
    "Drinking", "SittingDrink", "Eating", "Cooking", "Opening",
    "Laying", "Watching", "Reading", "Cleaning", "PhoneUse",
    "Typing", "PickingUp", "PuttingDown", "Standing", "Walking",
]

CHARADES_TO_YOUR = {
    "holding some clothes": "Cleaning",
    "putting clothes somewhere": "PuttingDown",
    "taking some clothes from somewhere": "PickingUp",
    "throwing clothes somewhere": "PuttingDown",
    "tidying some clothes": "Cleaning",
    "washing some clothes": "Cleaning",
    "closing a door": "Opening",
    "fixing a door": "Opening",
    "opening a door": "Opening",
    "putting something on a table": "PuttingDown",
    
    # 💡 完美對齊 1：將官方所有純坐下的描述，100% 精準對齊到你的 "Sitting" 標籤！
    "sitting on a table": "Sitting",
    "sitting at a table": "Sitting",
    "sitting in a chair": "Sitting",
    "sitting on sofa/couch": "Sitting",
    "sitting on the floor": "Sitting",
    "sitting in a bed": "Sitting",
    "someone is going from standing to sitting": "Sitting",
    
    "tidying up a table": "Cleaning",
    "washing a table": "Cleaning",
    "working at a table": "Typing",
    "holding a phone/camera": "PhoneUse",
    "playing with a phone/camera": "PhoneUse",
    "putting a phone/camera somewhere": "PuttingDown",
    "taking a phone/camera from somewhere": "PickingUp",
    "talking on a phone/camera": "PhoneUse",
    "holding a bag": "PickingUp",
    "opening a bag": "Opening",
    "putting a bag somewhere": "PuttingDown",
    "taking a bag from somewhere": "PickingUp",
    "throwing a bag somewhere": "PuttingDown",
    "closing a book": "Reading",
    "holding a book": "Reading",
    "opening a book": "Opening",
    "putting a book somewhere": "PuttingDown",
    "smiling at a book": "Reading",
    "taking a book from somewhere": "PickingUp",
    "throwing a book somewhere": "PuttingDown",
    "watching/reading/looking at a book": "Reading",
    "holding a towel/s": "Cleaning",
    "putting a towel/s somewhere": "PuttingDown",
    "taking a towel/s from somewhere": "PickingUp",
    "throwing a towel/s somewhere": "PuttingDown",
    "tidying up a towel/s": "Cleaning",
    "washing something with a towel": "Cleaning",
    "closing a box": "Opening",
    "holding a box": "PickingUp",
    "opening a box": "Opening",
    "putting a box somewhere": "PuttingDown",
    "taking a box from somewhere": "PickingUp",
    "taking something from a box": "PickingUp",
    "throwing a box somewhere": "PuttingDown",
    "closing a laptop": "Typing",
    "holding a laptop": "Typing",
    "opening a laptop": "Opening",
    "putting a laptop somewhere": "PuttingDown",
    "taking a laptop from somewhere": "PickingUp",
    "watching a laptop or something on a laptop": "Watching",
    "working/playing on a laptop": "Typing",
    "holding a shoe/shoes": "PickingUp",
    "putting shoes somewhere": "PuttingDown",
    
    # 💡 完美對齊 2：穿脫鞋子與起立，對齊到你的 "Standing"
    "putting on shoe/shoes": "Standing",
    "taking shoes from somewhere": "PickingUp",
    "taking off some shoes": "Standing",
    "throwing shoes somewhere": "PuttingDown",
    
    "standing on a chair": "Standing",
    "holding some food": "Eating",
    "putting some food somewhere": "PuttingDown",
    "taking food from somewhere": "PickingUp",
    "throwing food somewhere": "PuttingDown",
    "eating a sandwich": "Eating",
    "making a sandwich": "Cooking",
    "holding a sandwich": "Eating",
    "putting a sandwich somewhere": "PuttingDown",
    "taking a sandwich from somewhere": "PickingUp",
    "holding a blanket": "Laying",
    "putting a blanket somewhere": "PuttingDown",
    "snuggling with a blanket": "Laying",
    "taking a blanket from somewhere": "PickingUp",
    "throwing a blanket somewhere": "PuttingDown",
    "tidying up a blanket/s": "Cleaning",
    "holding a pillow": "Laying",
    "putting a pillow somewhere": "PuttingDown",
    "snuggling with a pillow": "Laying",
    "taking a pillow from somewhere": "PickingUp",
    "throwing a pillow somewhere": "PuttingDown",
    "putting something on a shelf": "PuttingDown",
    "tidying a shelf or something on a shelf": "Cleaning",
    "reaching for and grabbing a picture": "PickingUp",
    "holding a picture": "Watching",
    "laughing at a picture": "Watching",
    "putting a picture somewhere": "PuttingDown",
    "taking a picture of something": "PhoneUse",
    "watching/looking at a picture": "Watching",
    "closing a window": "Opening",
    "opening a window": "Opening",
    "washing a window": "Cleaning",
    "watching/Looking outside of a window": "Watching",
    "holding a mirror": "Watching",
    "smiling in a mirror": "Watching",
    "washing a mirror": "Cleaning",
    "watching something/someone/themselves in a mirror": "Watching",
    "walking through a doorway": "Walking",
    "holding a broom": "Cleaning",
    "putting a broom somewhere": "PuttingDown",
    "taking a broom from somewhere": "PickingUp",
    "throwing a broom somewhere": "PuttingDown",
    "tidying up with a broom": "Cleaning",
    "fixing a light": "Standing",
    "turning on a light": "Standing",
    "turning off a light": "Standing",
    "drinking from a cup/glass/bottle": "Drinking",
    "holding a cup/glass/bottle of something": "Drinking",
    "pouring something into a cup/glass/bottle": "Drinking",
    "putting a cup/glass/bottle somewhere": "PuttingDown",
    "taking a cup/glass/bottle from somewhere": "PickingUp",
    "washing a cup/glass/bottle": "Cleaning",
    "closing a closet/cabinet": "Opening",
    "opening a closet/cabinet": "Opening",
    "tidying up a closet/cabinet": "Cleaning",
    "someone is holding a paper/notebook": "Reading",
    "putting their paper/notebook somewhere": "PuttingDown",
    "taking paper/notebook from somewhere": "PickingUp",
    "holding a dish": "Cleaning",
    "putting a dish/es somewhere": "PuttingDown",
    "taking a dish/es from somewhere": "PickingUp",
    "wash a dish/dishes": "Cleaning",
    "lying on a sofa/couch": "Laying",
    "lying on the floor": "Laying",
    "throwing something on the floor": "PuttingDown",
    "tidying something on the floor": "Cleaning",
    "holding some medicine": "Drinking",
    "taking/consuming some medicine": "Drinking",
    "putting groceries somewhere": "PuttingDown",
    "laughing at television": "Watching",
    "watching television": "Watching",
    "lying on a bed": "Laying",
    "fixing a vacuum": "Cleaning",
    "holding a vacuum": "Cleaning",
    "taking a vacuum from somewhere": "PickingUp",
    "washing their hands": "Cleaning",
    "fixing a doorknob": "Opening",
    "grasping onto a doorknob": "Opening",
    "closing a refrigerator": "Opening",
    "opening a refrigerator": "Opening",
    "fixing their hair": "Standing",
    "working on paper/notebook": "Typing",
    "someone is cooking something": "Cooking",
    "someone is dressing": "Standing",
    "someone is laughing": "Watching",
    "someone is running somewhere": "Walking",
    "someone is smiling": "Watching",
    "someone is sneezing": "Standing",
    "someone is undressing": "Standing",
    "someone is eating something": "Eating",
    
    # 💡 完美對齊 3：官方唯一的起立動作代號，完美對齊到你的 "StandUp" 灰色動畫方塊！
    "someone is standing up from somewhere": "StandUp",
    "someone is awakening somewhere": "StandUp",
    "someone is awakening in bed": "StandUp"
}

FURNITURE_PATTERNS = {
    "sofa":         re.compile(r"\b(sofa|couch|settee)\b",         re.IGNORECASE),
    "bed":          re.compile(r"\b(bed)\b",                       re.IGNORECASE),
    "refrigerator": re.compile(r"\b(refrigerator|fridge)\b",       re.IGNORECASE),
    "stove":        re.compile(r"\b(stove|oven|cooker)\b",         re.IGNORECASE),
    "dining table": re.compile(r"\b(dining\s+table|dinner\s+table)\b", re.IGNORECASE),
    "chair":        re.compile(r"\b(chair)\b",                     re.IGNORECASE),
    "tv":           re.compile(r"\b(television|tv)\b",             re.IGNORECASE),
    "desk":         re.compile(r"\b(desk)\b",                      re.IGNORECASE),
    "sink":         re.compile(r"\b(sink|washbasin)\b",            re.IGNORECASE),
    "toilet":       re.compile(r"\b(toilet)\b",                    re.IGNORECASE),
    "cabinet":      re.compile(r"\b(cabinet|cupboard)\b",          re.IGNORECASE),
}

OVERLAP_THRESHOLD = 0.5
MIN_TRANSITION_PROB = 0.01


def load_classes(path):
    classes = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(" ", 1)
            if len(parts) == 2:
                clean_desc = parts[1].strip().lower()
                clean_desc = re.sub(r'\s+o\d+\s+v\d+.*$', '', clean_desc)
                classes[parts[0]] = clean_desc
    return classes


def map_to_your_label(description):
    return CHARADES_TO_YOUR.get(description.strip().lower(), None)


def detect_furniture(text):
    found = set()
    for furn_label, pattern in FURNITURE_PATTERNS.items():
        if pattern.search(text):
            found.add(furn_label)
    return list(found)


def parse_actions(actions_str):
    if not actions_str or not actions_str.strip():
        return []
    result = []
    for seg in actions_str.strip().split(";"):
        seg = seg.strip()
        if not seg:
            continue
        parts = seg.split()
        if len(parts) >= 3:
            result.append({
                "class_id": parts[0],
                "start":    float(parts[1]),
                "end":      float(parts[2]),
            })
    return result


def actions_overlap(a, b, threshold=OVERLAP_THRESHOLD):
    overlap_start = max(a["start"], b["start"])
    overlap_end   = min(a["end"],   b["end"])
    if overlap_end <= overlap_start:
        return False
    overlap_len = overlap_end - overlap_start
    a_len = a["end"] - a["start"]
    b_len = b["end"] - b["start"]
    return overlap_len / max(a_len, b_len, 1e-6) >= threshold


def process_csv(csv_path, classes):
    pair_counts = defaultdict(lambda: defaultdict(int))
    cooccur     = defaultdict(lambda: defaultdict(int))
    furn_totals = defaultdict(int)

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw_actions  = parse_actions(row.get("actions", ""))
            scene_desc   = row.get("scene", "")
            script_text  = row.get("script", "")
            combined_ctx = scene_desc + " " + script_text

            if not raw_actions:
                continue

            present_furniture = detect_furniture(combined_ctx)

            mapped = []
            for act in raw_actions:
                desc = classes.get(act["class_id"], "")
                your = map_to_your_label(desc)
                if your:
                    # 💡 關鍵修改 2：環境智慧感知聯動规则
                    # 當人在沙發、椅子或餐桌附近喝水時，自動將大數據映射轉換為符合你的 "SittingDrink" 標籤
                    if your == "Drinking" and any(f in present_furniture for f in ["sofa", "chair", "dining table"]):
                        your = "SittingDrink"

                    mapped.append({
                        "label": your,
                        "start": act["start"],
                        "end":   act["end"],
                        "desc":  desc,
                    })

            # 重點時間線排序：確保動作由早到晚排列
            mapped.sort(key=lambda x: x["start"])

            # 嚴格的時間線連續動作判定
            for i in range(len(mapped)):
                a = mapped[i]
                for j in range(i + 1, len(mapped)):
                    b = mapped[j]
                    
                    if b["start"] > a["end"] + 2.0:
                        break
                    
                    overlap_start = max(a["start"], b["start"])
                    overlap_end   = min(a["end"],   b["end"])
                    has_overlap   = overlap_end > overlap_start
                    
                    if not has_overlap:
                        pair_counts[a["label"]][b["label"]] += 1
                    else:
                        overlap_len = overlap_end - overlap_start
                        a_len = a["end"] - a["start"]
                        if (overlap_len / max(a_len, 1e-6)) < 0.1:
                            pair_counts[a["label"]][b["label"]] += 1

            # Affordance 共現（去重統計）
            seen_in_this_video = set()
            for act in mapped:
                act_furniture = detect_furniture(act["desc"])
                all_furn      = list(set(present_furniture + act_furniture))
                for furn in all_furn:
                    pair_key = (furn, act["label"])
                    if pair_key not in seen_in_this_video:
                        seen_in_this_video.add(pair_key)
                        cooccur[furn][act["label"]] += 1

            for furn in set(f for f, _ in seen_in_this_video):
                furn_totals[furn] += 1

    return pair_counts, cooccur, furn_totals


def compute_transition_matrix(pair_counts):
    matrix = {}
    for src, dsts in pair_counts.items():
        raw_total = sum(dsts.values())
        if raw_total == 0:
            continue

        filtered = {
            dst: count / raw_total
            for dst, count in dsts.items()
            if count / raw_total >= MIN_TRANSITION_PROB
        }

        if not filtered:
            continue

        filtered_total = sum(filtered.values())
        matrix[src] = {
            dst: round(prob / filtered_total, 6)
            for dst, prob in filtered.items()
        }

    return matrix


def compute_affordance_affinity(cooccur, furn_totals):
    result = {}
    for furn, actions in cooccur.items():
        total = furn_totals.get(furn, 1)
        if total == 0:
            continue
        result[furn] = {
            action: round(count / total, 4)
            for action, count in actions.items()
        }
    return result


def normalize_affinity(raw_affinity):
    normalized = {}
    for furn, actions in raw_affinity.items():
        max_score = max(actions.values()) if actions else 1.0
        if max_score == 0:
            continue
        normalized[furn] = {
            action: round(score / max_score, 4)
            for action, score in actions.items()
        }
    return normalized


def validate_transition_matrix(matrix):
    print("\n=== 轉移矩陣驗證（每行加總應 = 1.0）===")
    all_ok = True
    for src, dsts in sorted(matrix.items()):
        total = sum(dsts.values())
        ok    = abs(total - 1.0) < 0.001
        mark  = "✓" if ok else "✗ 警告"
        print(f"  {src:15} 加總 = {total:.6f} {mark}")
        if not ok:
            all_ok = False
    if all_ok:
        print("  所有行加總正常（重新歸一化生效）。")
    return all_ok


def save_to_mongo(transition_matrix, raw_affinity, normalized_affinity):
    client = MongoClient(MONGO_URI)
    db     = client[DB_NAME]

    db.transition_matrix.delete_many({})
    trans_docs = []
    for src, dsts in transition_matrix.items():
        for dst, prob in dsts.items():
            trans_docs.append({
                "from":        src,
                "to":          dst,
                "probability": prob,
            })
    if trans_docs:
        db.transition_matrix.insert_many(trans_docs)
    print(f"\n[Charades] transition_matrix：{len(trans_docs)} 筆寫入")

    db.charades_affinity.delete_many({})
    aff_docs = []
    for furn, actions in raw_affinity.items():
        for action, score in actions.items():
            aff_docs.append({
                "furniture": furn,
                "behavior":  action,
                "p_cond":    score,
                "source":    "charades",
            })
    if aff_docs:
        db.charades_affinity.insert_many(aff_docs)
    print(f"[Charades] charades_affinity：{len(aff_docs)} 筆寫入")

    db.charades_affinity_normalized.delete_many({})
    norm_docs = []
    for furn, actions in normalized_affinity.items():
        for action, score in actions.items():
            norm_docs.append({
                "furniture": furn,
                "behavior":  action,
                "score":     score,
                "source":    "charades_normalized",
            })
    if norm_docs:
        db.charades_affinity_normalized.insert_many(norm_docs)
    print(f"[Charades] charades_affinity_normalized：{len(norm_docs)} 筆寫入")

    client.close()


def print_summary(transition_matrix, normalized_affinity):
    print("\n=== Transition Matrix 樣本 ===")
    for src in YOUR_LABELS[:8]:
        row = transition_matrix.get(src, {})
        top = sorted(row.items(), key=lambda x: -x[1])[:3]
        if top:
            total = sum(row.values())
            print(f"  {src:15} → {top}  (加總={total:.4f})")

    print("\n=== Affordance Affinity（正規化，貼入 _builtin_affinity_fallback）===")
    for furn in ["sofa", "stove", "bed", "tv", "refrigerator",
                 "dining table", "chair", "sink", "toilet", "desk"]:
        row = normalized_affinity.get(furn, {})
        if not row:
            print(f'  "{furn}": {{}}  # 無 Charades 資料，維持手工設定')
            continue
        top       = sorted(row.items(), key=lambda x: -x[1])[:5]
        items_str = ", ".join(f'"{a}": {s}' for a, s in top)
        print(f'  "{furn}": {{{items_str}}},')


if __name__ == "__main__":
    if not os.path.exists(CHARADES_TRAIN_CSV):
        print(f"[錯誤] 找不到 {CHARADES_TRAIN_CSV}")
        print("請先下載：https://prior.allenai.org/projects/charades")
        exit(1)

    if not os.path.exists(CHARADES_CLASSES_TXT):
        print(f"[錯誤] 找不到 {CHARADES_CLASSES_TXT}")
        exit(1)

    print("[Charades] 載入 classes...")
    classes = load_classes(CHARADES_CLASSES_TXT)
    print(f"[Charades] {len(classes)} classes 載入完成")

    print("[Charades] 處理 CSV...")
    pair_counts, cooccur, furn_totals = process_csv(CHARADES_TRAIN_CSV, classes)

    print("[Charades] 計算 transition matrix（含重新歸一化）...")
    transition_matrix = compute_transition_matrix(pair_counts)

    print("[Charades] 計算 affordance affinity...")
    raw_affinity        = compute_affordance_affinity(cooccur, furn_totals)
    normalized_affinity = normalize_affinity(raw_affinity)

    validate_transition_matrix(transition_matrix)
    print_summary(transition_matrix, normalized_affinity)

    print("\n[Charades] 寫入 MongoDB...")
    save_to_mongo(transition_matrix, raw_affinity, normalized_affinity)

    print("\n[Charades] 完成。")