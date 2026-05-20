"""
test_vlm_prompt.py — VLM Prompt A/B/C 比較測試工具

Usage:
  python3 test_vlm_prompt.py                        # 預設：跑前50張，只顯示 summary
  python3 test_vlm_prompt.py --limit 0 --verbose     # 跑所有圖，且顯示每張詳細結果
  python3 test_vlm_prompt.py --strategy B            # 只跑 Strategy B
  python3 test_vlm_prompt.py --show-fails            # 只顯示辨識錯的
  python3 test_vlm_prompt.py --image xxx.jpg --gt Cleaning --show-raw
  python3 test_vlm_prompt.py --model bakllava:latest
"""

import os
import base64
import json
import argparse
import requests
from pathlib import Path

OLLAMA_URL = "http://localhost:11434"
DEFAULT_MODEL = "llava-phi3:latest"

BEHAVIOR_LABELS = [
    "Eating", "Drinking", "SittingDrink", "Cooking", "Opening",
    "Laying", "Watching", "Reading", "Cleaning", "PhoneUse", "Typing",
    "Standing", "Walking",
]

NORMALIZE = {
    "eating": "Eating", "drinking": "Drinking", "sittingdrink": "SittingDrink",
    "cooking": "Cooking", "opening": "Opening", "laying": "Laying",
    "watching": "Watching", "reading": "Reading", "cleaning": "Cleaning",
    "phoneuse": "PhoneUse", "typing": "Typing", "standing": "Standing",
    "walking": "Walking", "sleep": "Laying", "sweeping": "Cleaning",
    "mopping": "Cleaning", "lying": "Laying", "lying down": "Laying",
}

STRATEGIES = {

"A": """Analyze this home surveillance camera image.
Output ONLY valid JSON:
{
  "body_posture": "brief description",
  "gaze_target": "where looking",
  "hand_state": "what in hands or empty",
  "person_near": "nearest furniture",
  "summary": "one sentence"
}""",

"B": """You are a home robot camera. Analyze this image and identify the exact activity.
Output ONLY valid JSON, no explanation:
{
  "activity": "MUST be exactly one of: Eating/Drinking/SittingDrink/Cooking/Opening/Laying/Watching/Reading/Cleaning/PhoneUse/Typing/Standing/Walking",
  "held_object": "object in hand or none",
  "person_near": "nearest furniture",
  "confidence": "high/medium/low"
}

Strict Rules to overcome failures:
- If the person is standing close to a refrigerator, cupboard, or door with their hand touching or near the handle, you MUST classify this as "Opening".
- If the person is standing or moving near the kitchen counter/stove holding ingredients, pans, or kitchen tools, classify as "Cooking".
- If holding a cup/bottle/glass while sitting = SittingDrink.
- If holding a cup/bottle/glass while standing = Drinking.
- If putting food into mouth or holding utensils (fork/spoon/chopsticks) while near table/counter = Eating.
- horizontal body on bed/sofa = Laying.""",

"C": """Look at this indoor camera image step by step.
1. Body position: standing, sitting, or lying?
2. What is in the person's hands?
3. What furniture is closest?
4. What activity matches best?

Output ONLY valid JSON:
{
  "body_position": "standing/sitting/lying",
  "held_object": "object or none",
  "nearest_furniture": "furniture name",
  "activity": "exactly one of: Eating/Drinking/SittingDrink/Cooking/Opening/Laying/Watching/Reading/Cleaning/PhoneUse/Typing/Standing"
}""",

}


def encode_image(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def call_vlm(image_b64: str, prompt: str, model: str) -> str:
    try:
        resp = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model":  model,
                "prompt": prompt,
                "images": [image_b64],
                "stream": False,
                "options": {"temperature": 0.0, "num_predict": 200},
            },
            timeout=60,
        )
        return resp.json().get("response", "")
    except Exception as e:
        return f"ERROR: {e}"


def extract_activity(raw: str, strategy: str) -> str:
    try:
        start = raw.find("{")
        end   = raw.rfind("}") + 1
        if start < 0 or end <= start:
            return "parse_error"
        data = json.loads(raw[start:end])

        if strategy == "A":
            text = " ".join([
                data.get("summary", ""),
                data.get("hand_state", ""),
                data.get("body_posture", ""),
            ]).lower()
            for key, val in NORMALIZE.items():
                if key in text:
                    return val
            return "Unknown"

        elif strategy in ("B", "C"):
            act = data.get("activity", "").strip()
            key = act.lower().replace(" ", "").replace("_", "").replace("/", "")
            if key in NORMALIZE:
                return NORMALIZE[key]
            for k, v in NORMALIZE.items():
                if k in act.lower():
                    return v
            return act if act else "Unknown"

    except Exception:
        pass
    return "Unknown"


def get_gt(filename: str) -> str:
    stem  = Path(filename).stem
    parts = stem.split("_")
    if len(parts) >= 4:
        key = parts[3].lower()
        if key in NORMALIZE:
            return NORMALIZE[key]
    for key, val in NORMALIZE.items():
        if key in stem.lower():
            return val
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image",     help="single image path")
    parser.add_argument("--dir",       default="debug_images")
    parser.add_argument("--strategy",  help="A/B/C or comma-sep, default=all")
    parser.add_argument("--gt",        help="ground truth for single image")
    parser.add_argument("--model",     default=DEFAULT_MODEL)
    
    # 變更這裡：預設只跑前 50 張
    parser.add_argument("--limit",     type=int, default=50, help="max images to test (default: 50)")
    
    # 變更這裡：改用 --verbose 來控制印出詳細資訊。預設是安靜模式（不印每張圖的進度）
    parser.add_argument("--verbose",   action="store_true", help="show individual image results")
    
    parser.add_argument("--show-fails",action="store_true")
    parser.add_argument("--show-raw",  action="store_true")
    args = parser.parse_args()

    strategies = (args.strategy.upper().split(",")
                  if args.strategy else list(STRATEGIES.keys()))

    print(f"Model     : {args.model}")
    print(f"Strategies: {strategies}")

    if args.image:
        b64     = encode_image(args.image)
        gt      = args.gt
        results = {}
        for s in strategies:
            raw  = call_vlm(b64, STRATEGIES[s], args.model)
            act  = extract_activity(raw, s)
            ok   = (act == gt) if gt else None
            results[s] = {"activity": act, "correct": ok, "raw": raw}
            mark = "✅" if ok else ("❌" if ok is False else "❓")
            print(f"  [{s}] {mark} {act}")
            if args.show_raw:
                print(f"       {raw[:300]}")
        return

    image_dir = Path(args.dir)
    if not image_dir.exists():
        print(f"No directory: {image_dir}/")
        print("Run Flask + Unity first, images saved automatically.")
        return

    images = sorted(image_dir.glob("*.jpg")) + sorted(image_dir.glob("*.png"))
    if not images:
        print(f"No images in {image_dir}/")
        return

    if args.limit > 0:
        images = images[:args.limit]

    print(f"Testing {len(images)} images (Limit: {args.limit})")
    print("Analyzing... Please wait...")

    totals       = {s: {"c": 0, "t": 0} for s in strategies}
    act_stats    = {}
    all_results  = []

    for img in images:
        gt      = get_gt(img.name)
        b64     = encode_image(str(img))
        results = {}

        for s in strategies:
            raw = call_vlm(b64, STRATEGIES[s], args.model)
            act = extract_activity(raw, s)
            ok  = (act == gt) if gt else None
            results[s] = {"activity": act, "correct": ok, "raw": raw}
            if ok is not None:
                totals[s]["t"] += 1
                if ok:
                    totals[s]["c"] += 1
            if gt:
                if gt not in act_stats:
                    act_stats[gt] = {s: {"c":0,"t":0} for s in strategies}
                act_stats[gt][s]["t"] += 1
                if ok:
                    act_stats[gt][s]["c"] += 1

        all_results.append((img.name, results, gt))

        # 只有在使用者開了 --verbose 的情況下才印出每張圖的進度
        if args.verbose:
            has_fail = any(
                not r["correct"] for r in results.values()
                if r["correct"] is not None)
            if args.show_fails and not has_fail:
                continue
            print(f"  {img.name}")
            print(f"  GT: {gt or '?'}")
            for s, r in results.items():
                mark = "✅" if r["correct"] else ("❌" if r["correct"] is False else "❓")
                print(f"    [{s}] {mark} {r['activity']}")
                if args.show_raw:
                    print(f"         {r['raw'][:200]}")
            print()

    print(f"\n{'='*55}")
    print("  SUMMARY — Overall Accuracy")
    print(f"{'='*55}")
    for s in strategies:
        t = totals[s]
        if t["t"] > 0:
            acc = t["c"] / t["t"] * 100
            print(f"  Strategy {s}: {t['c']:3}/{t['t']:3} = {acc:.0f}%")

    print(f"\n{'─'*55}")
    print("  Per-activity Accuracy")
    print(f"{'─'*55}")
    header = f"  {'Activity':14}"
    for s in strategies:
        header += f"  [{s}]"
    print(header)
    for act in sorted(act_stats):
        row = f"  {act:14}"
        for s in strategies:
            st = act_stats[act][s]
            if st["t"] > 0:
                acc = st["c"] / st["t"] * 100
                row += f"  {acc:3.0f}%({st['t']:2})"
            else:
                row += "   n/a   "
        print(row)

    best = max(strategies, key=lambda s: totals[s]["c"] / max(totals[s]["t"],1))
    print(f"\n  → Best strategy: {best}")


if __name__ == "__main__":
    main()