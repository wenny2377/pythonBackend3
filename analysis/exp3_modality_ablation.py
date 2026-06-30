import os, sys, re, json, requests
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

import datetime
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict
from pymongo import MongoClient

from exp_config import (
    MONGO_URI, DB_BASELINE, OLLAMA_URL, LLM_MODEL,
    COL_SEMANTIC,
    COL_ABL_NO_SKELETON,
    COL_ABL_NO_OBJECT, COL_ABL_NO_SPATIAL,
    ADL_LABELS, ROOM_IMPOSSIBLE, C,
    FONT_TITLE, FONT_AXIS, FONT_ANNOT, FONT_TICK,
    FIG_DPI, RESULTS_DIR,
    apply_style, load_docs, compute_accuracy, normalize_gt,
)

apply_style()

ABLATION_MODES = {
    "no_skeleton": COL_ABL_NO_SKELETON,
    "no_object":   COL_ABL_NO_OBJECT,
    "no_spatial":  COL_ABL_NO_SPATIAL,
}

ABLATION_LABELS = {
    "full":        "Full System",
    "no_skeleton": "w/o Skeleton",
    "no_object":   "w/o Object Events",
    "no_spatial":  "w/o Spatial Context",
}


def _prune_candidates(room: str) -> list:
    candidates = set(ADL_LABELS)
    candidates -= ROOM_IMPOSSIBLE.get(room or "", set())
    return sorted(candidates) if candidates else sorted(ADL_LABELS)


def _build_scene_text(doc: dict, mask: str) -> str:
    parts = []
    room  = doc.get("room_name", "Unknown")
    slot  = doc.get("time_slot", "Unknown")
    parts.append(f"Room: {room}. Time: {slot}.")

    if mask != "no_skeleton":
        skel_parts = []
        for field, label in [
            ("body_axis_angle", "body axis angle"),
            ("head_pitch",      "head pitch"),
            ("hand_to_head",    "hand to head ratio"),
            ("knee_hip_ratio",  "knee hip ratio"),
            ("arm_elevation",   "arm elevation"),
        ]:
            v = doc.get(field, -1)
            if v is not None and float(v) >= 0:
                skel_parts.append(f"{label}={float(v):.2f}")
        if skel_parts:
            parts.append("Skeleton cues: " + ", ".join(skel_parts) + ".")

    if mask != "no_object":
        held = doc.get("held_event", "none")
        if held and held not in ("none", "", "null"):
            parts.append(f"Object event: {held}.")
        items = doc.get("interacting_items", [])
        if items:
            parts.append(f"Nearby objects: {', '.join(items)}.")

    if mask != "no_spatial":
        zone = doc.get("zone_label", "")
        if zone and "Unknown" not in zone:
            parts.append(f"Nearest furniture zone: {zone}.")

    return " ".join(parts)


def _llm_predict(scene_text: str, room: str) -> str:
    candidates = _prune_candidates(room)
    prompt = (
        f"You are an activity recognition system.\n"
        f"Scene: {scene_text}\n"
        f"Choose ONE activity from: {', '.join(candidates)}\n"
        f"Reply with ONLY the activity name, nothing else."
    )
    try:
        resp = requests.post(
            f"{OLLAMA_URL}/api/chat",
            json={
                "model":    LLM_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "stream":   False,
                "options":  {"temperature": 0.0, "num_predict": 10},
            },
            timeout=30,
        )
        resp.raise_for_status()
        raw = resp.json()["message"]["content"].strip()
        for label in candidates:
            if label.lower() in raw.lower():
                return label
        return raw.split()[0] if raw else "Unknown"
    except Exception as e:
        print(f"  [LLM error] {e}")
        return "Unknown"


def run_ablation_mode(db, docs: list, mask: str, col_name: str) -> list:
    col   = db[col_name]
    now   = datetime.datetime.utcnow()
    preds = []

    print(f"  [{mask}] Running {len(docs)} episodes...")
    for i, doc in enumerate(docs):
        scene   = _build_scene_text(doc, mask=mask)
        room    = doc.get("room_name", "")
        pred    = _llm_predict(scene, room)
        gt      = normalize_gt(doc.get("ground_truth", ""))
        pred_n  = normalize_gt(pred)
        correct = (gt == pred_n) if gt else None
        preds.append(pred_n)

        col.update_one(
            {"episode_id": doc.get("episode_id", str(doc.get("_id", "")))},
            {"$set": {
                "episode_id":    doc.get("episode_id", ""),
                "user":          doc.get("user", ""),
                "ground_truth":  gt,
                "predicted":     pred_n,
                "correct":       correct,
                "ablation_mode": mask,
                "room_name":     room,
                "time_slot":     doc.get("time_slot", ""),
                "virtual_day":   doc.get("virtual_day"),
                "virtual_hour":  doc.get("virtual_hour"),
                "timestamp":     now,
            }},
            upsert=True,
        )

        if (i + 1) % 20 == 0:
            tot  = sum(1 for d in docs[:i+1] if d.get("ground_truth") in ADL_LABELS)
            done = sum(1 for p, d in zip(preds, docs[:i+1])
                       if normalize_gt(d.get("ground_truth", "")) == p
                       and d.get("ground_truth") in ADL_LABELS)
            if tot:
                print(f"  [{mask}] {i+1}/{len(docs)} | acc so far: {done/tot:.1%}")

    return preds


def _load_ablation_results(db, col_name: str) -> list:
    docs = list(db[col_name].find(
        {"ground_truth": {"$exists": True, "$ne": ""},
         "predicted":    {"$exists": True}},
    ))
    for d in docs:
        d["_pred"]        = normalize_gt(d.get("predicted", ""))
        d["ground_truth"] = normalize_gt(d.get("ground_truth", ""))
    return docs


def plot_ablation_bar(results: dict, save_path: str):
    order    = ["full", "no_skeleton", "no_object", "no_spatial"]
    names    = [ABLATION_LABELS[k] for k in order if k in results]
    accs     = [results[k]["acc"] * 100 for k in order if k in results]
    full_acc = results.get("full", {}).get("acc", 0) * 100
    colors   = [C["baseline"] if k == "full" else C["ablation"]
                for k in order if k in results]

    order_idx = sorted(range(len(names)), key=lambda i: accs[i])
    names  = [names[i]  for i in order_idx]
    accs   = [accs[i]   for i in order_idx]
    colors = [colors[i] for i in order_idx]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(range(len(names)), accs, color=colors,
                   alpha=0.88, height=0.55, edgecolor="white")

    for bar, acc in zip(bars, accs):
        delta = full_acc - acc
        label = f"{acc:.1f}%" if delta < 0.1 else f"{acc:.1f}%  (-{delta:.1f}%)"
        ax.text(bar.get_width() + 0.5,
                bar.get_y() + bar.get_height() / 2,
                label, va="center", fontsize=FONT_ANNOT,
                color=C["highlight"] if delta > 5 else "#333",
                fontweight="bold" if delta > 5 else "normal")

    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=FONT_TICK)
    ax.set_xlabel("Recognition Accuracy (%)", fontsize=FONT_AXIS)
    ax.set_xlim(0, 115)
    ax.set_title(
        "Modality Ablation Study\n(LLM re-inference with each modality masked)",
        fontsize=FONT_TITLE, fontweight="bold", pad=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close()
    print(f"[exp3] Saved: {save_path}")


def plot_ablation_table(results: dict, save_path: str):
    order    = ["full", "no_skeleton", "no_object", "no_spatial"]
    full_acc = results.get("full", {}).get("acc", 0)

    modality_map = {
        "full":        (True,  True,  True),
        "no_skeleton": (False, True,  True),
        "no_object":   (True,  False, True),
        "no_spatial":  (True,  True,  False),
    }

    col_headers = ["Method", "Skeleton", "Object\nEvents", "Spatial\nCtx", "Acc (%)", "Delta"]
    rows = []

    for k in order:
        if k not in results:
            continue
        skel, obj, spa = modality_map[k]
        acc   = results[k]["acc"]
        delta = (full_acc - acc) * 100
        rows.append([
            ABLATION_LABELS[k],
            "+" if skel else "-",
            "+" if obj  else "-",
            "+" if spa  else "-",
            f"{acc*100:.1f}%",
            "--" if k == "full" else f"-{delta:.1f}%",
        ])

    fig, ax = plt.subplots(figsize=(11, 3.5))
    ax.axis("off")
    table = ax.table(
        cellText=rows, colLabels=col_headers,
        cellLoc="center", loc="center",
        colWidths=[0.32, 0.12, 0.13, 0.13, 0.14, 0.12])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.2)

    header_color = "#2C3E50"
    for j in range(len(col_headers)):
        cell = table[0, j]
        cell.set_facecolor(header_color)
        cell.set_text_props(color="white", fontweight="bold")

    row_colors = ["#EBF5FB", "#FDFEFE"] * 5
    for i, row_data in enumerate(rows):
        delta_str = row_data[5]
        try:
            is_high = (delta_str not in ("--",) and
                       float(delta_str.replace("-", "").replace("%", "")) > 5)
        except ValueError:
            is_high = False

        for j in range(len(col_headers)):
            cell = table[i + 1, j]
            cell.set_facecolor("#FDEDEC" if is_high else row_colors[i])
            if j in (1, 2, 3):
                txt = row_data[j]
                cell.set_text_props(
                    color="#27AE60" if txt == "+" else "#E74C3C",
                    fontweight="bold", fontsize=13)
            if j == 5 and is_high:
                cell.set_text_props(color="#E74C3C", fontweight="bold")

    ax.set_title("Modality Ablation Study",
                 fontsize=FONT_TITLE, fontweight="bold", pad=16)
    plt.tight_layout()
    plt.savefig(save_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close()
    print(f"[exp3] Saved: {save_path}")


def save_summary(results: dict, save_path: str):
    lines = [
        "Experiment 3: Modality Ablation",
        f"DB: {DB_BASELINE}",
        "",
        f"{'Method':<26} {'Acc':>6} {'Delta':>8} {'N':>6}",
        "-" * 50,
    ]
    full_acc = results.get("full", {}).get("acc", 0)
    for k in ["full", "no_skeleton", "no_object", "no_spatial"]:
        if k not in results:
            continue
        acc       = results[k]["acc"]
        delta     = full_acc - acc
        n         = results[k]["total"]
        delta_str = "--" if k == "full" else f"-{delta:.1%}"
        lines.append(
            f"{ABLATION_LABELS[k]:<26} {acc:>5.1%} {delta_str:>8} {n:>6}")

    with open(save_path, "w") as f:
        f.write("\n".join(lines))
    print(f"[exp3] Saved: {save_path}")
    print("\n".join(lines))


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot-only", action="store_true",
                        help="Skip LLM, read existing ablation collections")
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    client = MongoClient(MONGO_URI)
    db     = client[DB_BASELINE]

    docs = load_docs(db, COL_SEMANTIC)
    if not docs:
        print(f"[exp3] No data in {DB_BASELINE}.{COL_SEMANTIC}")
        return
    print(f"[exp3] {len(docs)} baseline episodes loaded")

    results = {}
    acc, correct, total = compute_accuracy(docs)
    results["full"] = {"acc": acc, "correct": correct, "total": total}
    print(f"[exp3] Full system: {acc:.1%} ({correct}/{total})")

    for mask, col_name in ABLATION_MODES.items():
        if args.plot_only:
            abl_docs = _load_ablation_results(db, col_name)
            if not abl_docs:
                print(f"[exp3] No data in {col_name}, skipping {mask}")
                continue
            acc, correct, total = compute_accuracy(abl_docs)
            results[mask] = {"acc": acc, "correct": correct, "total": total}
            print(f"[exp3] {mask}: {acc:.1%} ({correct}/{total}) [from DB]")
        else:
            print(f"\n[exp3] Running ablation: {mask}")
            preds    = run_ablation_mode(db, docs, mask, col_name)
            abl_docs = []
            for doc, pred in zip(docs, preds):
                d        = dict(doc)
                d["_pred"] = pred
                abl_docs.append(d)
            acc, correct, total = compute_accuracy(abl_docs)
            results[mask] = {"acc": acc, "correct": correct, "total": total}
            print(f"[exp3] {mask}: {acc:.1%} ({correct}/{total})")

    plot_ablation_bar(
        results, os.path.join(RESULTS_DIR, "exp3_ablation_bar.png"))
    plot_ablation_table(
        results, os.path.join(RESULTS_DIR, "exp3_ablation_table.png"))
    save_summary(
        results, os.path.join(RESULTS_DIR, "exp3_summary.txt"))

    print("\n[exp3] Done.")


if __name__ == "__main__":
    main()