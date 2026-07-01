import os

MONGO_URI     = "mongodb://127.0.0.1:27017/"
DB_BASELINE   = "robot_exp_baseline"
DB_CORRUPTION = "robot_exp_corruption"
BACKEND_URL   = "http://127.0.0.1:5000"
OLLAMA_URL    = "http://localhost:11434"
LLM_MODEL     = "llama3.1:8b"

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

COL_BASELINE          = "experiment_logs_semantic"
COL_SEMANTIC          = "experiment_logs_semantic"
COL_VLM               = "experiment_logs_vlm"
COL_CORRUPTION_LIGHT  = "experiment_logs_corruption_light_semantic"
COL_CORRUPTION_MEDIUM = "experiment_logs_corruption_medium_semantic"
COL_CORRUPTION_HEAVY  = "experiment_logs_corruption_heavy_semantic"

COL_ABL_NO_SKELETON   = "ablation_no_skeleton"
COL_ABL_NO_VLM        = "ablation_no_vlm"
COL_ABL_NO_OBJECT     = "ablation_no_object"
COL_ABL_NO_SPATIAL    = "ablation_no_spatial"

ADL_LABELS = [
    "Drinking", "Sitting", "Eating", "Cooking",
    "Laying", "Watching", "Reading", "Cleaning",
    "UsingPhone", "Typing", "Opening",
]

USERS = ["User_Mom", "User_Dad"]

GT_NORMALIZE_MAP = {
    "seateddrinking": "Drinking",
    "dadreading":     "Reading",
    "dadcleaning":    "Cleaning",
    "dadphone":       "UsingPhone",
}

ROOM_IMPOSSIBLE = {
    "DadRoom":    {"Cooking", "Cleaning"},
    "Kitchen":    {"Laying", "Typing"},
    "LivingRoom": {"Typing", "Cooking"},
}

C = {
    # 系統角色 — 跨所有實驗保持一致
    "baseline":          "#4C9BE8",   # 藍，System A / Full System / Baseline
    "vlm":               "#888888",   # 灰，System B / VLM
    "mom":               "#E8507A",   # 粉紅，User_Mom
    "dad":               "#5B8DB8",   # 鋼藍，User_Dad

    # Corruption 程度 — 藍→橘→橘紅→紅，越重越暖
    "corruption_light":  "#F5A623",   # 橘，Light
    "corruption_medium": "#E8734C",   # 橘紅，Medium
    "corruption_heavy":  "#D94F3D",   # 紅，Heavy

    # Ablation 專用
    "ablation":          "#7B68EE",   # 紫，ablation 比較用（w/o Spatial）

    # 強調 / 警示
    "highlight":         "#D94F3D",   # 紅橘，誤判 / 警示數字
    "pass":              "#4CAF50",   # 綠，通過/正常（保留備用）
    "threshold":         "#9E9E9E",   # 灰，門檻線
}

FONT_TITLE  = 13
FONT_AXIS   = 11
FONT_TICK   = 10
FONT_ANNOT  = 9
LINE_WIDTH  = 2.0
MARKER_SIZE = 7
FIG_DPI     = 200


def apply_style():
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family":       "DejaVu Sans",
        "font.size":         FONT_TICK,
        "axes.titlesize":    FONT_TITLE,
        "axes.labelsize":    FONT_AXIS,
        "xtick.labelsize":   FONT_TICK,
        "ytick.labelsize":   FONT_TICK,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.grid":         True,
        "grid.alpha":        0.25,
        "figure.dpi":        FIG_DPI,
    })


def normalize_gt(label: str) -> str:
    if not label:
        return label
    key = label.lower().strip().replace(" ", "").replace("_", "")
    return GT_NORMALIZE_MAP.get(key, label)


def load_docs(db, collection: str) -> list:
    docs = list(db[collection].find(
        {"ground_truth": {"$exists": True, "$ne": ""},
         "spatial_action": {"$exists": True}},
    ))
    for d in docs:
        d["ground_truth"] = normalize_gt(d.get("ground_truth", ""))
        pred = d.get("spatial_action") or d.get("vlm_output", "")
        d["_pred"] = normalize_gt(pred)
    return docs


def compute_accuracy(docs) -> tuple:
    total = correct = 0
    for d in docs:
        gt = d.get("ground_truth", "")
        if gt in ADL_LABELS:
            total   += 1
            correct += int(gt == d.get("_pred", ""))
    return (correct / total if total else 0.0), correct, total