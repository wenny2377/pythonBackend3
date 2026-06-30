import os

MONGO_URI     = "mongodb://127.0.0.1:27017/"
DB_BASELINE   = "robot_exp_baseline"
DB_CORRUPTION = "robot_exp_corruption"
BACKEND_URL   = "http://127.0.0.1:5000"
OLLAMA_URL    = "http://localhost:11434"
LLM_MODEL     = "llama3.1:8b"

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

# ─── collections ──────────────────────────────────────────────────────────────
# These MUST match the collectionSuffix values Unity's ExperimentRunner.AllSchedule
# actually sends, which app.py/PerceptionEngine turn into f"experiment_logs{suffix}":
#   Baseline / systemMode=semantic  -> collectionSuffix="_semantic"
#       -> experiment_logs_semantic                  (DB_BASELINE)
#   Baseline / systemMode=vlm_som   -> collectionSuffix="_vlm_som"
#       -> experiment_logs_vlm_som                    (DB_BASELINE)
#   CorruptionLight                 -> collectionSuffix="_corruption_light_semantic"
#       -> experiment_logs_corruption_light_semantic   (DB_CORRUPTION)
#   CorruptionMedium                -> collectionSuffix="_corruption_medium_semantic"
#       -> experiment_logs_corruption_medium_semantic  (DB_CORRUPTION)
#   CorruptionHeavy                 -> collectionSuffix="_corruption_heavy_semantic"
#       -> experiment_logs_corruption_heavy_semantic   (DB_CORRUPTION)
#
# PREVIOUS BUG: COL_SEMANTIC pointed at the bare "experiment_logs" (no suffix),
# and the three corruption collections were missing the trailing "_semantic".
# Both caused exp1/exp2/exp3 to silently read 0 episodes from collections that
# were never actually written to, while the real data sat in the correctly-
# suffixed collections above. Fixed below.
COL_BASELINE          = "experiment_logs_semantic"
COL_SEMANTIC          = "experiment_logs_semantic"
COL_VLM_SOM           = "experiment_logs_vlm_som"
COL_CORRUPTION_LIGHT  = "experiment_logs_corruption_light_semantic"
COL_CORRUPTION_MEDIUM = "experiment_logs_corruption_medium_semantic"
COL_CORRUPTION_HEAVY  = "experiment_logs_corruption_heavy_semantic"

# These four are NOT written by Unity/PerceptionEngine directly — they are
# created by exp3_modality_ablation.py itself (run_ablation_mode) during its
# own offline re-inference pass. Confirmed consistent: exp3 imports
# COL_ABL_NO_SKELETON/NO_OBJECT/NO_SPATIAL and writes to exactly these names.
# COL_ABL_NO_VLM is defined for symmetry with PerceptionEngine's online
# ablation modes (no_skeleton/no_object/no_vlm) but is NOT currently used by
# exp3_modality_ablation.py, which implements its own offline masking
# (no_skeleton/no_object/no_spatial) — a deliberately different set, since
# masking "VLM input" has no meaning when re-inferring from already-captured
# text. Kept here in case a future online VLM-ablation analysis script needs it.
COL_ABL_NO_SKELETON   = "ablation_no_skeleton"
COL_ABL_NO_VLM        = "ablation_no_vlm"
COL_ABL_NO_OBJECT     = "ablation_no_object"
COL_ABL_NO_SPATIAL    = "ablation_no_spatial"

# NOTE: this list currently has 11 entries. PerceptionEngine's BEHAVIOR_LABELS
# has 17 total; excluding the 5 transition/no-weight actions
# (PickingUp, PuttingDown, Walking, Standing, StandUp) leaves 12, not 11 —
# SeatedDrinking is missing from this list below. If SeatedDrinking episodes
# exist in your logs, they will silently fall outside ADL_LABELS and be
# excluded from every per-class accuracy table and from compute_accuracy()'s
# denominator (since compute_accuracy only counts docs where
# gt in ADL_LABELS). Confirm whether this exclusion is intentional; if not,
# add "SeatedDrinking" back to this list.
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
    "baseline":          "#2196F3",
    "corruption_light":  "#FF9800",
    "corruption_medium": "#F44336",
    "corruption_heavy":  "#B71C1C",
    "pass":              "#4CAF50",
    "warn":              "#FF9800",
    "mom":               "#E91E63",
    "dad":               "#1976D2",
    "threshold":         "#9E9E9E",
    "ablation":          "#7E57C2",
    "highlight":         "#FF5722",
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