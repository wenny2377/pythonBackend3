import os

# ── MongoDB ───────────────────────────────────────────────────────────────────
MONGO_URI     = "mongodb://127.0.0.1:27017/"
DB_BASELINE   = "robot_exp_baseline"
DB_CORRUPTION = "robot_exp_corruption"
BACKEND_URL   = "http://127.0.0.1:5000"

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

# ── ADL Labels ────────────────────────────────────────────────────────────────
ADL_LABELS = [
    "Drinking", "SittingDrink", "Sitting", "Eating", "Cooking",
    "Laying", "Watching", "Reading", "Cleaning", "PhoneUse", "Typing",
]

# Opening / StandUp are captured but not evaluated
ALL_BEHAVIORS = ADL_LABELS + ["Opening", "StandUp",
    "Standing", "Walking", "PickingUp", "PuttingDown"]

USERS = ["User_Mom", "User_Dad"]

# ── Color palette (used across all plots) ────────────────────────────────────
C = {
    "baseline":   "#2196F3",   # blue  — baseline / positive result
    "corruption": "#F44336",   # red   — corruption / negative result
    "pass":       "#4CAF50",   # green — above threshold
    "warn":       "#FF9800",   # orange — below threshold
    "mom":        "#E91E63",   # pink  — User_Mom
    "dad":        "#1976D2",   # dark blue — User_Dad
    "threshold":  "#9E9E9E",   # grey  — reference lines
    "ablation":   "#7E57C2",   # purple — ablation bars
    "highlight":  "#FF5722",   # deep orange — key result callout
}

# ── Plot style ────────────────────────────────────────────────────────────────
FONT_TITLE  = 13
FONT_AXIS   = 11
FONT_TICK   = 10
FONT_ANNOT  = 9
LINE_WIDTH  = 2.0
MARKER_SIZE = 7
FIG_DPI     = 200

def apply_style():
    """Call once at the top of each script."""
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

# ── Ground-truth habits (from ExperimentRunner.cs TimeSlots) ─────────────────
# Format: (user, action, time_slot)
# These are the "stable" behaviors designed into the experiment.
GT_HABITS = [
    # User_Mom
    ("User_Mom", "Eating",       "Morning"),
    ("User_Mom", "Drinking",     "Morning"),
    ("User_Mom", "Cleaning",     "Morning"),
    ("User_Mom", "Reading",      "Noon"),
    ("User_Mom", "Laying",       "Noon"),
    ("User_Mom", "Reading",      "Afternoon"),
    ("User_Mom", "Cleaning",     "Afternoon"),
    ("User_Mom", "SittingDrink", "Afternoon"),
    ("User_Mom", "Watching",     "Evening"),
    ("User_Mom", "Eating",       "Evening"),
    ("User_Mom", "Laying",       "Night"),
    ("User_Mom", "Reading",      "Night"),
    # User_Dad
    ("User_Dad", "Typing",       "Morning"),
    ("User_Dad", "Eating",       "Morning"),
    ("User_Dad", "Drinking",     "Morning"),
    ("User_Dad", "Laying",       "Noon"),
    ("User_Dad", "Watching",     "Noon"),
    ("User_Dad", "Typing",       "Afternoon"),
    ("User_Dad", "PhoneUse",     "Afternoon"),
    ("User_Dad", "PhoneUse",     "Evening"),
    ("User_Dad", "SittingDrink", "Evening"),
    ("User_Dad", "Watching",     "Evening"),
    ("User_Dad", "Laying",       "Night"),
    ("User_Dad", "PhoneUse",     "Night"),
]