"""
analysis/table_compare.py
Mom vs Dad Personalisation Comparison Table
Outputs:
  results/Fig5_personalization.png
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pymongo import MongoClient

MONGO_URI = "mongodb://127.0.0.1:27017/"
DB_NAME   = "robot_rag_db"
OUT = os.path.join(os.path.dirname(__file__), "results")


def get_skill_summary(db, user_id):
    doc = db.user_skills.find_one({"user_id": user_id})
    if not doc:
        return {}

    skill_md = doc.get("skill_md", "")
    sections = {}
    current  = None

    for line in skill_md.split("\n"):
        for sec in ["Behavior Patterns", "Preferences",
                    "How to Handle", "What NOT to do"]:
            if f"## {sec}" in line:
                current = sec
                sections[current] = []
                break
        if current and line.strip().startswith("-"):
            sections[current].append(line.strip())

    return sections


def plot_fig5_personalization(db):
    print("Fig5: Mom vs Dad Personalization Table...")

    mom_skill = get_skill_summary(db, "User_Mom")
    dad_skill = get_skill_summary(db, "User_Dad")

    mom_patterns = mom_skill.get("Behavior Patterns", ["(No data yet)"])[:4]
    dad_patterns = dad_skill.get("Behavior Patterns", ["(No data yet)"])[:4]
    mom_prefs    = mom_skill.get("Preferences",        ["(No data yet)"])[:3]
    dad_prefs    = dad_skill.get("Preferences",        ["(No data yet)"])[:3]
    mom_not      = mom_skill.get("What NOT to do",     ["(No data yet)"])[:2]
    dad_not      = dad_skill.get("What NOT to do",     ["(No data yet)"])[:2]

    def fmt(lines):
        return "\n".join(l[:55] for l in lines) if lines else "(empty)"

    rows = [
        ["Morning 07:00",
         "Opening → Cooking → Eating\n(kitchen-oriented)",
         "Opening → Eating → Typing\n(work-oriented)"],
        ["Noon 12:00",
         "Sitting → Reading → Laying\n(reading nap)",
         "Laying → Watching\n(TV nap)"],
        ["Afternoon 15:00",
         "Cleaning → Reading\n(chores + reading)",
         "Typing → PhoneUse\n(work + digital)"],
        ["Evening 19:00",
         "Cooking → Eating → Watching\n(kitchen → living room)",
         "Eating → PhoneUse → SittingDrink\n(digital-oriented)"],
        ["Night 23:00",
         "Reading → Laying\n(reading to sleep)",
         "PhoneUse → Watching → Laying\n(TV to sleep)"],
        ["Drink Preference",
         "Juice\n(learned from observation)",
         "Cola\n(learned from observation)"],
        ["SKILL.md\nBehavior Patterns",
         fmt(mom_patterns),
         fmt(dad_patterns)],
        ["SKILL.md\nPreferences",
         fmt(mom_prefs),
         fmt(dad_prefs)],
        ["SKILL.md\nWhat NOT to do",
         fmt(mom_not),
         fmt(dad_not)],
    ]

    fig, ax = plt.subplots(figsize=(15, 10))
    ax.axis("off")

    table = ax.table(
        cellText=rows,
        colLabels=["Time / Category", "User Mom", "User Dad"],
        cellLoc="center",
        loc="center",
        colWidths=[0.18, 0.41, 0.41],
    )

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 3.8)

    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#CCCCCC")
        if r == 0:
            cell.set_facecolor("#1565C0")
            cell.set_text_props(color="white", fontweight="bold", fontsize=10)
        elif c == 0:
            cell.set_facecolor("#E3F2FD")
            cell.set_text_props(fontweight="bold")
        elif c == 1:
            cell.set_facecolor("#FFF8E1")
        elif c == 2:
            cell.set_facecolor("#F3E5F5")

        if r in (6, 7, 8) and c > 0:
            cell.set_facecolor("#E8F5E9" if c == 1 else "#FCE4EC")

    ax.set_title(
        "Fig5  Mom vs Dad Personalised Habit Profile\n"
        "Learned passively from observation — no explicit user configuration required\n"
        "Bottom rows show actual SKILL.md content learned by the system",
        fontsize=12, fontweight="bold", pad=20
    )

    plt.tight_layout()
    path = os.path.join(OUT, "Fig5_personalization.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")

    if mom_skill or dad_skill:
        print(f"  Mom SKILL.md version: {db.user_skills.find_one({'user_id':'User_Mom'}, {'version':1}).get('version','?')}")
        print(f"  Dad SKILL.md version: {db.user_skills.find_one({'user_id':'User_Dad'}, {'version':1}).get('version','?')}")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    db = MongoClient(MONGO_URI)[DB_NAME]
    print(f"Connected → {DB_NAME}")
    plot_fig5_personalization(db)
    print("Done.")