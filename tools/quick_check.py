from pymongo import MongoClient

client = MongoClient("mongodb://127.0.0.1:27017/")

print("=== robot_exp_baseline ===")
db = client["robot_exp_baseline"]
candidates_baseline = [
    "experiment_logs",
    "experiment_logs_semantic",
    "experiment_logs_vlm",
    "ablation_no_skeleton",
    "ablation_no_object",
    "ablation_no_spatial",
    "observation_logs",
    "transition_counts",
]
for col in candidates_baseline:
    n = db[col].count_documents({})
    print(f"{col:45} {n} docs")

print("\n=== robot_exp_corruption ===")
db = client["robot_exp_corruption"]
candidates_corruption = [
    "experiment_logs",
    "experiment_logs_corruption_light_semantic",
    "experiment_logs_corruption_medium_semantic",
    "experiment_logs_corruption_heavy_semantic",
]
for col in candidates_corruption:
    n = db[col].count_documents({})
    print(f"{col:45} {n} docs")