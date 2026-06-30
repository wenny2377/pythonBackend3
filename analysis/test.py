import os
from pymongo import MongoClient

MONGO_URI = "mongodb://127.0.0.1:27017/"
DB_NAME   = os.environ.get("DB_NAME", "robot_exp_baseline")

COL_A = "experiment_logs_semantic"
COL_B = "experiment_logs_vlm_som"


def main():
    client = MongoClient(MONGO_URI)
    db     = client[DB_NAME]

    docs_a = list(db[COL_A].find({}, {
        "episode_id": 1, "user": 1, "ground_truth": 1, "spatial_action": 1,
        "vlm_output": 1, "vlm_scene_desc": 1, "vlm_key_object": 1,
        "vlm_confidence": 1, "vlm_timed_out": 1, "upgrade_reason": 1,
        "experiment_mode": 1, "system_mode": 1, "collection_suffix": 1,
        "virtual_day": 1, "virtual_hour": 1, "timestamp": 1,
    }))
    docs_b = list(db[COL_B].find({}, {
        "episode_id": 1, "user": 1, "ground_truth": 1, "spatial_action": 1,
        "vlm_output": 1, "vlm_scene_desc": 1, "vlm_key_object": 1,
        "vlm_confidence": 1, "vlm_timed_out": 1, "upgrade_reason": 1,
        "experiment_mode": 1, "system_mode": 1, "collection_suffix": 1,
        "virtual_day": 1, "virtual_hour": 1, "timestamp": 1,
    }))

    print(f"DB={DB_NAME}")
    print(f"{COL_A}: {len(docs_a)} docs")
    print(f"{COL_B}: {len(docs_b)} docs")
    print("=" * 90)

    # check 1: do the two collections share identical episode_id values?
    ids_a = {d.get("episode_id", "") for d in docs_a if d.get("episode_id")}
    ids_b = {d.get("episode_id", "") for d in docs_b if d.get("episode_id")}
    overlap = ids_a & ids_b
    print(f"\n[Check 1] episode_id overlap between collections: {len(overlap)}")
    if overlap:
        print("  -> If this is non-zero, the two collections share actual episodes,")
        print("     meaning they are NOT independent runs.")
        for eid in list(overlap)[:5]:
            print(f"     shared episode_id: {eid}")

    # check 2: system_mode field stored on each doc -- should be 'semantic' for A, 'vlm_som' for B
    modes_a = {}
    modes_b = {}
    for d in docs_a:
        m = d.get("system_mode", "<missing>")
        modes_a[m] = modes_a.get(m, 0) + 1
    for d in docs_b:
        m = d.get("system_mode", "<missing>")
        modes_b[m] = modes_b.get(m, 0) + 1
    print(f"\n[Check 2] system_mode values stored in {COL_A}: {modes_a}")
    print(f"[Check 2] system_mode values stored in {COL_B}: {modes_b}")
    print("  -> Expect COL_A all 'semantic', COL_B all 'vlm_som'.")
    print("     If COL_B shows 'semantic', the VLM branch was never actually triggered")
    print("     for that collection.")

    # check 3: are vlm_scene_desc / vlm_key_object non-empty in B but empty in A?
    nonempty_vlm_a = sum(1 for d in docs_a if d.get("vlm_scene_desc", "").strip())
    nonempty_vlm_b = sum(1 for d in docs_b if d.get("vlm_scene_desc", "").strip())
    nonnone_keyobj_a = sum(1 for d in docs_a if d.get("vlm_key_object", "none") != "none")
    nonnone_keyobj_b = sum(1 for d in docs_b if d.get("vlm_key_object", "none") != "none")
    print(f"\n[Check 3] Non-empty vlm_scene_desc: {COL_A}={nonempty_vlm_a}/{len(docs_a)}  "
          f"{COL_B}={nonempty_vlm_b}/{len(docs_b)}")
    print(f"[Check 3] Non-'none' vlm_key_object: {COL_A}={nonnone_keyobj_a}/{len(docs_a)}  "
          f"{COL_B}={nonnone_keyobj_b}/{len(docs_b)}")
    print("  -> Expect COL_A near 0 (VLM skipped in semantic mode), COL_B mostly non-zero")
    print("     (VLM actually ran). If both are near 0, VLM never ran for either collection.")

    # check 4: vlm_timed_out / vlm_confidence distribution in B
    timed_out_b = sum(1 for d in docs_b if d.get("vlm_timed_out"))
    conf_zero_b = sum(1 for d in docs_b if d.get("vlm_confidence", -1) == 0.0)
    print(f"\n[Check 4] {COL_B}: vlm_timed_out=True count: {timed_out_b}/{len(docs_b)}")
    print(f"[Check 4] {COL_B}: vlm_confidence==0.0 count: {conf_zero_b}/{len(docs_b)}")
    print("  -> If vlm_confidence is 0.0 for nearly all docs, VLM calls may have failed")
    print("     silently or the field was never populated.")

    # check 5: compare spatial_action field-by-field for episodes with matching (user, virtual_day, virtual_hour)
    def _key(d):
        return (d.get("user", ""), d.get("virtual_day"), d.get("virtual_hour"))

    map_a = {_key(d): d for d in docs_a}
    map_b = {_key(d): d for d in docs_b}
    common_keys = set(map_a.keys()) & set(map_b.keys())
    same_pred = 0
    diff_pred = 0
    for k in common_keys:
        pa = map_a[k].get("spatial_action", "")
        pb = map_b[k].get("spatial_action", "")
        if pa == pb:
            same_pred += 1
        else:
            diff_pred += 1
    print(f"\n[Check 5] Matched (user, virtual_day, virtual_hour) pairs: {len(common_keys)}")
    print(f"[Check 5] Same spatial_action in both: {same_pred}")
    print(f"[Check 5] Different spatial_action: {diff_pred}")
    print("  -> If 'same' is very high relative to total, the two collections are producing")
    print("     near-identical predictions despite different system_mode, which is the")
    print("     anomaly to explain.")

    # check 6: timestamps -- were the two collections actually written at different times?
    ts_a = sorted(d.get("timestamp") for d in docs_a if d.get("timestamp"))
    ts_b = sorted(d.get("timestamp") for d in docs_b if d.get("timestamp"))
    if ts_a and ts_b:
        print(f"\n[Check 6] {COL_A} timestamp range: {ts_a[0]} to {ts_a[-1]}")
        print(f"[Check 6] {COL_B} timestamp range: {ts_b[0]} to {ts_b[-1]}")
        print("  -> If these ranges are identical or heavily overlapping, double-check")
        print("     whether both experiments actually ran as separate Unity passes.")


if __name__ == "__main__":
    main()