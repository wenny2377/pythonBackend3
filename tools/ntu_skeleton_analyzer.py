import os
import re
import json
import math
import argparse
import numpy as np
from collections import defaultdict

# ── Action ID mapping ─────────────────────────────────────────────────────────
# Maps NTU action ID (1-based) to your system's action label
# Only include actions that exist in your system

ACTION_MAP = {
    1:  "Eating",       # A001 eat meal/snack
    3:  "Drinking",     # A003 drink water/juice
    11: "Laying",       # A011 lying down
    22: "Reading",      # A022 writing — proxy for Reading (similar head-down posture)
                        # NTU 120 A098 = reading book (exact), but requires s018-s032
    27: "PhoneUse",     # A027 use a phone
    30: "Typing",       # A030 typing on keyboard
    31: "Sitting",      # A031 sitting down
    45: "Cleaning",     # A045 clean/sweep floor
    # NTU 120 actions (requires nturgbd_skeletons_s018_to_s032.zip)
    49: "Cooking",      # A049 cooking
    60: "Watching",     # A060 watching TV
    # Uncomment if you download s018-s032:
    # 98: "Reading",    # A098 reading book (exact match, overrides A022 proxy)
}

# For SittingDrink, use same stats as Drinking (similar posture)
ALIAS_MAP = {
    "SittingDrink": "Drinking",
}

# ── NTU Joint indices ─────────────────────────────────────────────────────────
J_SPINE_BASE    = 0
J_NECK          = 2
J_HEAD          = 3
J_SHOULDER_R    = 8
J_ELBOW_R       = 9
J_WRIST_R       = 10
J_HAND_R        = 11
J_HIP_R         = 16
J_KNEE_R        = 17
J_ANKLE_R       = 18
J_SHOULDER_L    = 4
J_WRIST_L       = 6
J_HAND_L        = 7

# ── Feature extraction ────────────────────────────────────────────────────────

def body_height(joints: np.ndarray) -> float:
    """Estimate body height from SpineBase to Head."""
    head  = joints[J_HEAD]
    base  = joints[J_SPINE_BASE]
    h     = abs(head[1] - base[1])
    return h if h > 0.3 else -1.0


def head_pitch(joints: np.ndarray) -> float:
    """
    Head pitch angle in degrees.
    Positive = looking down, Negative = looking up.
    Computed from Head relative to Neck direction.
    """
    head = joints[J_HEAD]
    neck = joints[J_NECK]
    vec  = head - neck
    if np.linalg.norm(vec) < 1e-6:
        return 0.0
    # Project onto the sagittal plane (y-z)
    # Angle from vertical (y-axis)
    angle = math.degrees(math.atan2(-vec[2], vec[1]))
    return angle


def hand_to_head_normalized(joints: np.ndarray, bh: float) -> float:
    """
    Normalized distance from right hand to head.
    (distance / body_height)
    """
    if bh <= 0:
        return -1.0
    dist = np.linalg.norm(joints[J_HAND_R] - joints[J_HEAD])
    return dist / bh


def arm_elevation(joints: np.ndarray) -> float:
    """
    Right arm elevation angle in degrees.
    0° = arm pointing straight up, 180° = arm pointing straight down.
    """
    shoulder = joints[J_SHOULDER_R]
    wrist    = joints[J_WRIST_R]
    vec      = wrist - shoulder
    if np.linalg.norm(vec) < 1e-6:
        return 90.0
    vec_norm = vec / np.linalg.norm(vec)
    up       = np.array([0, 1, 0])
    cos_a    = np.clip(np.dot(vec_norm, up), -1, 1)
    return math.degrees(math.acos(cos_a))


def wrist_height_normalized(joints: np.ndarray, bh: float) -> float:
    """
    Right wrist height relative to hip, normalized by body height.
    Positive = above hip, Negative = below hip.
    """
    if bh <= 0:
        return -999.0
    hip_y   = joints[J_SPINE_BASE][1]
    wrist_y = joints[J_WRIST_R][1]
    return (wrist_y - hip_y) / bh


def wrist_z_normalized(joints: np.ndarray, bh: float) -> float:
    """
    Right wrist forward/backward position relative to hip,
    normalized by body height.
    Positive = in front of body.
    """
    if bh <= 0:
        return -999.0
    hip_z   = joints[J_SPINE_BASE][2]
    wrist_z = joints[J_WRIST_R][2]
    return (wrist_z - hip_z) / bh


def extract_features(joints: np.ndarray) -> dict | None:
    """Extract all features from a single frame's joint array."""
    bh = body_height(joints)
    if bh < 0:
        return None

    return {
        "head_pitch":   head_pitch(joints),
        "hand_to_head": hand_to_head_normalized(joints, bh),
        "arm_elevation": arm_elevation(joints),
        "wrist_height": wrist_height_normalized(joints, bh),
        "wrist_z":      wrist_z_normalized(joints, bh),
    }


# ── Skeleton file parser ──────────────────────────────────────────────────────

def parse_skeleton_file(filepath: str) -> list:
    """
    Parse a NTU RGB+D .skeleton file.
    Returns list of per-frame joint arrays (shape: [25, 3]).
    Only returns frames where exactly one body is tracked.
    """
    frames = []
    try:
        with open(filepath, 'r') as f:
            lines = f.read().splitlines()

        idx          = 0
        n_frames     = int(lines[idx]); idx += 1

        for _ in range(n_frames):
            n_bodies = int(lines[idx]); idx += 1

            if n_bodies == 0:
                continue

            # Read first body only (skip multi-person frames)
            # Body header line
            idx += 1  # skip body metadata

            n_joints = int(lines[idx]); idx += 1
            if n_joints != 25:
                idx += n_joints
                # Skip remaining bodies
                for _ in range(n_bodies - 1):
                    idx += 1        # body metadata
                    nj = int(lines[idx]); idx += 1
                    idx += nj
                continue

            joints = np.zeros((25, 3), dtype=np.float32)
            for j in range(25):
                parts     = lines[idx].split(); idx += 1
                joints[j] = [float(parts[0]),
                              float(parts[1]),
                              float(parts[2])]

            # Skip remaining bodies in this frame
            for _ in range(n_bodies - 1):
                idx += 1  # body metadata
                nj = int(lines[idx]); idx += 1
                idx += nj

            if n_bodies == 1:
                frames.append(joints)

    except Exception as e:
        print(f"  [parse error] {filepath}: {e}")

    return frames


# ── Main analysis ─────────────────────────────────────────────────────────────

def get_action_id(filename: str) -> int | None:
    """Extract action ID from NTU filename like S001C001P001R001A001.skeleton"""
    m = re.search(r'A(\d{3})', filename)
    return int(m.group(1)) if m else None


def analyze_dataset(data_dir: str, alpha: float = 0.5) -> dict:
    """
    Main analysis loop.
    Returns statistics per action label.
    """
    # Collect feature samples per action
    samples = defaultdict(lambda: defaultdict(list))

    skeleton_files = [
        f for f in os.listdir(data_dir)
        if f.endswith('.skeleton')
    ]

    if not skeleton_files:
        print(f"[error] No .skeleton files found in {data_dir}")
        return {}

    print(f"Found {len(skeleton_files)} skeleton files")

    processed = 0
    for filename in sorted(skeleton_files):
        action_id = get_action_id(filename)
        if action_id not in ACTION_MAP:
            continue

        label    = ACTION_MAP[action_id]
        filepath = os.path.join(data_dir, filename)
        frames   = parse_skeleton_file(filepath)

        for joints in frames:
            feat = extract_features(joints)
            if feat is None:
                continue
            for key, val in feat.items():
                if val is not None and val > -990:
                    samples[label][key].append(val)

        processed += 1
        if processed % 50 == 0:
            print(f"  Processed {processed} files...")

    print(f"\nProcessed {processed} files across {len(samples)} actions")

    # Compute statistics
    stats = {}
    feature_order = [
        "head_pitch", "spine_angle", "arm_elevation",
        "hand_to_head", "wrist_height", "wrist_z",
        "hip_height", "knee_height",
    ]

    for label, feat_samples in samples.items():
        stats[label] = {}
        n_frames = len(feat_samples.get("head_pitch", []))
        print(f"\n  {label} ({n_frames} frames):")

        for feat in feature_order:
            vals = feat_samples.get(feat, [])
            if not vals:
                stats[label][feat] = {"mu": 0.0, "sigma": 0.0, "n": 0}
                continue

            arr   = np.array(vals)
            mu    = float(np.mean(arr))
            sigma = float(np.std(arr))
            stats[label][feat] = {
                "mu":    round(mu, 4),
                "sigma": round(sigma, 4),
                "n":     len(vals),
            }
            print(f"    {feat:<18} μ={mu:+7.3f}  σ={sigma:.3f}  n={len(vals)}")

    # Add aliases
    for alias, source in ALIAS_MAP.items():
        if source in stats:
            stats[alias] = stats[source].copy()
            print(f"\n  {alias} ← copied from {source}")

    return stats, alpha


def generate_noise_params(stats: dict, alpha: float) -> dict:
    """
    Apply alpha scaling to get final noise parameters.
    sigma_corrupt = alpha * sigma_NTU
    """
    noise_params = {}
    feature_order = [
        "head_pitch", "spine_angle", "arm_elevation",
        "hand_to_head", "wrist_height", "wrist_z",
        "hip_height", "knee_height",
    ]

    for label, feat_stats in stats.items():
        noise_params[label] = {}
        for feat in feature_order:
            sigma_ntu   = feat_stats.get(feat, {}).get("sigma", 0.0)
            sigma_noise = round(alpha * sigma_ntu, 4)
            noise_params[label][feat] = sigma_noise

    return noise_params


def generate_csharp(noise_params: dict) -> str:
    """
    Generate C# code for SkeletonHelper.cs INTRA_CLASS_STD dictionary.
    Paste into SkeletonHelper.cs to replace the placeholder zeros.
    """
    feature_order = [
        "head_pitch", "spine_angle", "arm_elevation",
        "hand_to_head", "wrist_height", "wrist_z",
        "hip_height", "knee_height",
    ]

    lines = [
        "// Auto-generated by tools/ntu_skeleton_analyzer.py",
        "// Source: NTU RGB+D (Shahroudy et al., 2016)",
        "// Noise = alpha * sigma_NTU, alpha=0.5",
        "// (MediaPipe Pose noise / natural variation ratio)",
        "//",
        "// Format: [pitch, spine, arm, h2h, wrist_h, wrist_z, hip, knee]",
        "",
        "static readonly Dictionary<string, float[]> INTRA_CLASS_STD =",
        "    new Dictionary<string, float[]>",
        "{",
    ]

    action_labels = [
        "Eating", "Drinking", "SittingDrink", "Reading", "PhoneUse",
        "Typing", "Cleaning", "Cooking", "Watching", "Laying",
        "Sitting", "Opening", "Standing",
    ]

    for label in action_labels:
        params = noise_params.get(label, {})
        vals   = []
        for feat in feature_order:
            v = params.get(feat, 0.0)
            vals.append(f"{v:.4f}f")
        vals_str = ", ".join(vals)
        lines.append(
            f'    {{ "{label}", new float[]{{ {vals_str} }} }},')

    lines.append("};")

    return "\n".join(lines)


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Extract skeleton noise parameters from NTU RGB+D")
    parser.add_argument(
        "--data_dir", type=str, required=True,
        help="Path to NTU RGB+D nturgb+d_skeletons/ directory")
    parser.add_argument(
        "--output", type=str, default="ntu_skeleton_stats.json",
        help="Output JSON path for full statistics")
    parser.add_argument(
        "--alpha", type=float, default=0.5,
        help="Noise scaling factor: sigma_corrupt = alpha * sigma_NTU (default 0.5)")
    args = parser.parse_args()

    if not os.path.isdir(args.data_dir):
        print(f"[error] Directory not found: {args.data_dir}")
        return

    print("=" * 60)
    print("NTU RGB+D Skeleton Analyzer")
    print(f"  data_dir : {args.data_dir}")
    print(f"  alpha    : {args.alpha}")
    print("=" * 60)

    # Run analysis
    stats, alpha = analyze_dataset(args.data_dir, args.alpha)
    if not stats:
        return

    # Save full statistics
    with open(args.output, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\n[ok] Full statistics saved: {args.output}")

    # Generate noise parameters
    noise_params = generate_noise_params(stats, alpha)
    noise_path   = "skeleton_noise_params.json"
    with open(noise_path, "w") as f:
        json.dump(noise_params, f, indent=2)
    print(f"[ok] Noise parameters saved: {noise_path}")

    # Generate C# code
    csharp_code = generate_csharp(noise_params)
    csharp_path = "SkeletonHelper_INTRA_CLASS_STD.txt"
    with open(csharp_path, "w") as f:
        f.write(csharp_code)
    print(f"[ok] C# code saved: {csharp_path}")

    # Print summary table
    print("\n" + "=" * 60)
    print("Noise parameters (sigma_corrupt = alpha * sigma_NTU):")
    print(f"{'Action':<16} {'pitch':>7} {'arm':>7} {'h2h':>7} {'wrist_h':>9} {'wrist_z':>9}")
    print("-" * 60)

    feature_order = [
        "head_pitch", "spine_angle", "arm_elevation",
        "hand_to_head", "wrist_height", "wrist_z",
        "hip_height", "knee_height",
    ]

    for label in [
        "Eating", "Drinking", "SittingDrink", "Reading",
        "PhoneUse", "Typing", "Cleaning", "Watching", "Laying", "Sitting"
    ]:
        p = noise_params.get(label, {})
        print(
            f"{label:<16} "
            f"{p.get('head_pitch', 0):>7.4f} "
            f"{p.get('arm_elevation', 0):>7.4f} "
            f"{p.get('hand_to_head', 0):>7.4f} "
            f"{p.get('wrist_height', 0):>9.4f} "
            f"{p.get('wrist_z', 0):>9.4f}"
        )

    print("=" * 60)
    print("\nNext step:")
    print(f"  1. Open SkeletonHelper.cs")
    print(f"  2. Find INTRA_CLASS_STD dictionary")
    print(f"  3. Replace placeholder zeros with values from {csharp_path}")
    print(f"  4. Set skeletonNoiseEnabled = true in ExperimentRunner Inspector")


if __name__ == "__main__":
    main()