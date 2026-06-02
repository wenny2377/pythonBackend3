# Home Behaviour Recognition Engine

A multimodal home activity recognition system that fuses visual language models, skeleton posture estimation, spatial geometry, and personalised habit learning to recognise 17 daily living activities from fixed-corner surveillance cameras.

---

## Overview

This system addresses the challenge of distinguishing visually similar activities (e.g. Sitting vs Watching, Sitting vs Typing) under real surveillance camera conditions: fixed corners, 15–30° tilt, partial occlusion, and low resolution.

The core contribution is a **Neuro-Symbolic Weighted Scoring Architecture** that combines:

- Physical constraint rules (BODY_IMPOSSIBLE hard filter)
- Skeleton-based posture classification (normalised hip height + head pitch)
- Charades-v1 corpus furniture–behaviour affinity statistics
- Dynamic object tracking (item-to-action inference)
- VLM single-frame semantic understanding (gemma3:4b)
- Personalised habit learning (ManifoldEngine MLP)

---

## Architecture

```
Unity Simulation Layer
  ├── SkeletonHelper.cs       Normalised hip height + head pitch (Gaussian noise, σ=0.02/5°)
  ├── DynamicSyncManager.cs   Object tracking (held_by, sensor_pos, 0.5s/2s intervals)
  ├── StaticCameraManager.cs  Still-detection trigger (v < 0.05 m/s, 1.5s) + multi-view selection
  └── VirtualCameraBrain.cs   Multi-burst capture (topN=2, angle > 30°, 2 burst frames each)
         │
         │ POST /predict (4 images + skeleton payload + user_pos)
         ▼
Flask Backend
  ├── PerceptionEngine        Weighted scoring engine (8 evidence layers)
  │     ├── Layer 0: BODY_IMPOSSIBLE      Physical constraint hard filter
  │     ├── Layer 1: HEAD_PITCH_PROFILE   W=0.45, per-behaviour Gaussian scoring
  │     ├── Layer X: held_object          W=0.55, item-to-action mapping
  │     ├── Layer X2: nearby objects      W=0.25, proximity-based inference
  │     ├── Layer 3: Proximity affinity   W=0.30, Charades-v1 furniture statistics
  │     ├── Layer 4: Ray-cast             W=0.25, directional gaze toward TV/stove/fridge
  │     ├── Layer 5: Zone affinity        W=0.25, semantic zone scoring
  │     ├── Layer 6: VLM hint             W=0.20, gemma3:4b weighted vote
  │     ├── Layer 7: Time prior           W=0.10, hour-based behavioural prior
  │     └── Layer 8: Temporal inertia     W=0.10, Markov transition matrix
  ├── SceneEngine             Zone discovery with importance-weighted anchor selection
  ├── HabitEngine             FAT-based personalised zone affinity learning
  ├── ManifoldEngine          21-dim MLP for personalised behaviour intent prediction
  └── SayCanEngine            Say × Can fusion for robot service recommendation
         │
         ▼
MongoDB
  ├── eval_logs               Recognition experiment results
  ├── observation_logs        Habit learning data
  ├── affinity_matrix         Charades-v1 + builtin furniture affinity
  ├── transition_matrix       Behaviour transition probabilities
  └── manifold_training_data  Per-user MLP training samples
```

---

## Behaviour Labels (17)

| Category | Behaviours |
|---|---|
| High-specificity | Cooking, Opening, Laying, Watching, Typing, Cleaning |
| Medium | Eating, Drinking, Standing, Walking, StandUp |
| Low (visually ambiguous) | SittingDrink, Sitting, Reading, PhoneUse |
| Transition | PickingUp, PuttingDown |

---

## Key Design Decisions

### Why Weighted Scoring over Cascade

A cascade (first-match-wins) silently overrides weaker evidence. Weighted fusion makes all evidence compete explicitly, enabling full score traceability in `upgrade_reason` field.

### Why Charades-v1 Affinity over SBERT

SBERT computes semantic similarity, not behavioural co-occurrence frequency. Charades-v1 statistics reflect real domestic activity distributions. For example, SBERT gives `chair → Sitting = 1.00, Eating = 0.25`, whereas Charades statistics correctly reflect that kitchen chairs are primarily associated with Eating.

### Why hip_height Dominates body_position

Fixed-corner cameras (15–30° tilt) cause VLM to misclassify forward-leaning standing postures (Opening, Cleaning) as `sitting`, triggering BODY_IMPOSSIBLE false positives. The normalised hip height ratio simulates MediaPipe Pose `(hip_y / body_height)` which is view-invariant and achieves >95% sitting/standing/lying classification accuracy.

### Multi-view Camera Selection

Based on Weinland et al. (2011) and Shan & Akella (2014): select topN=2 cameras with angular diversity > 30° to ensure complementary occlusion coverage. 2 cameras × 2 burst frames = 4 images per inference call.

### Still-detection Capture Trigger

Replaces per-behaviour settle times with a unified kinematic threshold (speed < 0.05 m/s for 1.5s), matching real surveillance camera operation where the system has no prior knowledge of current activity.

---

## Experiment Results

### RecognitionExp (425 episodes, 11 behaviours × ~20 samples each)

| Group | Accuracy |
|---|---|
| High-specificity (Cooking, Opening, Laying, Cleaning, PhoneUse) | 100.0% |
| Medium (Eating, Drinking) | ~50–80% |
| Low / Visually ambiguous (SittingDrink, Sitting, Reading, Typing) | ~47–88% |
| **Overall** | **66.6%** |

### Ablation Study (Fig.C)

| Config | Accuracy | Delta |
|---|---|---|
| VLM Only (Baseline) | 21.9% | — |
| + Skeleton (hip+head) | 65.6% | +43.7% |
| + Geometry (affinity+ray) | 62.5% | −3.1%* |
| + Object Context (held+nearby) | 62.5% | +0.0% |
| Full System (+temporal) | 62.5% | +0.0% |

*Geometry layer shows negative delta when BODY_IMPOSSIBLE interacts with VLM body_position misclassification. Resolved by restoring skeleton-authoritative body_position.

---

## Real Deployment Mapping

| Unity Simulation | Real Deployment | Notes |
|---|---|---|
| hip_height (Animator + σ=0.02 noise) | MediaPipe Pose landmark[23,24] normalised | One-time per-user calibration |
| head_pitch (Animator + σ=5° noise) | MediaPipe Face Mesh pitch estimation | Interface identical |
| user_pos (Unity world coords) | Camera calibration back-projection | One-time spatial calibration |
| user_forward (transform.forward) | Shoulder keypoint direction | Interface identical |
| DynamicSyncManager | Co-located object detection (YOLO etc.) | Output format alignment required |

---

## Tech Stack

| Component | Technology |
|---|---|
| Simulation | Unity 2022 (C#), NavMesh, Humanoid Animator |
| Backend | Python 3.10, Flask |
| VLM | gemma3:4b via Ollama |
| LLM | llama3.1:8b via Ollama |
| Embedding | SBERT all-MiniLM-L6-v2 |
| Vector search | FAISS |
| Database | MongoDB |
| ML | PyTorch (ManifoldEngine MLP) |
| Activity corpus | Charades-v1 (9,848 videos) |

---

## Project Structure

```
robotBrain/
├── app.py                      Flask entry point
├── modules/
│   ├── perception_engine.py    Weighted scoring engine (core)
│   ├── scene_engine.py         Zone discovery + affinity matrix
│   ├── habit_engine.py         FAT-based habit learning
│   ├── manifold_engine.py      Per-user MLP intent prediction
│   ├── saycan_engine.py        Robot service recommendation
│   └── classifier.py           Behaviour classifier utilities
├── config/
│   └── robot_ontology.yaml     Behaviour constraints + item-to-action mapping
├── charades_data/
│   └── build_charades_matrix.py  Transition + affinity statistics from Charades-v1
├── new_exp/
│   └── exp2_habit.py           Experiment analysis (Fig.A/B/C/D/F/G)
├── reset.py                    Database reset utility
└── check_system.py             System health monitor
```

---

## References

- Weinland, D., Ronfard, R., & Boyer, E. (2011). A survey of vision-based methods for action representation, recognition and understanding. *CVIU*, 115(2), 224–241.
- Shan, Y., & Akella, S. (2014). View selection for observing articulated human motions. *IJRR*.
- Vishwakarma, S., & Agrawal, A. (2013). A survey on activity recognition and behavior understanding in video surveillance. *The Visual Computer*, 29(10), 983–1009.
- Poppe, R. (2010). A survey on vision-based human action recognition. *Image and Vision Computing*, 28(6), 976–990.
- Sigurdsson, G. A., et al. (2016). Hollywood in homes: Crowdsourcing data collection for activity understanding. *ECCV*.
- Liu, H., et al. (2023). Visual Instruction Tuning (LLaVA). *NeurIPS*.
- Lugaresi, C., et al. (2019). MediaPipe: A framework for perceiving and processing reality. *CVPR Workshop*.
