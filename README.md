# OmniSORT

**Re-Engineering Sort-Based Algorithms for Low Cost Small Object Tracking from Omnidirectional Footage**

---

## Overview

Most multi-object trackers are designed for standard cameras and often rely on heavy appearance models. These assumptions do not work well for low-cost omnidirectional footage, where seam discontinuities and tiny fast-moving targets make tracking difficult.

This repository presents two modifications of SORT-based trackers (SORT and OCSORT) to adapt multi small objects tracking from omnidirectional footages.

## Contributions

- A **seam-aware Kalman filter** that preserves spherical continuity across projection borders;
- **OmniEuc + GIoU** ($E_{fuse}$) for more robust object association in omnidirectional tracking;
- **OmniSmall**, a benchmark for small-object tracking in omnidirectional footage, with real-world trajectories, strong distortion, and frequent seam crossings.
---
## Dataset Snapshot
<p align="center">
  <img src="assets/images/demo_trajectory_bbc_earth_lvl0_crop.png" width="49%" alt="Dataset sample 1">
  <img src="assets/images/demo_trajectory_Q360_20250912_120202_full_crop.png" width="49%" alt="Dataset sample 2">
</p>
<p align="center">
  <img src="assets/images/demo_trajectory_qoocam_patio_crop.png" width="49%" alt="Dataset sample 3">
  <img src="assets/images/demo_trajectory_R0010116_crop.png" width="49%" alt="Dataset sample 4">
</p>
Representative frames from four omnidirectional sequences illustrating object trajectories. Coloured bounding boxes accumulated over time visualise the per-object tracks in the equirectangular projection. The upper-right frame shows honey bees around a flower bed (after a 90&deg vertical rotation of projection), while the remaining frames show avian species. Insets show zoomed-in crops of several tracked identities, highlighting strong appearance ambiguity.

## Method

Our approach revisits SORT-style tracking for omnidirectional small-object scenarios and introduces modifications designed for limit compute resource, weak-appearance, motion-dominated tracking.

### 1. $Kalman\ Speed\ Update$: Seam-Aware Motion Model

Standard motion updates can break near the association at borders of an equirectangular frame, where an object may appear to jump across the seam. Our seam-aware Kalman speed update accounts for this wrap-around behaviour, giving more stable motion prediction in omnidirectional footage.
### 2. $OmniEuc$: Seam-Aware Euclidean Distance

Standard Euclidean distance does not reflect true proximity when objects are close across the image seam. OmniEuc fixes this by measuring distance in a seam-aware way, making object association more reliable in omnidirectional views.
### 3. $E_{fuse}$: Composite Association Metrics

$E_{fuse}$ combines **OmniEuc** with **GIoU** to build a stronger association cost. This helps the tracker use both seam-aware position cues and box-overlap cues when matching detections across frames.

---

## Results
[![Watch the demo](./assets/images/thumbnail_bbc_earth.png)](./assets/demo_bbc_earth.mp4)
### Main Tracking Results

Replace this with your final numbers.

| Method | HOTA ↑ | MOTA ↑ | IDF1 ↑ | ID Sw. ↓ |
|---|---:|---:|---:|---:|
| SORT | -- | -- | -- | -- |
| OC-SORT | -- | -- | -- | -- |
| OmniSORT | -- | -- | -- | -- |

---

### Ablation Study

You mentioned including ablations, so here is a template that matches that structure.

| Variant | Association Metric | Kalman Update | HOTA ↑ | MOTA ↑ | IDF1 ↑ | ID Sw. ↓ |
|---|---|---|---:|---:|---:|---:|
| Baseline SORT | IoU | Original | -- | -- | -- | -- |
| SORT + modified Kalman | IoU | Proposed | -- | -- | -- | -- |
| SORT + modified Kalman | GIoU | Proposed | -- | -- | -- | -- |
| SORT + modified Kalman | OmniEuc | Proposed | -- | -- | -- | -- |
| OmniSORT | OmniEuc | Proposed | -- | -- | -- | -- |

You can also add a short takeaway under the table:

> The ablation study shows that the proposed Kalman speed update improves the SORT baseline, and replacing standard association with OmniEuc yields further gains. Combining both gives the best overall performance.

---

## Installation

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>

# create environment
conda create -n omnisort python=3.10 -y
conda activate omnisort

# install dependencies
pip install -r requirements.txt