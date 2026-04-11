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
  <img src="assets/image/demo_trajectory_bbc_earth_lvl0_crop.png" width="49%" alt="selection of objects from BBC Earth footate">
  <img src="assets/image/demo_trajectory_Q360_20250912_120202_full_crop.png" width="49%" alt="TCD Flower Bed (QooCam)">
</p>
<p align="center">
  <img src="assets/image/demo_trajectory_qoocam_patio_crop.png" width="49%" alt="Wicklow Patio (QooCam)">
  <img src="assets/image/demo_trajectory_R0010116_crop.png" width="49%" alt="Tanzania (ThetaX)">
</p>
Representative frames from four omnidirectional sequences illustrating object trajectories. Coloured bounding boxes accumulated over time visualise the per-object tracks in the equirectangular projection. The upper-right frame shows honey bees around a flower bed (after a 90&deg vertical rotation of projection), while the remaining frames show avian species. Insets show zoomed-in crops of several tracked identities, highlighting strong appearance ambiguity.

Links to dataset:


## Method
<p align="center">
  <img src="assets/image/OmniSORT-flowdiagram-Ver-02.png" width="100%" alt="Workflow of proposed OmniSORT">
</p>

Our approach revisits SORT-style tracking for omnidirectional small-object scenarios and introduces modifications designed for limit compute resource, weak-appearance, motion-dominated tracking.

### 1. $Kalman\ Speed\ Update$: Seam-Aware Motion Model

Standard motion updates can break near the association at borders of an equirectangular frame, where an object may appear to jump across the seam. Our seam-aware Kalman speed update accounts for this wrap-around behaviour, giving more stable motion prediction in omnidirectional footage.
### 2. $OmniEuc$: Seam-Aware Euclidean Distance

Standard Euclidean distance does not reflect true proximity when objects are close across the image seam. OmniEuc fixes this by measuring distance in a seam-aware way, making object association more reliable in omnidirectional views.
### 3. $E_{fuse}$: Composite Association Metrics

$E_{fuse}$ combines **OmniEuc** with **GIoU** to build a stronger association cost. This helps the tracker use both seam-aware position cues and box-overlap cues when matching detections across frames.

---
## Results
<video
  src="https://github.com/user-attachments/assets/39593f4a-2d6e-4132-9428-f2031833091a"
  poster="./assets/image/thumbnail_bbc_earth.png"
  controls
  height="800">
</video>
### Tracking Results on OmniSmall dataset

### with ground-truth labels

| Method | HOTA $\uparrow$ | MOTA $\uparrow$ | IDF1 $\uparrow$ | IDSw $\downarrow$ | FPS $\uparrow$ |
|---|---:|---:|---:|---:|---:|
| SORT | 64.16 | 82.39 | 76.19 | 394 | 0.63 |
| ByteTrack | 67.95 | 84.71 | 84.74 | 93 | 1.51 |
| OCSORT | 75.30 | 77.51 | 74.10 | 170 | 1.26 |
| HybridSORT | 60.49 | 66.83 | 54.23 | 1154 | 0.61 |
| **OmniSORT + $E_{fuse}$ (ours)** | 90.88 | 97.94 | 94.04 | 18 | 1.17 |
| **OmniOCSORT + $E_{fuse}$ (ours)** | 92.21 | 96.20 | 91.12 | 37 | 0.97 |

### With YOLOX detection labels

| Method | HOTA $\uparrow$ | MOTA $\uparrow$ | IDF1 $\uparrow$ | IDSw $\downarrow$ | FPS $\uparrow$ |
|---|---:|---:|---:|---:|---:|
| SORT | 36.43 | 1.91 | 44.85 | 353 | 0.51 |
| ByteTrack | 31.85 | 20.25 | 34.94 | 37 | 2.02 |
| OCSORT | 41.63 | 32.29 | 50.94 | 111 | 1.01 |
| HybridSORT | 40.78 | 32.70 | 49.03 | 183 | 0.66 |
| **OmniSORT + $E_{fuse}$ (ours)** | 35.57 | 2.81 | 43.36 | 394 | 1.08 |
| **OmniOCSORT + $E_{fuse}$ (ours)** | 44.19 | 36.44 | 55.61 | 172 | 1.00 |

---

### Ablation Study
These ablation experiments evaluate the impact of each proposed component in the tracking pipeline. For **OmniSORT**, the baseline is **SORT**; for **OmniOCSORT**, the baseline is **OCSORT**. The reported **p-values** indicate whether the change from the corresponding baseline is statistically significant.

**Takeaway.** The results show that the proposed seam-aware association design improves tracking performance, especially on the omnidirectional benchmark. In general, **OmniEuc** and **$E_{fuse}$** bring the largest gains in the target setting, while the effect of each component is more mixed on JRDB. Overall, the combined design is most effective on challenging omnidirectional small-object tracking.
### Performance on OmniSmall dataset with Ground-Truth labels

| Method | HOTA | MOTA | IDF1 | IDSw | p-value<br>(HOTA) | p-value<br>(MOTA) | p-value<br>(IDF1) |
|---|---:|---:|---:|---:|---:|---:|---:|
| OmniSORT + IoU | 19.40 | 18.05 | 14.02 | 2215 | 0.0005 | 0.0008 | 0.0006 |
| OmniSORT + GIoU | 57.81 | 64.14 | 52.90 | 1306 | 0.0683 | 0.0091 | 0.0069 |
| OmniSORT + OmniEuc | 88.88 | 97.82 | 91.02 | 36 | 0.0012 | 0.0285 | 0.0084 |
| OmniSORT + $E_{fuse}$ ($\lambda$=0.9) | 90.88 | 97.94 | 94.04 | 18 | 0.0095 | 0.0506 | 0.0037 |
| OmniOCSORT + IoU | 76.29 | 79.54 | 72.96 | 190 | 0.0725 | 0.2446 | 0.2499 |
| OmniOCSORT + GIoU | 88.98 | 88.92 | 88.13 | 50 | 0.0397 | 0.0485 | 0.0331 |
| OmniOCSORT + OmniEuc | 90.98 | 95.64 | 90.31 | 54 | 0.0049 | 0.0192 | 0.0034 |
| OmniOCSORT + $E_{fuse}$ ($\lambda$=0.7) | 92.21 | 96.20 | 91.12 | 37 | 0.0049 | 0.0195 | 0.0029 |

### Performance on JRDB dataset with Ground-Truth labels

| Method | HOTA | MOTA | IDF1 | IDSw | p-value<br>(HOTA) | p-value<br>(MOTA) | p-value<br>(IDF1) |
|---|---:|---:|---:|---:|---:|---:|---:|
| OmniSORT + IoU | 53.67 | 84.62 | 44.03 | 18346 | 0.0000 | 0.0000 | 0.0000 |
| OmniSORT + GIoU | 63.27 | 92.16 | 53.97 | 13073 | 0.0000 | 0.0093 | 0.0724 |
| OmniSORT + OmniEuc | 41.15 | 92.21 | 34.43 | 26201 | 0.0000 | 0.6314 | 0.0000 |
| OmniSORT + $E_{fuse}$ ($\lambda$=0.3) | 65.23 | 94.07 | 56.56 | 9125 | 0.0000 | 0.0000 | 0.0002 |
| OmniOCSORT + IoU | 67.64 | 90.33 | 60.70 | 5907 | 0.0014 | 0.0003 | 0.7380 |
| OmniOCSORT + GIoU | 63.43 | 91.99 | 57.85 | 7594 | 0.2370 | 0.0000 | 0.4088 |
| OmniOCSORT + OmniEuc | 63.22 | 91.93 | 57.75 | 7741 | 0.5108 | 0.0002 | 0.5212 |
| OmniOCSORT + $E_{fuse}$ ($\lambda$=0.9) | 64.79 | 91.61 | 58.73 | 6931 | 0.2855 | 0.0001 | 0.3082 |

### Performance on OmniSmall dataset with YOLOX detection labels

| Method | HOTA | MOTA | IDF1 | IDSw | p-value<br>(HOTA) | p-value<br>(MOTA) | p-value<br>(IDF1) |
|---|---:|---:|---:|---:|---:|---:|---:|
| OmniSORT + IoU | 26.97 | 1.47 | 30.31 | 771 | 0.0036 | 0.5235 | 0.0303 |
| OmniSORT + GIoU | 36.12 | 2.81 | 43.47 | 392 | 0.3461 | 0.1775 | 0.4099 |
| OmniSORT + OmniEuc | 33.32 | -7.96 | 39.66 | 785 | 0.3045 | 0.2170 | 0.3540 |
| OmniSORT + $E_{fuse}$ ($\lambda$=0.1) | 35.57 | 2.81 | 43.36 | 394 | 0.2015 | 0.2322 | 0.2226 |
| OmniOCSORT + IoU | 42.84 | 34.89 | 52.14 | 105 | 0.0031 | 0.0148 | 0.0019 |
| OmniOCSORT + GIoU | 42.96 | 34.87 | 52.47 | 104 | 0.0038 | 0.0185 | 0.0198 |
| OmniOCSORT + OmniEuc | 43.02 | 36.06 | 53.09 | 194 | 0.0050 | 0.0187 | 0.0133 |
| OmniOCSORT + $E_{fuse}$ ($\lambda$=0.6) | 44.19 | 36.44 | 55.61 | 172 | 0.0080 | 0.0100 | 0.0042 |

### Performance on JRDB dataset with YOLOX detection labels

| Method | HOTA | MOTA | IDF1 | IDSw | p-value<br>(HOTA) | p-value<br>(MOTA) | p-value<br>(IDF1) |
|---|---:|---:|---:|---:|---:|---:|---:|
| OmniSORT + IoU | 22.13 | 16.21 | 20.30 | 38552 | 0.0000 | 0.2426 | 0.0000 |
| OmniSORT + GIoU | 25.26 | 15.89 | 23.83 | 25142 | 0.4844 | 0.1247 | 0.0413 |
| OmniSORT + OmniEuc | 11.70 | -3.74 | 10.81 | 116906 | 0.0000 | 0.0000 | 0.0000 |
| OmniSORT + $E_{fuse}$ ($\lambda$=0.1) | 25.55 | 15.64 | 24.16 | 24030 | 0.8478 | 0.1581 | 0.1046 |
| OmniOCSORT + IoU | 27.03 | 38.22 | 27.79 | 13475 | 0.0000 | 0.0764 | 0.0000 |
| OmniOCSORT + GIoU | 24.15 | 37.50 | 25.48 | 15426 | 0.0000 | 0.0621 | 0.0003 |
| OmniOCSORT + OmniEuc | 22.44 | 36.77 | 23.81 | 17265 | 0.0000 | 0.0425 | 0.0000 |
| OmniOCSORT + $E_{fuse}$ ($\lambda$=0.5) | 25.55 | 37.95 | 26.96 | 14401 | 0.0000 | 0.0321 | 0.0000 |

---

## Installation & Sample usage(TODO)

```bash
git clone https://github.com/Xin-Shu/OmniSORT.git
cd OmniSORT
