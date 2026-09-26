# LiDAR-Based SLAM
Siddharth Rama Sushil

---

## Overview

This project implements Simultaneous Localization and Mapping (SLAM) for a differential-drive robot using encoder/IMU odometry, 2-D LiDAR scan matching (ICP), a 2-D occupancy grid map, a Kinect RGB-D texture map, and GTSAM pose-graph optimization with loop-closure detection.

The pipeline is split into four sequential parts, each in its own Python file. Each file can be run independently as `python <file>.py` and will produce and save all relevant plots.

---

## File Structure

```
lidar-slam-gtsam/
├── load_data.py              # Data loading helper
├── map_utils.py              # Map plotting and Bresenham ray tracing
├── part1_odometry.py         # Part 1 — Encoder + IMU dead-reckoning odometry
├── part2_scan_matching.py    # Part 2b — 2-D ICP scan matching & trajectory refinement
├── part3_mapping.py          # Part 3 — Occupancy grid map + Kinect RGB-D texture map
├── part4_gtsam.py            # Part 4 — GTSAM pose-graph optimization + loop closure
└── icp_warm_up/
    ├── icp_3d.py             # Part 2a — 3-D ICP warm-up implementation
    ├── test_icp.py           # Test harness for the 3-D ICP warm-up
    └── utils.py              # Warm-up helper: point cloud loading & visualization
```

---

## Data Paths

The code expects data to be located at:

| Data type | Path |
|-----------|------|
| Encoders  | `data/Encoders<dataset>.npz` |
| IMU       | `data/Imu<dataset>.npz` |
| LiDAR (Hokuyo) | `data/Hokuyo<dataset>.npz` |
| Kinect timestamps | `data/Kinect<dataset>.npz` |
| Disparity images  | `dataRGBD/Disparity<dataset>/disparity<dataset>_<k>.png` |
| RGB images        | `dataRGBD/RGB<dataset>/rgb<dataset>_<k>.png` |
| ICP warm-up objects | `icp_warm_up/data/` |

Datasets used: **20** (Parts 3 & 4) and **21** (Parts 1 & 2).

---

## Dependencies

```bash
pip install numpy matplotlib scikit-learn tqdm opencv-python scipy gtsam
```

---

## Part-by-Part Description & How to Run

---

### Part 1 — Encoder + IMU Odometry (`part1_odometry.py`)

**What it does:**
Estimates the robot's trajectory using the differential-drive motion model. At each encoder timestamp, it:
1. Computes left/right wheel velocities from the four encoder counts (0.0022 m/tick, 40 Hz).
2. Looks up the nearest IMU yaw rate (rad/s) to use as the angular velocity `ω`.
3. Integrates the motion model step-by-step: `x += v·cos(θ)·dt`, `y += v·sin(θ)·dt`, `θ += ω·dt`.
4. Saves a comparison plot of IMU vs. encoder-derived yaw rates and the resulting (x, y) trajectory.

**Key functions:**
- `compute_wheel_velocities()` — converts raw encoder counts to left/right wheel speeds
- `motion_model_step()` — single-step SE(2) propagation
- `integrate_odometry()` — full trajectory integration loop
- `plot_trajectory()` — saves the trajectory plot

**Run:**
```bash
python part1_odometry.py
```
**Output plots saved to:** `plots/part1_odometry/`

---

### Part 2a — 3-D ICP Warm-Up (`icp_warm_up/icp_3d.py` + `test_icp.py`)

**What it does:**
Implements a full 3-D Iterative Closest Point (ICP) algorithm to estimate the rigid-body pose of two objects (a drill and a liquid container) given multiple depth-image point clouds. This warm-up validates the ICP implementation before applying it to 2-D scan matching.

Key steps in `icp_3d.py`:
1. `find_nearest_neighbors()` — KD-tree nearest-neighbor search
2. `compute_optimal_transform()` — SVD-based optimal rotation and translation
3. `build_transform()` — packs (R, t) into a 4×4 SE(3) matrix
4. `icp_3d()` — the main ICP loop with convergence check
5. `icp_with_yaw_init()` — tries 36 evenly-spaced initial yaw angles and picks the best result (lowest MSE) to avoid local minima

**Run:**
```bash
cd icp_warm_up
python test_icp.py
```

---

### Part 2b — 2-D LiDAR Scan Matching (`part2_scan_matching.py`)

**What it does:**
Refines the odometry trajectory using consecutive LiDAR scan pairs aligned with 2-D ICP. For each pair of adjacent scans:
1. Converts raw LiDAR ranges to (x, y) points in the sensor frame, then transforms them to the robot body frame using the known sensor offset (13.323 cm forward).
2. Uses the odometry-derived relative transform as the ICP initial guess.
3. Runs the 2-D ICP loop (SVD-based, with median-distance outlier rejection) to find the refined relative pose.
4. Chains all relative poses to build the full corrected trajectory.
5. Falls back to the odometry guess for degenerate scans (fewer than 20 valid points).

The `__main__` block runs four experiments and saves plots:
- **Phase 1:** Baseline ICP trajectory vs. raw odometry (Dataset 21)
- **Phase 2:** Downsampling factor sweep (N = 1, 2, 5, 10) — runtime vs. accuracy trade-off
- **Phase 3:** Visual before/after alignment of a single scan pair (scans 500 & 501) and ICP convergence curve
- **Phase 4:** ICP convergence tolerance sweep (1e-4 to 1e-8)

**Key functions:**
- `lidar_scan_to_points()` — range-to-Cartesian conversion with validity filtering
- `get_lidar_to_body_transform()` — fixed SE(2) sensor-to-body extrinsic
- `icp_2d()` — 2-D ICP with outlier rejection
- `scan_matching_trajectory()` — full trajectory estimation loop

**Run:**
```bash
python part2_scan_matching.py
```
**Output plots saved to:** `plots/part2b_scan_matching/`

---

### Part 3 — Occupancy Grid + Texture Map (`part3_mapping.py`)

**What it does:**
Uses the ICP-refined trajectory from Part 2 to build two maps:

**Occupancy grid map:**
For each LiDAR scan, transforms the hit points from sensor frame → body frame → world frame, then updates a log-odds occupancy grid. Free cells (along each ray) are decremented; occupied cells (at the endpoint) are incremented. Two ray-tracing methods are implemented and switchable via comments:
- *Method 1 (Bresenham):* Traces each ray cell-by-cell (slower, explicit)
- *Method 2 (OpenCV fillPoly):* Fills the scanner wedge as a polygon (much faster, used by default)

**Texture map:**
For each Kinect disparity frame, back-projects every valid depth pixel to a 3-D point in the camera frame, transforms it to the robot body frame using the full SE(3) extrinsic (position + roll/pitch/yaw), and then to the world frame using the nearest ICP pose. Points near the floor plane (`|z_body| < 0.5 m`) are projected onto the grid and colored with the corresponding RGB pixel.

The `__main__` block also runs three paramter-sweep experiments:
- **Exp 1:** Log-odds hit value sweep (0.4, 0.85, 2.0)
- **Exp 2:** Map resolution sweep (0.025, 0.05, 0.10, 0.20 m/cell)
- **Exp 3:** Progressive map snapshots at 10%, 25%, 50%, 75%, 100% completion

**Key functions:**
- `init_occupancy_map()` — dynamically sizes the grid to the trajectory extent
- `update_map_with_scan()` — log-odds update (OpenCV method by default)
- `lidar_points_to_world()` — body-to-world frame transform
- `build_occupancy_map()` — main occupancy mapping loop
- `get_kinect_to_body_transform()` — full SE(3) Kinect extrinsic
- `build_texture_map()` — vectorized RGBD-to-floor projection loop

**Run:**
```bash
python part3_mapping.py
```
**Output plots saved to:** `plots/part3_mapping/`

> **Note:** This script runs ICP internally (downsample=1) before building the maps, so it may take several minutes.

---

### Part 4 — GTSAM Pose Graph Optimization (`part4_gtsam.py`)

**What it does:**
Refines the ICP trajectory further using GTSAM factor graph optimization with proximity-based loop closure detection.

**Phase 1 — Build the factor graph:**
Consecutive ICP relative transforms are added as `BetweenFactorPose2` edges. A prior factor fixes the origin at (0, 0, 0°).

**Phase 2 — Loop closure detection:**
Iterates over every 20th pose and finds spatially nearby candidates (within 2.0 m, at least 50 poses and 10 m of travel apart). For each candidate pair, runs ICP and accepts the edge only if:
- ICP MSE < threshold (0.001), **and**
- Overlap ratio ≥ 50% (fraction of aligned source points within 0.3 m of the target)

This two-gate filter prevents false loop closures from being added to the graph.

**Phase 3 — Optimization:**
Runs Levenberg-Marquardt optimization on the factor graph and extracts the optimized poses.

**Phase 4 — Map reconstruction & visualization:**
Re-runs the occupancy and texture mapping pipelines using the GTSAM-optimized poses, then saves:
- Trajectory comparison (raw ICP vs. GTSAM-optimized)
- Side-by-side occupancy map comparison (before/after)
- GTSAM convergence curve
- Loop closure histogram (spatial & temporal distance distributions)
- Top 5 loop closure visualizations (first-visit vs. re-visit scan overlays)
- Final fused occupancy + texture map

**Key functions:**
- `build_factor_graph()` — constructs the GTSAM graph from ICP relative poses
- `detect_loop_closures()` — proximity search + ICP + overlap-ratio gating
- `optimize_factor_graph()` — Levenberg-Marquardt via GTSAM
- `extract_poses_from_result()` — pulls optimized Pose2 values from GTSAM result

**Run:**
```bash
python part4_gtsam.py
```
**Output plots saved to:** `plots/part4_gtsam/`

> **Note:** This script runs the full pipeline (odometry → ICP → graph build → loop closure → optimization → mapping) and is the most time-intensive. Expect ~10–20 minutes depending on hardware.

---


## Recommended Run Order

```bash
python part1_odometry.py        # ~10 seconds
cd icp_warm_up && python test_icp.py && cd ..   # ~2–5 minutes
python part2_scan_matching.py   # ~5–15 minutes (runs 4 experiments)
python part3_mapping.py         # ~10–20 minutes
python part4_gtsam.py           # ~15–30 minutes
```

Each script is self-contained and saves all output automatically. No command-line arguments are needed.
