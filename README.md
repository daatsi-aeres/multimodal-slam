# Multimodal SLAM & State Estimation

Three state-estimation and SLAM pipelines, from IMU-only orientation tracking to full LiDAR and visual–inertial SLAM.

| Project | What it does | Report |
|---|---|---|
| [`imu-orientation-panorama/`](imu-orientation-panorama) | 3D orientation tracking from IMU data via quaternion kinematics and projected gradient descent on the manifold, then panorama stitching from the tracked orientation | [PDF](reports/imu-orientation-panorama-report.pdf) |
| [`lidar-slam-gtsam/`](lidar-slam-gtsam) | Encoder/IMU odometry, ICP scan matching, occupancy-grid and RGB-D texture mapping, and pose-graph optimization with loop closure in GTSAM | [PDF](reports/lidar-slam-report.pdf) |
| [`visual-inertial-slam-ekf/`](visual-inertial-slam-ekf) | Visual–inertial SLAM with an extended Kalman filter on SE(3): IMU prediction, stereo feature tracking and landmark updates | [PDF](reports/vi-slam-report.pdf) |

Timelapses of the LiDAR SLAM occupancy and texture maps are in [`media/`](media).

Each folder has its own README with run instructions. Datasets are not included.

Built with Shahid Mohammad Mulla.
