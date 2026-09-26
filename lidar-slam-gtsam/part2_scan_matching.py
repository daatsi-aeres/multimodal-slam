import time

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree
from tqdm import tqdm


def load_lidar_data(dataset=20):
    """Load Hokuyo LiDAR data. Returns ranges array and timestamps."""
    with np.load("data/Hokuyo%d.npz"%dataset) as data:
        lidar_angle_min = data["angle_min"].item()
        lidar_angle_max = data["angle_max"].item()
        lidar_angle_increment = data["angle_increment"].item()
        lidar_range_min = data["range_min"].item()
        lidar_range_max = data["range_max"].item()
        lidar_ranges = data["ranges"]
        lidar_timestamps = data["time_stamps"]
    return lidar_ranges, lidar_timestamps, lidar_angle_min, lidar_angle_increment, lidar_range_min, lidar_range_max


def lidar_scan_to_points(ranges, angle_min, angle_increment, range_min, range_max):
    """
    Convert a single LiDAR scan (1081 range values) to (x, y) points
    in the SENSOR frame.

    Steps:
      1. Build angles array: angle_min + i * angle_increment for i in 0..1080
      2. Filter out ranges < range_min or > range_max
      3. x = r * cos(angle),  y = r * sin(angle)

    Returns: (K, 2) array of valid (x, y) points where K <= 1081
    """
    angles = angle_min + np.arange(len(ranges)) * angle_increment
    valid = (ranges >= range_min) & (ranges <= range_max)
    x = ranges[valid] * np.cos(angles[valid])
    y = ranges[valid] * np.sin(angles[valid])
    return np.stack((x, y), axis=-1)


def get_lidar_to_body_transform():
    """
    Return the fixed 3x3 SE(2) transform from LiDAR sensor frame
    to robot body frame.

    Look up the sensor offset from docs/RobotConfiguration.pdf.
    (x_offset, y_offset, yaw_offset relative to robot center)
    """
    x_offset = 0.13323  # LiDAR is 13.323 cm in front of the robot center
    y_offset = 0.0
    yaw_offset = 0.0
    return pose_to_transform_2d(x_offset, y_offset, yaw_offset)


def pose_to_transform_2d(x, y, theta):
    """
    Build a 3x3 SE(2) homogeneous transform matrix from (x, y, theta).

    [ cos(θ)  -sin(θ)  x ]
    [ sin(θ)   cos(θ)  y ]
    [   0        0     1 ]
    """
    c = np.cos(theta)
    s = np.sin(theta)
    T = np.array([[c, -s, x],
                  [s,  c, y],
                  [0,  0, 1]])
    return T


def transform_points_2d(points_2d, T):
    """
    Apply a 3x3 SE(2) transform T to an (N, 2) array of points.
    Use homogeneous coordinates.

    Returns: (N, 2) transformed points
    """
    N = points_2d.shape[0]
    homogeneous_points = np.hstack((points_2d, np.ones((N, 1))))
    transformed_homogeneous = homogeneous_points @ T.T
    return transformed_homogeneous[:, :2]


def icp_2d(source, target, init_pose=np.eye(3), max_iter=50, tolerance=1e-6):
    """
    2D version of ICP with robust dynamic outlier rejection.
    """
    current_source = init_pose @ np.hstack((source, np.ones((source.shape[0], 1)))).T
    tree = KDTree(target)
    previous_mse = float('inf')
    prev_pose = init_pose.copy()
    mse_history = []

    for i in range(max_iter):
        distances, indices = tree.query(current_source[:2, :].T)
        distances = distances.flatten()
        indices = indices.flatten()

        # --- OUTLIER REJECTION ---
        median_dist = np.median(distances)
        valid_mask = distances < (3.0 * median_dist)

        if np.sum(valid_mask) < 10:
            break

        valid_source = current_source[:2, :].T[valid_mask]
        valid_target = target[indices[valid_mask]]

        # --- SVD MATH (ONLY ON VALID POINTS) ---
        centroid_source = np.mean(valid_source, axis=0)
        centroid_target = np.mean(valid_target, axis=0)
        source_centered = valid_source - centroid_source
        target_centered = valid_target - centroid_target

        H = source_centered.T @ target_centered
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T

        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T

        t = centroid_target - R @ centroid_source

        pose = np.eye(3)
        pose[:2, :2] = R
        pose[:2, 2] = t

        accumulated_pose = pose @ prev_pose
        current_source = (accumulated_pose @ np.hstack((source, np.ones((source.shape[0], 1)))).T)
        prev_pose = accumulated_pose.copy()

        mse = np.mean(distances[valid_mask]**2)
        mse_history.append(mse)

        if abs(mse - previous_mse) < tolerance:
            break
        previous_mse = mse

    return accumulated_pose, mse, mse_history


def scan_matching_trajectory(lidar_data, odometry_poses, odometry_stamps, downsample_factor=1, tolerance=1e-6):
    """
    Trajectory Estimation via Scan Matching with Degenerate Scan fallback and Progress Bar.
    """
    lidar_ranges, lidar_timestamps, angle_min, angle_increment, range_min, range_max = lidar_data
    icp_poses = []
    icp_stamps = []

    odom_idx_0 = np.argmin(np.abs(odometry_stamps - lidar_timestamps[0]))
    initial_odom = odometry_poses[odom_idx_0]
    current_pose = pose_to_transform_2d(*initial_odom)
    # current_pose is already set from odometry above — don't reset to eye(3)

    icp_poses.append(initial_odom.tolist())
    icp_stamps.append(lidar_timestamps[0])

    pbar_desc = f"ICP (N={downsample_factor}, tol={tolerance})"

    for i in tqdm(range(1, len(lidar_timestamps)), desc=pbar_desc, unit="scan"):
        scan_i = lidar_scan_to_points(lidar_ranges[:, i], angle_min, angle_increment, range_min, range_max)
        scan_j = lidar_scan_to_points(lidar_ranges[:, i-1], angle_min, angle_increment, range_min, range_max)

        scan_i_bodyframe = transform_points_2d(scan_i, get_lidar_to_body_transform())[::downsample_factor]
        scan_j_bodyframe = transform_points_2d(scan_j, get_lidar_to_body_transform())[::downsample_factor]

        odom_idx_i = np.argmin(np.abs(odometry_stamps - lidar_timestamps[i]))
        odom_idx_j = np.argmin(np.abs(odometry_stamps - lidar_timestamps[i-1]))
        odom_pose_i = odometry_poses[odom_idx_i]
        odom_pose_j = odometry_poses[odom_idx_j]

        T_j = pose_to_transform_2d(odom_pose_j[0], odom_pose_j[1], odom_pose_j[2])
        T_i = pose_to_transform_2d(odom_pose_i[0], odom_pose_i[1], odom_pose_i[2])
        init_pose = np.linalg.inv(T_j) @ T_i

        # --- DEGENERATE SCAN FIX ---
        if len(scan_i_bodyframe) < 20 or len(scan_j_bodyframe) < 20:
            current_pose = current_pose @ init_pose
        else:
            icp_result, _, _ = icp_2d(scan_i_bodyframe, scan_j_bodyframe, init_pose=init_pose, tolerance=tolerance)
            current_pose = current_pose @ icp_result

        icp_poses.append(current_pose[:2, 2].tolist() + [np.arctan2(current_pose[1, 0], current_pose[0, 0])])
        icp_stamps.append(lidar_timestamps[i])

    return np.array(icp_poses), np.array(icp_stamps)


def plot_trajectories(odometry_poses, icp_poses):
    """Overlay odometry and ICP trajectories for comparison."""
    plt.figure(figsize=(10, 10))
    plt.plot(odometry_poses[:, 0], odometry_poses[:, 1], label='Odometry', alpha=0.7)
    plt.plot(icp_poses[:, 0], icp_poses[:, 1], label='ICP', alpha=0.7)
    plt.scatter(odometry_poses[0, 0], odometry_poses[0, 1], c='green', marker='o', label='Start')
    plt.scatter(odometry_poses[-1, 0], odometry_poses[-1, 1], c='red', marker='x', label='End')
    plt.legend()
    plt.title('Odometry vs ICP Trajectory')
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.axis('equal')
    plt.grid(True)
    plt.savefig("plots/part2b_scan_matching/trajectory_comparison.png", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    import os
    from part1_odometry import load_sensor_data, integrate_odometry

    dataset = 21
    print(f"Part 2b: Scan Matching (Dataset {dataset})")

    encoder_data, imu_data = load_sensor_data(dataset=dataset)
    odometry_poses, odometry_stamps = integrate_odometry(encoder_data, imu_data)
    lidar_data = load_lidar_data(dataset=dataset)

    base_plot_dir = "plots/part2b_scan_matching"
    os.makedirs(base_plot_dir, exist_ok=True)

    print("\nPhase 1: Base ICP trajectory vs odometry")
    base_icp_poses, _ = scan_matching_trajectory(lidar_data, odometry_poses, odometry_stamps, downsample_factor=1)

    plt.figure(figsize=(10, 10))
    plt.plot(odometry_poses[:, 0], odometry_poses[:, 1], label='Odometry (Drifting)', alpha=0.7, linestyle='--')
    plt.plot(base_icp_poses[:, 0], base_icp_poses[:, 1], label='ICP (Corrected)', alpha=0.9, linewidth=2)
    plt.scatter(odometry_poses[0, 0], odometry_poses[0, 1], c='green', marker='o', s=100, label='Start')
    plt.scatter(odometry_poses[-1, 0], odometry_poses[-1, 1], c='red', marker='x', s=100, label='End (Odom)')
    plt.scatter(base_icp_poses[-1, 0], base_icp_poses[-1, 1], c='purple', marker='*', s=150, label='End (ICP)')
    plt.legend(fontsize=12)
    plt.title('Odometry vs ICP Trajectory (Updated Logic)', fontsize=14, fontweight='bold')
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.axis('equal')
    plt.grid(True)

    p1_path = os.path.join(base_plot_dir, "trajectory_comparison_updatedLogic.png")
    plt.savefig(p1_path, dpi=300, bbox_inches="tight")
    print(f"  -> Saved: {p1_path}")

    print("\nPhase 2: Downsampling factor sweep (N=1, 2, 5, 10)")
    N_values = [1, 2, 5, 10]
    execution_times = []
    trajectories = {}

    for N in N_values:
        start_time = time.time()
        poses, _ = scan_matching_trajectory(lidar_data, odometry_poses, odometry_stamps, downsample_factor=N)
        elapsed = time.time() - start_time
        execution_times.append(elapsed)
        trajectories[N] = poses
        print(f"  -> N={N} completed in {elapsed:.2f} seconds.")

    plt.figure(figsize=(8, 5))
    plt.plot(N_values, execution_times, marker='s', color='purple', linestyle='-', linewidth=2)
    plt.title('ICP Execution Time vs Downsampling Factor (N)', fontweight='bold')
    plt.xlabel('Downsampling Factor (Every N-th point)')
    plt.ylabel('Time (seconds)')
    plt.grid(True)
    p2a_path = os.path.join(base_plot_dir, "time_vs_N_updatedLogic.png")
    plt.savefig(p2a_path, dpi=300, bbox_inches="tight")
    print(f"  -> Saved: {p2a_path}")

    plt.figure(figsize=(10, 10))
    plt.plot(odometry_poses[:, 0], odometry_poses[:, 1], label='Odometry (Raw)', color='black', linestyle=':', linewidth=2)
    colors = ['blue', 'green', 'orange', 'red']
    for i, N in enumerate(N_values):
        poses = trajectories[N]
        plt.plot(poses[:, 0], poses[:, 1], label=f'ICP (N={N})', alpha=0.7, color=colors[i])
    plt.scatter(odometry_poses[0, 0], odometry_poses[0, 1], c='green', marker='o', label='Start')
    plt.title('Trajectory Comparison Across Downsampling Rates', fontweight='bold')
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.axis('equal')
    plt.legend()
    plt.grid(True)
    p2b_path = os.path.join(base_plot_dir, "trajectory_vs_N_updatedLogic.png")
    plt.savefig(p2b_path, dpi=300, bbox_inches="tight")
    print(f"  -> Saved: {p2b_path}")

    print("\nPhase 3: Standalone scan alignment (scans 500 & 501)")
    lidar_ranges, lidar_timestamps, angle_min, angle_increment, range_min, range_max = lidar_data
    idx_prev, idx_curr = 500, 501

    scan_prev = lidar_scan_to_points(lidar_ranges[:, idx_prev], angle_min, angle_increment, range_min, range_max)
    scan_curr = lidar_scan_to_points(lidar_ranges[:, idx_curr], angle_min, angle_increment, range_min, range_max)
    scan_prev_body = transform_points_2d(scan_prev, get_lidar_to_body_transform())
    scan_curr_body = transform_points_2d(scan_curr, get_lidar_to_body_transform())

    odom_idx_prev = np.argmin(np.abs(odometry_stamps - lidar_timestamps[idx_prev]))
    odom_idx_curr = np.argmin(np.abs(odometry_stamps - lidar_timestamps[idx_curr]))
    T_prev = pose_to_transform_2d(odometry_poses[odom_idx_prev, 0], odometry_poses[odom_idx_prev, 1], odometry_poses[odom_idx_prev, 2])
    T_curr = pose_to_transform_2d(odometry_poses[odom_idx_curr, 0], odometry_poses[odom_idx_curr, 1], odometry_poses[odom_idx_curr, 2])
    init_pose_guess = np.linalg.inv(T_prev) @ T_curr

    final_pose, _, mse_history = icp_2d(scan_curr_body, scan_prev_body, init_pose=init_pose_guess)

    scan_curr_init = transform_points_2d(scan_curr_body, init_pose_guess)
    scan_curr_final = transform_points_2d(scan_curr_body, final_pose)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.scatter(scan_prev_body[:, 0], scan_prev_body[:, 1], s=2, c='red', label='Target (Scan 500)')
    plt.scatter(scan_curr_init[:, 0], scan_curr_init[:, 1], s=2, c='blue', label='Source (Scan 501) - Initial')
    plt.title('Before ICP (Odometry Guess)', fontweight='bold')
    plt.axis('equal'); plt.legend(); plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.scatter(scan_prev_body[:, 0], scan_prev_body[:, 1], s=2, c='red', label='Target (Scan 500)')
    plt.scatter(scan_curr_final[:, 0], scan_curr_final[:, 1], s=2, c='green', label='Source (Scan 501) - Final')
    plt.title('After ICP Alignment', fontweight='bold')
    plt.axis('equal'); plt.legend(); plt.grid(True)

    plt.tight_layout()
    p3a_path = os.path.join(base_plot_dir, "icp_alignment_updatedLogic.png")
    plt.savefig(p3a_path, dpi=300, bbox_inches="tight")
    print(f"  -> Saved: {p3a_path}")

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(mse_history) + 1), mse_history, marker='o', linestyle='-', color='b')
    plt.title(f'ICP Convergence (Scans {idx_prev} to {idx_curr})', fontweight='bold')
    plt.xlabel('Iteration')
    plt.ylabel('Mean Squared Error (MSE)')
    plt.grid(True)
    p3b_path = os.path.join(base_plot_dir, "icp_convergence_updatedLogic.png")
    plt.savefig(p3b_path, dpi=300, bbox_inches="tight")
    print(f"  -> Saved: {p3b_path}")

    print("\nPhase 4: Tolerance sweep (1e-4 to 1e-8)")
    tolerances = [1e-4, 1e-5, 1e-6, 1e-7, 1e-8]
    tol_execution_times = []
    tol_trajectories = {}

    for tol in tolerances:
        start_time = time.time()
        # Using downsample_factor=5 for speed during the sweep
        poses, _ = scan_matching_trajectory(lidar_data, odometry_poses, odometry_stamps, downsample_factor=5, tolerance=tol)
        elapsed = time.time() - start_time
        tol_execution_times.append(elapsed)
        tol_trajectories[tol] = poses
        print(f"  -> Tol={tol} completed in {elapsed:.2f} seconds.")

    plt.figure(figsize=(8, 5))
    plt.plot([str(t) for t in tolerances], tol_execution_times, marker='D', color='teal', linestyle='-', linewidth=2)
    plt.title('ICP Execution Time vs Tolerance', fontweight='bold')
    plt.xlabel('Tolerance (MSE Change Threshold)')
    plt.ylabel('Time (seconds)')
    plt.grid(True)
    p4a_path = os.path.join(base_plot_dir, "time_vs_tolerance_updatedLogic.png")
    plt.savefig(p4a_path, dpi=300, bbox_inches="tight")
    print(f"  -> Saved: {p4a_path}")

    plt.figure(figsize=(10, 10))
    plt.plot(odometry_poses[:, 0], odometry_poses[:, 1], label='Odometry (Raw)', color='black', linestyle=':', linewidth=2)
    tol_colors = ['magenta', 'cyan', 'orange', 'blue', 'red']
    for i, tol in enumerate(tolerances):
        poses = tol_trajectories[tol]
        current_linewidth = max(1, 5 - i)
        plt.plot(poses[:, 0], poses[:, 1], label=f'ICP (tol={tol})', alpha=0.7, color=tol_colors[i], linewidth=current_linewidth)
    plt.scatter(odometry_poses[0, 0], odometry_poses[0, 1], c='green', marker='o', label='Start')
    plt.title('Trajectory Comparison Across Tolerances', fontweight='bold')
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.axis('equal')
    plt.legend()
    plt.grid(True)
    p4b_path = os.path.join(base_plot_dir, "trajectory_vs_tolerance_updatedLogic.png")
    plt.savefig(p4b_path, dpi=300, bbox_inches="tight")
    print(f"  -> Saved: {p4b_path}")

    print("\nAll experiments complete.")
