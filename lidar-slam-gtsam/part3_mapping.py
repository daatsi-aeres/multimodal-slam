import cv2
import numpy as np
import matplotlib.pyplot as plt
from map_utils import bresenham2D, plot_map
from part2_scan_matching import get_lidar_to_body_transform
from tqdm import tqdm


def init_occupancy_map(poses, padding=10.0, resolution=0.05):
    """
    Create the occupancy grid map dynamically sized to fit the robot's trajectory tightly.

    poses: (N, 3) array of [x, y, theta] trajectory coordinates
    padding: extra space in meters around the trajectory to catch LiDAR hits
    resolution: meters per cell
    """
    min_x = np.min(poses[:, 0]) - padding
    max_x = np.max(poses[:, 0]) + padding
    min_y = np.min(poses[:, 1]) - padding
    max_y = np.max(poses[:, 1]) + padding

    nx = int(np.ceil((max_x - min_x) / resolution))
    ny = int(np.ceil((max_y - min_y) / resolution))

    MAP = {
        'res': np.array([resolution, resolution]),
        'min': np.array([min_x, min_y]),
        'max': np.array([max_x, max_y]),
        'size': np.array([nx, ny]),
        'map': np.zeros((nx, ny))
    }
    return MAP


LOG_ODDS_HIT  = +0.85   # tune this
LOG_ODDS_MISS = -0.35   # tune this
LOG_ODDS_MAX  =  10.0   # clamp to prevent saturation
LOG_ODDS_MIN  = -10.0


def world_to_map_cell(x_world, y_world, MAP):
    """
    Convert world-frame (x, y) coordinates to integer grid cell indices.

    Returns: (col, row) integer indices, or None if out of bounds.
    """
    col = int((x_world - MAP['min'][0]) / MAP['res'][0])
    row = int((y_world - MAP['min'][1]) / MAP['res'][1])
    if 0 <= col < MAP['size'][0] and 0 <= row < MAP['size'][1]:
        return col, row
    else:
        return None


def update_map_with_scan(MAP, robot_x, robot_y, lidar_points_world, log_odds_hit, log_odds_miss):
    """Update map using parameterized log-odds, togglable between Bresenham and OpenCV."""
    robot_cell = world_to_map_cell(robot_x, robot_y, MAP)
    if robot_cell is None:
        return

    cols_all = ((lidar_points_world[:, 0] - MAP['min'][0]) / MAP['res'][0]).astype(int)
    rows_all = ((lidar_points_world[:, 1] - MAP['min'][1]) / MAP['res'][1]).astype(int)

    valid = (cols_all >= 0) & (cols_all < MAP['size'][0]) & \
            (rows_all >= 0) & (rows_all < MAP['size'][1])

    hit_cols = cols_all[valid]
    hit_rows = rows_all[valid]
    hit_cells = np.column_stack((hit_cols, hit_rows))

    if len(hit_cells) == 0:
        return

    # Toggle: move the """ to switch between METHOD 1 (Bresenham) and METHOD 2 (OpenCV).

    # --- METHOD 1: Bresenham Ray Tracing (Slower, strictly traces rays) ---
    """
    ray_cells_list = [bresenham2D(robot_cell[0], robot_cell[1], hit_cell[0], hit_cell[1]).astype(int) for hit_cell in hit_cells]

    for ray_cells in ray_cells_list:
        cols = ray_cells[0, :-1]
        rows = ray_cells[1, :-1]
        MAP['map'][cols, rows] += log_odds_miss

        hit_col = ray_cells[0, -1]
        hit_row = ray_cells[1, -1]
        MAP['map'][hit_col, hit_row] += log_odds_hit

    np.clip(MAP['map'], LOG_ODDS_MIN, LOG_ODDS_MAX, out=MAP['map'])
    """

    # --- METHOD 2: OpenCV fillPoly (Ultra-Fast, fills the scanner wedge) ---
    # """
    mask = np.zeros((MAP['size'][1], MAP['size'][0]), dtype=np.uint8)  # (ny, nx) for OpenCV

    pts = np.vstack(([robot_cell], hit_cells)).astype(np.int32)
    cv2.fillPoly(mask, [pts], 1)

    # mask is (ny, nx) — OpenCV fills mask[row, col]; transpose back to (nx, ny) for our map convention
    MAP['map'][mask.T == 1] += log_odds_miss

    # fillPoly marks endpoints as miss too, so add the difference to correct them
    MAP['map'][hit_cols, hit_rows] += (log_odds_hit - log_odds_miss)

    np.clip(MAP['map'], LOG_ODDS_MIN, LOG_ODDS_MAX, out=MAP['map'])
    # """


def lidar_points_to_world(lidar_points_body, robot_pose):
    """
    Transform 2D LiDAR points from robot body frame to world frame.

    lidar_points_body: (K, 2) points in body frame
    robot_pose: [x, y, theta] of robot in world

    Returns: (K, 2) points in world frame
    """
    x, y, theta = robot_pose
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s],
                  [s,  c]])
    t = np.array([x, y])
    return (R @ lidar_points_body.T).T + t


def build_occupancy_map(icp_poses, icp_stamps, lidar_data, resolution=0.05, log_odds_hit=0.85, log_odds_miss=-0.35, snapshot_percents=None, map_range_max=None):
    """Build map with sweep parameters and optional snapshots.

    map_range_max : float or None
        If set, LiDAR rays longer than this value (metres) are discarded when
        building the occupancy map.  Use e.g. 8.0 to suppress the starburst
        ghost caused by long, unobstructed rays in open / glass-walled areas.
        Defaults to None (use the sensor's full range_max).
    """
    MAP = init_occupancy_map(icp_poses, padding=10.0, resolution=resolution)
    snapshots_dict = {}

    lidar_ranges, lidar_timestamps, angle_min, angle_increment, range_min, range_max = lidar_data
    effective_range_max = map_range_max if map_range_max is not None else range_max
    total_scans = len(lidar_timestamps) - 1

    for i in tqdm(range(1, len(lidar_timestamps)), desc="Building Occupancy Map", unit="scan"):
        t = lidar_timestamps[i]
        icp_idx = np.argmin(np.abs(icp_stamps - t))
        robot_pose = icp_poses[icp_idx]

        ranges = lidar_ranges[:, i]
        angles = angle_min + np.arange(len(ranges)) * angle_increment
        valid_mask = (ranges >= range_min) & (ranges <= effective_range_max)
        ranges = ranges[valid_mask]
        angles = angles[valid_mask]

        x_body = ranges * np.cos(angles)
        y_body = ranges * np.sin(angles)
        lidar_points_sensorframe = np.stack([x_body, y_body], axis=1)
        lidar_points_body = get_lidar_to_body_transform() @ np.hstack((lidar_points_sensorframe, np.ones((lidar_points_sensorframe.shape[0], 1)))).T
        lidar_points_body = lidar_points_body[:2, :].T

        lidar_points_world = lidar_points_to_world(lidar_points_body, robot_pose)

        update_map_with_scan(MAP, robot_pose[0], robot_pose[1], lidar_points_world, log_odds_hit, log_odds_miss)

        if snapshot_percents is not None:
            for p in snapshot_percents:
                if i == int((p / 100.0) * total_scans):
                    snapshots_dict[p] = MAP['map'].copy()

    if snapshot_percents is not None:
        snapshots_dict[100] = MAP['map'].copy()
        return MAP, snapshots_dict

    return MAP


def load_kinect_data(dataset=20):
    """
    Load Kinect RGB and disparity timestamps.

    Returns disp_stamps, rgb_stamps
    """
    with np.load("data/Kinect%d.npz"%dataset) as data:
        disp_stamps = data["disparity_time_stamps"]
        rgb_stamps = data["rgb_time_stamps"]
    return disp_stamps, rgb_stamps


def init_texture_map(occupancy_MAP):
    """
    Create an RGB texture map with the same dimensions as the occupancy map.
    Initialize all cells to a neutral color (e.g., gray).

    Returns: (nx, ny, 3) uint8 array
    """
    nx, ny = occupancy_MAP['size']
    texture_map = np.full((ny, nx, 3), fill_value=128, dtype=np.uint8)
    return texture_map


def disparity_to_depth_and_rgb_pixel(d, i, j):
    """
    Convert disparity value d at depth pixel (i, j) to:
      - depth (meters)
      - RGB image pixel coordinates (rgbi, rgbj)

    Formulas:
      dd    = -0.00304 * d + 3.31
      depth = 1.03 / dd
      rgbi  = (526.37 * i + 19276 - 7877.07 * dd) / 585.051
      rgbj  = (526.37 * j + 16662) / 585.051
    """
    dd = -0.00304 * d + 3.31
    depth = 1.03 / dd
    rgbi = (526.37 * i + 19276 - 7877.07 * dd) / 585.051
    rgbj = (526.37 * j + 16662) / 585.051
    rounded_rgbi = int(round(rgbi))
    rounded_rgbj = int(round(rgbj))
    return depth, rounded_rgbi, rounded_rgbj


def depth_pixel_to_camera_frame(i, j, depth, K_inv):
    """
    Back-project a depth pixel (i, j) with given depth to a 3D point
    in the camera coordinate frame using intrinsic matrix K.

    Formula: p_cam = depth * K^{-1} @ [j, i, 1]^T
    (Note: image row=i maps to v-axis, col=j maps to u-axis)

    Returns: (3,) point in camera frame
    """
    pixel_homog = np.array([j, i, 1])
    p_cam = depth * K_inv @ pixel_homog
    return p_cam


def get_kinect_to_body_transform():
    """
    Return the 4x4 SE(3) transform from Kinect depth camera frame
    to robot body frame, accounting for optical frame orientation.

    From spec: position (0.18, 0.005, 0.36) m,
               roll=0, pitch=0.36 rad, yaw=0.021 rad
    """
    x, y, z = 0.18, 0.005, 0.36
    roll, pitch, yaw = 0.0, 0.36, 0.021

    c_r, s_r = np.cos(roll), np.sin(roll)
    c_p, s_p = np.cos(pitch), np.sin(pitch)
    c_y, s_y = np.cos(yaw), np.sin(yaw)

    # Base rotation mapping camera optical frame to robot body frame:
    # Camera X (right)   -> Body -Y (right)
    # Camera Y (down)    -> Body -Z (down)
    # Camera Z (forward) -> Body +X (forward)
    R_base = np.array([[ 0,  0,  1],
                       [-1,  0,  0],
                       [ 0, -1,  0]])

    R_roll = np.array([[1, 0, 0],
                       [0, c_r, -s_r],
                       [0, s_r, c_r]])

    R_pitch = np.array([[c_p, 0, s_p],
                        [0, 1, 0],
                        [-s_p, 0, c_p]])

    R_yaw = np.array([[c_y, -s_y, 0],
                      [s_y, c_y, 0],
                      [0, 0, 1]])

    R = R_yaw @ R_pitch @ R_roll @ R_base

    T_kinect_to_body = np.eye(4)
    T_kinect_to_body[:3, :3] = R
    T_kinect_to_body[:3, 3] = [x, y, z]

    return T_kinect_to_body


def load_disparity_frame(k, dataset=20):
    """
    Load the k-th disparity for the given dataset.
    """
    disp_path = f"dataRGBD/Disparity{dataset}/disparity{dataset}_{k}.png"
    disparity = cv2.imread(disp_path, cv2.IMREAD_UNCHANGED)
    return disparity


def load_rgb_frame(k, dataset=20):
    """
    Load the k-th RGB image for the given dataset.
    """
    rgb_path = f"dataRGBD/RGB{dataset}/rgb{dataset}_{k}.png"
    rgb = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)  # convert from BGR to RGB
    return rgb


def build_texture_map(icp_poses, icp_stamps, kinect_data, occupancy_MAP, dataset=20):
    """
    Fully vectorized texture mapping pipeline.
    """
    texture = init_texture_map(occupancy_MAP)

    K = np.array([[585.05, 0, 242.94],
                  [0, 585.05, 315.84],
                  [0, 0,      1     ]])

    T_kinect_to_body = get_kinect_to_body_transform()
    K_inv = np.linalg.inv(K)
    disp_stamps, rgb_stamps = kinect_data

    for i in tqdm(range(len(disp_stamps)), desc="Building Texture Map", unit="frame"):
        t = disp_stamps[i]

        icp_idx = np.argmin(np.abs(icp_stamps - t))
        robot_pose = icp_poses[icp_idx]
        disparity = load_disparity_frame(i+1, dataset)

        rgb_idx = np.argmin(np.abs(rgb_stamps - t))
        rgb = load_rgb_frame(rgb_idx+1, dataset)

        rows, cols = np.where(disparity > 0)
        d = disparity[rows, cols]

        dd = -0.00304 * d + 3.31
        depth = 1.03 / dd
        rgbi = np.round((526.37 * rows + 19276 - 7877.07 * dd) / 585.051).astype(int)
        rgbj = np.round((526.37 * cols + 16662) / 585.051).astype(int)

        valid_depth = (depth > 0) & (depth < 5.0)
        valid_rgb_bounds = (rgbi >= 0) & (rgbi < rgb.shape[0]) & (rgbj >= 0) & (rgbj < rgb.shape[1])
        valid_mask = valid_depth & valid_rgb_bounds

        rows = rows[valid_mask]
        cols = cols[valid_mask]
        depth = depth[valid_mask]
        rgbi = rgbi[valid_mask]
        rgbj = rgbj[valid_mask]

        if len(depth) == 0:
            continue

        pixel_homog = np.vstack((cols, rows, np.ones_like(cols)))
        p_cam = depth * (K_inv @ pixel_homog)

        p_cam_homog = np.vstack((p_cam, np.ones_like(depth)))
        p_body = T_kinect_to_body @ p_cam_homog

        x_body = p_body[0, :]
        y_body = p_body[1, :]
        z_body = p_body[2, :]

        floor_mask = np.abs(z_body) < 0.5

        x_body = x_body[floor_mask]
        y_body = y_body[floor_mask]
        rgbi = rgbi[floor_mask]
        rgbj = rgbj[floor_mask]

        if len(x_body) == 0:
            continue

        cs, sn = np.cos(robot_pose[2]), np.sin(robot_pose[2])
        x_world = cs * x_body - sn * y_body + robot_pose[0]
        y_world = sn * x_body + cs * y_body + robot_pose[1]

        c_map_cols = ((x_world - occupancy_MAP['min'][0]) / occupancy_MAP['res'][0]).astype(int)
        c_map_rows = ((y_world - occupancy_MAP['min'][1]) / occupancy_MAP['res'][1]).astype(int)

        valid_map_bounds = (c_map_cols >= 0) & (c_map_cols < occupancy_MAP['size'][0]) & \
                           (c_map_rows >= 0) & (c_map_rows < occupancy_MAP['size'][1])

        c_map_cols = c_map_cols[valid_map_bounds]
        c_map_rows = c_map_rows[valid_map_bounds]
        rgbi = rgbi[valid_map_bounds]
        rgbj = rgbj[valid_map_bounds]

        texture[c_map_rows, c_map_cols] = rgb[rgbi, rgbj]

    return texture


if __name__ == "__main__":
    from part2_scan_matching import load_lidar_data, scan_matching_trajectory
    from part1_odometry import load_sensor_data, integrate_odometry
    from map_utils import plot_map
    import matplotlib.pyplot as plt
    import numpy as np
    import os

    dataset = 20
    print(f"Part 3: SLAM Mapping Pipeline (Dataset {dataset})")

    print("Loading sensor data & odometry...")
    encoder_data, imu_data = load_sensor_data(dataset)
    odometry_poses, odometry_stamps = integrate_odometry(encoder_data, imu_data)

    # Note: Downsample=1 for the final high-res report map!
    print("Running ICP scan matching (downsample=1)...")
    lidar_data = load_lidar_data(dataset)
    icp_poses, icp_stamps = scan_matching_trajectory(lidar_data, odometry_poses, odometry_stamps, downsample_factor=1)

    print("Loading Kinect RGBD data...")
    kinect_data = load_kinect_data(dataset)

    print("Building occupancy grid map...")
    occupancy_MAP = build_occupancy_map(icp_poses, icp_stamps, lidar_data)

    print("Building texture map...")
    texture_map = build_texture_map(icp_poses, icp_stamps, kinect_data, occupancy_MAP, dataset)

    print("Pipeline complete. Generating plots...")

    traj_cols = (icp_poses[:, 0] - occupancy_MAP['min'][0]) / occupancy_MAP['res'][0]
    traj_rows = (icp_poses[:, 1] - occupancy_MAP['min'][1]) / occupancy_MAP['res'][1]

    save_dir = "plots/part3_mapping"
    os.makedirs(save_dir, exist_ok=True)

    fig1 = plt.figure(figsize=(12, 22))

    plt.subplot(2, 1, 1)
    plot_map(occupancy_MAP['map'], cmap='binary')
    plt.plot(traj_cols, traj_rows, 'r-', linewidth=1.5, label='ICP Trajectory')
    plt.title("Occupancy Grid Map", fontsize=16, fontweight='bold')
    plt.xlabel("Y (cells)", fontsize=12)
    plt.ylabel("X (cells)", fontsize=12)
    plt.axis('equal')
    plt.legend(loc='upper right', fontsize=12)

    plt.subplot(2, 1, 2)
    plt.imshow(texture_map, origin="lower")
    plt.plot(traj_cols, traj_rows, 'r-', linewidth=1.5, label='ICP Trajectory')
    plt.title("Floor Texture Map Overlay", fontsize=16, fontweight='bold')
    plt.xlabel("Y (cells)", fontsize=12)
    plt.ylabel("X (cells)", fontsize=12)
    plt.axis('equal')
    plt.legend(loc='upper right', fontsize=12)

    plt.tight_layout(pad=4.0)
    save_path = f"{save_dir}/final_combined_maps_updated_scanMatchingLogic.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"  -> Saved Combined Map: {save_path}")

    print("Generating fused LiDAR + RGB texture map...")

    occ_map = occupancy_MAP['map']
    nx, ny = occ_map.shape
    fused_map = np.full((ny, nx, 3), 127, dtype=np.uint8)
    fused_map[occ_map.T > 0] = [0, 0, 0]
    fused_map[occ_map.T < 0] = [255, 255, 255]

    painted_mask = np.any(texture_map != 128, axis=-1)
    fused_map[painted_mask] = texture_map[painted_mask]

    fig2 = plt.figure(figsize=(14, 14))

    plt.imshow(fused_map, origin="lower")
    plt.plot(traj_cols, traj_rows, 'r-', linewidth=1.5, label='ICP Trajectory')
    plt.plot(traj_cols[0], traj_rows[0], 'g*', markersize=15, label='Start Point')
    plt.plot(traj_cols[-1], traj_rows[-1], 'bs', markersize=10, label='End Point')

    plt.title("Fused SLAM Output: LiDAR Occupancy + Kinect RGB (Part 3)", fontsize=18, fontweight='bold')
    plt.xlabel("Y (cells)", fontsize=14)
    plt.ylabel("X (cells)", fontsize=14)
    plt.axis('equal')
    plt.axis('off')
    plt.legend(loc='upper right', fontsize=12)
    plt.tight_layout()

    fused_path = f"{save_dir}/fused_texture_occupancy_map_updated_scanMatchingLogic.png"
    plt.savefig(fused_path, dpi=300, bbox_inches="tight", facecolor='white')
    print(f"  -> Saved Fused Map: {fused_path}")

    print("\nExp 1: Log-odds sweep")
    hit_values = [0.4, 0.85, 2.0]
    maps_exp1 = []

    for hit_val in hit_values:
        print(f"  -> Building map with LOG_ODDS_HIT = {hit_val}...")
        M = build_occupancy_map(icp_poses, icp_stamps, lidar_data, log_odds_hit=hit_val)
        maps_exp1.append(M['map'])

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for i, ax in enumerate(axes):
        ax.imshow(maps_exp1[i].T, origin="lower", cmap='binary')
        ax.set_title(f"LOG_ODDS_HIT = {hit_values[i]}")
        ax.axis('off')
        ax.axis('equal')

    plt.tight_layout()
    exp1_path = f"{save_dir}/exp1_log_odds_sweep_updated_scanMatchingLogic.png"
    plt.savefig(exp1_path, dpi=300)
    print(f"  -> Saved Exp 1 plot: {exp1_path}")

    print("\nExp 2: Resolution sweep")
    resolutions = [0.025, 0.05, 0.10, 0.20]
    maps_exp2 = []

    for res in resolutions:
        print(f"  -> Building map with resolution = {res}m...")
        M = build_occupancy_map(icp_poses, icp_stamps, lidar_data, resolution=res)
        maps_exp2.append(M['map'])

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes = axes.flatten()
    for i, ax in enumerate(axes):
        ax.imshow(maps_exp2[i].T, origin="lower", cmap='binary')
        ax.set_title(f"Resolution = {resolutions[i]}m")
        ax.axis('off')
        ax.axis('equal')

    plt.tight_layout()
    exp2_path = f"{save_dir}/exp2_resolution_sweep_updated_scanMatchingLogic.png"
    plt.savefig(exp2_path, dpi=300)
    print(f"  -> Saved Exp 2 plot: {exp2_path}")

    print("\nExp 3: Progressive snapshots")
    percents = [10, 25, 50, 75]
    print("  -> Building map and capturing frames...")
    final_MAP, snapshots = build_occupancy_map(icp_poses, icp_stamps, lidar_data, snapshot_percents=percents)

    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    plot_keys = [10, 25, 50, 75, 100]

    for i, ax in enumerate(axes):
        key = plot_keys[i]
        ax.imshow(snapshots[key].T, origin="lower", cmap='binary')
        ax.set_title(f"Progress: {key}%")
        ax.axis('off')
        ax.axis('equal')

    plt.tight_layout()
    exp3_path = f"{save_dir}/exp3_progressive_snapshots_updated_scanMatchingLogic.png"
    plt.savefig(exp3_path, dpi=300)
    print(f"  -> Saved Exp 3 plot: {exp3_path}")

    print("\nAll Part 3 experiments complete.")
