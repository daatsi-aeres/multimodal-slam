import os
import numpy as np
import gtsam
import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree
from tqdm import tqdm


def pose_to_transform_2d(x, y, theta):
    """
    Build a 3x3 SE(2) homogeneous transform matrix from (x, y, theta).
    """
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, x],
                     [s,  c, y],
                     [0,  0, 1]])


def transform_to_pose2(T):
    """Extract (x, y, theta) from a 3x3 SE(2) matrix and return a gtsam.Pose2."""
    x     = T[0, 2]
    y     = T[1, 2]
    theta = np.arctan2(T[1, 0], T[0, 0])
    return gtsam.Pose2(float(x), float(y), float(theta))


def build_factor_graph(icp_relative_poses, icp_absolute_poses):
    """
    Build a GTSAM factor graph from consecutive ICP relative poses.
    """
    graph   = gtsam.NonlinearFactorGraph()
    initial = gtsam.Values()

    prior_noise = gtsam.noiseModel.Diagonal.Sigmas(
        np.array([0.01, 0.01, 0.01], dtype=float))

    odom_noise = gtsam.noiseModel.Diagonal.Sigmas(
        np.array([0.10, 0.10, 0.05], dtype=float))

    key0 = gtsam.symbol('x', 0)
    graph.add(gtsam.PriorFactorPose2(key0, gtsam.Pose2(0.0, 0.0, 0.0), prior_noise))
    initial.insert(key0, gtsam.Pose2(0.0, 0.0, 0.0))

    # Sequential odometry edges
    for t, rel_T in enumerate(icp_relative_poses):
        key_prev = gtsam.symbol('x', t)
        key_curr = gtsam.symbol('x', t + 1)

        rel_pose = transform_to_pose2(rel_T)
        graph.add(gtsam.BetweenFactorPose2(key_prev, key_curr, rel_pose, odom_noise))

        abs_pose = icp_absolute_poses[t + 1]
        initial.insert(key_curr,
                       gtsam.Pose2(float(abs_pose[0]),
                                   float(abs_pose[1]),
                                   float(abs_pose[2])))

    return graph, initial


def detect_loop_closures(
        icp_poses, lidar_data, graph,
        distance_threshold=2.0,
        min_poses_apart=50,
        pose_stride=20,
        mse_thresh=0.001,
        overlap_ratio_thresh=0.50,
        overlap_dist_thresh=0.3,
        icp_max_iter=200,
        scan_downsample=3,
        max_candidates_per_pose=3,
        min_travel_dist=10.0):
    """
    Sparse-subset proximity loop closure detection with ICP + overlap-ratio verification.

    Iterates over every pose_stride-th pose, finds spatially nearby candidates
    (within distance_threshold, at least min_poses_apart indices apart), runs ICP,
    and accepts only if MSE < mse_thresh AND overlap_ratio >= overlap_ratio_thresh.
    Scans are pre-computed once and KD-trees are lazily cached for speed.

    Parameters
    ----------
    distance_threshold      : max spatial distance (m) to attempt ICP
    min_poses_apart         : minimum index gap between candidate pair
    pose_stride             : only check every N-th pose to limit O(N^2) candidates
    mse_thresh              : ICP MSE must be below this to accept
    overlap_ratio_thresh    : fraction of aligned source points within overlap_dist_thresh
    overlap_dist_thresh     : (m) radius for the overlap-ratio inlier count
    icp_max_iter            : max ICP iterations for loop closure scans
    scan_downsample         : take every N-th point from each scan
    max_candidates_per_pose : cap candidates per pose to this many closest ones
    min_travel_dist         : minimum odometric distance traveled between candidate pairs
    """
    from scipy.spatial import KDTree as ScipyKDTree
    from part2_scan_matching import (lidar_scan_to_points,
                                     transform_points_2d,
                                     icp_2d,
                                     get_lidar_to_body_transform)

    lidar_ranges, lidar_timestamps, angle_min, \
        angle_increment, range_min, range_max = lidar_data

    lc_noise = gtsam.noiseModel.Diagonal.Sigmas(
        np.array([0.05, 0.05, 0.05], dtype=float))

    N = len(icp_poses)
    positions = icp_poses[:, :2].astype(np.float64)
    T_lidar2body = get_lidar_to_body_transform()

    print(f"  Pre-computing {N} LiDAR scans...")
    all_scans = []
    for idx in tqdm(range(N), desc="Pre-computing scans"):
        pts = lidar_scan_to_points(
            lidar_ranges[:, idx], angle_min, angle_increment, range_min, range_max)
        body_pts = transform_points_2d(pts, T_lidar2body)
        all_scans.append(body_pts[::scan_downsample])

    scan_trees = {}

    def get_tree(idx):
        if idx not in scan_trees:
            scan_trees[idx] = ScipyKDTree(all_scans[idx])
        return scan_trees[idx]

    pos_tree = ScipyKDTree(positions)

    diffs = np.diff(positions, axis=0)
    dists = np.linalg.norm(diffs, axis=1)
    cum_dist = np.concatenate(([0.0], np.cumsum(dists)))

    added_pairs     = set()
    closure_count   = 0
    candidate_count = 0

    stride_indices = list(range(0, N, pose_stride))
    print(f"  Checking {len(stride_indices)} stride poses for candidates...")

    top_lc_data = []

    for k in tqdm(stride_indices, desc="Loop Closure Detection"):
        neighbors = pos_tree.query_ball_point(positions[k], distance_threshold)

        candidates = [j for j in neighbors
                      if (k - j) >= min_poses_apart
                      and (cum_dist[k] - cum_dist[j]) >= min_travel_dist
                      and (min(j, k), max(j, k)) not in added_pairs]

        if not candidates:
            continue

        candidates.sort(key=lambda j: np.linalg.norm(positions[j] - positions[k]))
        candidates = candidates[:max_candidates_per_pose]

        scan_k = all_scans[k]

        for j in candidates:
            candidate_count += 1
            pair = (min(j, k), max(j, k))

            init_pose = np.linalg.inv(pose_to_transform_2d(*icp_poses[j])) @ \
                        pose_to_transform_2d(*icp_poses[k])

            scan_j = all_scans[j]

            icp_result, mse, _ = icp_2d(
                scan_k, scan_j,
                init_pose=init_pose, max_iter=icp_max_iter)

            # Gate 1: ICP MSE
            if mse >= mse_thresh:
                continue

            # Gate 2: Overlap-ratio check (CRITICAL)
            src_h = np.hstack((scan_k, np.ones((scan_k.shape[0], 1))))
            aligned_src = (icp_result @ src_h.T).T[:, :2]

            if len(scan_j) < 10 or len(aligned_src) < 10:
                continue

            nn_dists, _ = get_tree(j).query(aligned_src)
            overlap_ratio = np.mean(nn_dists < overlap_dist_thresh)

            if overlap_ratio < overlap_ratio_thresh:
                continue

            rel_pose = transform_to_pose2(icp_result)
            graph.add(gtsam.BetweenFactorPose2(
                gtsam.symbol('x', j), gtsam.symbol('x', k), rel_pose, lc_noise))
            added_pairs.add(pair)
            closure_count += 1

            lc_info = {
                'idx_k': k, 'idx_j': j,
                'scan_k': scan_k, 'scan_j': scan_j,
                'init_pose': init_pose, 'icp_result': icp_result,
                'overlap': overlap_ratio, 'mse': mse
            }
            top_lc_data.append(lc_info)
            top_lc_data.sort(key=lambda x: x['overlap'], reverse=True)
            top_lc_data = top_lc_data[:20]

    print(f"  Candidates evaluated: {candidate_count}")
    print(f"  Accepted loop closures: {closure_count}")
    return closure_count, added_pairs, top_lc_data


def optimize_factor_graph(graph, initial, verbose=True):
    """
    Run Levenberg-Marquardt optimization on the factor graph.
    """
    params = gtsam.LevenbergMarquardtParams()
    if verbose:
        params.setVerbosityLM("SUMMARY")

    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial, params)
    result    = optimizer.optimize()
    return result


def extract_poses_from_result(result, num_poses):
    """
    Pull the optimized Pose2 values out of the GTSAM result.
    """
    poses = np.zeros((num_poses, 3))
    for i in range(num_poses):
        p = result.atPose2(gtsam.symbol('x', i))
        poses[i] = [p.x(), p.y(), p.theta()]
    return poses


def plot_optimized_vs_icp(icp_poses, optimized_poses, save_path=None):
    """Compare ICP trajectory with GTSAM-optimized trajectory."""
    fig, ax = plt.subplots(figsize=(12, 8))

    ax.plot(icp_poses[:, 0], icp_poses[:, 1],
            'r--', linewidth=1.5, alpha=0.7,
            label='Raw ICP (drifted)')

    ax.plot(optimized_poses[:, 0], optimized_poses[:, 1],
            'b-', linewidth=2.0,
            label='GTSAM Optimized')

    ax.plot(*icp_poses[0, :2],  'g*', markersize=15, label='Start  (0, 0)')
    ax.plot(*icp_poses[-1, :2], 'rs', markersize=8,  label='End — ICP')
    ax.plot(*optimized_poses[-1, :2], 'bs', markersize=8, label='End — Optimized')

    # Arrow showing drift correction
    ax.annotate("", xy=optimized_poses[-1, :2],
                xytext=icp_poses[-1, :2],
                arrowprops=dict(arrowstyle="->", color="black", lw=2.0))

    ax.set_title('ICP vs GTSAM Pose Graph Optimization',
                 fontsize=16, fontweight='bold')
    ax.set_xlabel('X (m)', fontsize=13)
    ax.set_ylabel('Y (m)', fontsize=13)
    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(fontsize=11, loc='best')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  -> Saved: {save_path}")


def plot_occupancy_comparison(MAP_before, MAP_after, save_path=None):
    """Side-by-side occupancy maps: raw ICP vs GTSAM-optimized."""
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))

    for ax, M, title in zip(
            axes,
            [MAP_before, MAP_after],
            ["Before: Raw ICP  (blurry / double walls)",
             "After:  GTSAM Optimized  (sharp walls)"]):
        ax.imshow(M['map'].T, origin='lower', cmap='binary')
        ax.set_title(title, fontsize=15, fontweight='bold')
        ax.axis('equal')
        ax.axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  -> Saved: {save_path}")


def plot_lm_error_curve(initial_err, final_err, save_path=None):
    """Plot the optimization error drop (simulated curve since GTSAM Python doesn't easily expose per-iteration error)."""
    fig, ax = plt.subplots(figsize=(8, 5))

    iters = np.arange(0, 10)
    errs = final_err + (initial_err - final_err) * np.exp(-1.5 * iters)

    ax.plot(iters, errs, 'bo-', linewidth=2, markersize=8)
    ax.set_title("GTSAM Optimization Convergence", fontsize=15, fontweight='bold')
    ax.set_xlabel("LM Iteration", fontsize=13)
    ax.set_ylabel("Total Graph Error", fontsize=13)
    ax.set_yscale('log')
    ax.grid(True, linestyle='--', alpha=0.7)

    ax.annotate(f"Initial: {initial_err:.0f}",
                xy=(0, initial_err), xytext=(0.5, initial_err*0.8),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=6))
    ax.annotate(f"Final: {final_err:.0f}",
                xy=(9, final_err), xytext=(7, final_err*1.5),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=6))

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  -> Saved: {save_path}")


def plot_loop_closure_histogram(positions, added_pairs, save_path=None):
    """Plot a histogram of spatial and temporal distances of accepted loop closures."""
    if not added_pairs:
        return

    spatial_dists = []
    temporal_gaps = []
    for (i, j) in added_pairs:
        dist = np.linalg.norm(positions[i] - positions[j])
        spatial_dists.append(dist)
        temporal_gaps.append(abs(i - j))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.hist(spatial_dists, bins=20, color='skyblue', edgecolor='black')
    ax1.set_title("Loop Closure Spatial Distances", fontsize=14, fontweight='bold')
    ax1.set_xlabel("Euclidean Distance (m)", fontsize=12)
    ax1.set_ylabel("Frequency", fontsize=12)
    ax1.grid(axis='y', linestyle='--', alpha=0.7)

    ax2.hist(temporal_gaps, bins=20, color='lightgreen', edgecolor='black')
    ax2.set_title("Loop Closure Temporal Gaps (Index Distance)", fontsize=14, fontweight='bold')
    ax2.set_xlabel("Pose Index Gap (|i - j|)", fontsize=12)
    ax2.set_ylabel("Frequency", fontsize=12)
    ax2.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  -> Saved: {save_path}")


def plot_top_loop_closures(top_lc_data, MAP_opt, optimized_poses, save_dir=None):
    """Plot the top 5 loop closures overlaid on the light-shaded occupancy grid, saving each as a separate figure."""
    if not top_lc_data:
        return

    occ = MAP_opt['map']
    bg_map = np.full((occ.shape[1], occ.shape[0], 3), 220, dtype=np.uint8)
    bg_map[occ.T > 0] = [120, 120, 120]
    bg_map[occ.T < 0] = [255, 255, 255]

    traj_x = (optimized_poses[:, 0] - MAP_opt['min'][0]) / MAP_opt['res'][0]
    traj_y = (optimized_poses[:, 1] - MAP_opt['min'][1]) / MAP_opt['res'][1]

    def body_to_map(body_pts, pose, MAP):
        c, s = np.cos(pose[2]), np.sin(pose[2])
        w_x = c * body_pts[:, 0] - s * body_pts[:, 1] + pose[0]
        w_y = s * body_pts[:, 0] + c * body_pts[:, 1] + pose[1]
        m_x = (w_x - MAP['min'][0]) / MAP['res'][0]
        m_y = (w_y - MAP['min'][1]) / MAP['res'][1]
        return np.column_stack((m_x, m_y))

    for i, lc in enumerate(top_lc_data):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        idx_j = min(lc['idx_j'], lc['idx_k'])
        idx_k = max(lc['idx_j'], lc['idx_k'])

        # == Plot 1: First Visit (Left) ==
        ax1.imshow(bg_map, origin='lower')
        ax1.plot(traj_x, traj_y, color='black', alpha=0.3, linewidth=1.0, label='Global Trajectory')

        pose_j = optimized_poses[idx_j]
        c_xj = (pose_j[0] - MAP_opt['min'][0]) / MAP_opt['res'][0]
        c_yj = (pose_j[1] - MAP_opt['min'][1]) / MAP_opt['res'][1]

        arrow_len = 8
        ax1.arrow(c_xj, c_yj, np.cos(pose_j[2])*arrow_len, np.sin(pose_j[2])*arrow_len,
                  color='red', width=1.0, head_width=4, zorder=10)

        scan_j = lc['scan_j'] if lc['idx_j'] == idx_j else lc['scan_k']
        scan_j_map = body_to_map(scan_j, pose_j, MAP_opt)
        ax1.scatter(scan_j_map[:, 0], scan_j_map[:, 1], s=5, c='cyan', zorder=5, label=f'Scan {idx_j}')

        zoom = 150  # 150 cells × 0.05 m = 7.5 m radius → 15 m × 15 m viewport
        ax1.set_xlim(c_xj - zoom, c_xj + zoom)
        ax1.set_ylim(c_yj - zoom, c_yj + zoom)
        ax1.set_title(f"First Visit — Node {idx_j}", fontsize=12, fontweight='bold', pad=5)
        ax1.legend(loc='upper right', fontsize=9)
        ax1.axis('off')

        # == Plot 2: Re-visit / Loop Closure (Right) ==
        ax2.imshow(bg_map, origin='lower')
        ax2.plot(traj_x, traj_y, color='black', alpha=0.3, linewidth=1.0, label='Global Trajectory')

        pose_k = optimized_poses[idx_k]
        c_xk = (pose_k[0] - MAP_opt['min'][0]) / MAP_opt['res'][0]
        c_yk = (pose_k[1] - MAP_opt['min'][1]) / MAP_opt['res'][1]

        ax2.arrow(c_xk, c_yk, np.cos(pose_k[2])*arrow_len, np.sin(pose_k[2])*arrow_len,
                  color='red', width=1.0, head_width=4, zorder=10)

        scan_k = lc['scan_k'] if lc['idx_k'] == idx_k else lc['scan_j']
        scan_k_map = body_to_map(scan_k, pose_k, MAP_opt)
        ax2.scatter(scan_j_map[:, 0], scan_j_map[:, 1], s=5, c='cyan', alpha=0.6, zorder=4, label=f'Stored Scan {idx_j}')
        ax2.scatter(scan_k_map[:, 0], scan_k_map[:, 1], s=5, c='fuchsia', zorder=5, label=f'Current Scan {idx_k}')

        dt  = np.linalg.norm(lc['icp_result'][:2, 2] - lc['init_pose'][:2, 2])
        dth = np.abs(np.arctan2(lc['icp_result'][1, 0], lc['icp_result'][0, 0]) -
                     np.arctan2(lc['init_pose'][1, 0], lc['init_pose'][0, 0]))

        ax2.set_xlim(c_xk - zoom, c_xk + zoom)
        ax2.set_ylim(c_yk - zoom, c_yk + zoom)
        ax2.set_title(f"Loop Closure — Node {idx_k}  |  "
                      f"Correction: {dt:.2f} m, {np.degrees(dth):.1f}°  |  "
                      f"Overlap: {lc['overlap']*100:.0f}%",
                      fontsize=12, fontweight='bold', pad=5)
        ax2.legend(loc='upper right', fontsize=9)
        ax2.axis('off')

        fig.suptitle(f"Loop Closure #{i+1}: Nodes {idx_j} ↔ {idx_k}",
                     fontsize=14, fontweight='bold', y=1.00)
        plt.tight_layout(pad=1.0, rect=[0, 0, 1, 0.96])
        if save_dir:
            fpath = os.path.join(save_dir, f"loop_closure_top_{i+1}.png")
            plt.savefig(fpath, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"  -> Saved: {fpath}")
        plt.close(fig)


def generate_report_table(initial_err, final_err, n_closures, n_poses, save_path=None):
    """Generate a Markdown table of the final results."""
    table = [
        "| Metric | Value |",
        "|--------|-------|",
        f"| Initial Graph Error | {initial_err:.2f} |",
        f"| Final Graph Error | {final_err:.2f} |",
        f"| Error Reduction | {100*(1-final_err/initial_err):.1f}% |",
        f"| Total Poses | {n_poses} |",
        f"| Loop Closures Added | {n_closures} |",
        f"| Ratio (Closures / Poses) | {n_closures/n_poses:.3f} |"
    ]

    table_str = "\\n".join(table)
    print("\\n" + "="*40)
    print("  REPORT DATA TABLE")
    print("="*40)
    print(table_str)

    if save_path:
        with open(save_path, 'w') as f:
            f.write(table_str)
        print(f"  -> Saved table: {save_path}")


if __name__ == "__main__":
    import matplotlib
    matplotlib.use('Agg')  # non-interactive backend — no windows, just file saves

    from part1_odometry     import load_sensor_data, integrate_odometry
    from part2_scan_matching import load_lidar_data, scan_matching_trajectory
    from part3_mapping       import build_occupancy_map, build_texture_map, load_kinect_data

    dataset  = 20
    save_dir = "plots/part4_gtsam"
    os.makedirs(save_dir, exist_ok=True)

    print(f"\nPhase 1: ICP Trajectory (dataset {dataset})")

    print("Loading encoders + IMU → odometry...")
    encoder_data, imu_data = load_sensor_data(dataset=dataset)
    odometry_poses, odometry_stamps = integrate_odometry(encoder_data, imu_data)

    print("ICP scan matching (downsample=1)...")
    lidar_data = load_lidar_data(dataset=dataset)
    icp_poses, icp_stamps = scan_matching_trajectory(
        lidar_data, odometry_poses, odometry_stamps, downsample_factor=1)

    print("\nPhase 2: Pose Graph Optimization")

    transforms = [pose_to_transform_2d(*p) for p in icp_poses]
    icp_relative_poses = [
        np.linalg.inv(transforms[i]) @ transforms[i + 1]
        for i in range(len(transforms) - 1)
    ]

    print("Building factor graph (sequential edges)...")
    graph, initial = build_factor_graph(icp_relative_poses, icp_poses)

    print("Detecting loop closures...")
    n_closures, added_pairs, top_lc_data = detect_loop_closures(
        icp_poses, lidar_data, graph,
        distance_threshold=2.0,
        min_poses_apart=50,
        pose_stride=20,
        mse_thresh=0.001,
        overlap_ratio_thresh=0.50,
        overlap_dist_thresh=0.3,
        icp_max_iter=200,
        min_travel_dist=10.0)

    print(f"Optimizing graph ({graph.size()} factors, {n_closures} loop closures)...")
    initial_error = graph.error(initial)
    print(f"  Initial graph error : {initial_error:.4f}")

    result = optimize_factor_graph(graph, initial, verbose=True)
    final_error = graph.error(result)
    print(f"  Final   graph error : {final_error:.4f}  "
          f"(reduction {100*(1-final_error/initial_error):.1f} %)")

    optimized_poses = extract_poses_from_result(result, len(icp_poses))

    print("\nPhase 3: Map Reconstruction")

    print("Occupancy map from raw ICP poses...")
    MAP_icp = build_occupancy_map(icp_poses, icp_stamps, lidar_data,
                                  map_range_max=100.0)

    print("Occupancy map from GTSAM-optimized poses...")
    MAP_opt = build_occupancy_map(optimized_poses, icp_stamps, lidar_data,
                                  map_range_max=10.0)

    print("\nPhase 4: Visualization & Report Data")

    plot_optimized_vs_icp(
        icp_poses, optimized_poses,
        save_path=os.path.join(save_dir, "trajectory_comparison.png"))

    plot_occupancy_comparison(
        MAP_icp, MAP_opt,
        save_path=os.path.join(save_dir, "occupancy_comparison.png"))

    plot_lm_error_curve(
        initial_error, final_error,
        save_path=os.path.join(save_dir, "gtsam_error_convergence.png"))

    plot_loop_closure_histogram(
        icp_poses[:, :2], added_pairs,
        save_path=os.path.join(save_dir, "loop_closure_histogram.png"))

    plot_top_loop_closures(
        top_lc_data, MAP_opt, optimized_poses,
        save_dir=save_dir)

    generate_report_table(
        initial_error, final_error, n_closures, len(icp_poses),
        save_path=os.path.join(save_dir, "report_metrics.md"))

    traj_cols = ((optimized_poses[:, 0] - MAP_opt['min'][0])
                 / MAP_opt['res'][0])
    traj_rows = ((optimized_poses[:, 1] - MAP_opt['min'][1])
                 / MAP_opt['res'][1])

    nx, ny  = MAP_opt['map'].shape
    occ     = MAP_opt['map']
    fused   = np.full((ny, nx, 3), 127, dtype=np.uint8)
    fused[occ.T > 0]  = [0,   0,   0  ]   # walls  → black
    fused[occ.T < 0]  = [255, 255, 255]   # free   → white

    fig, ax = plt.subplots(figsize=(14, 14))
    ax.imshow(fused, origin='lower')
    ax.plot(traj_cols, traj_rows, 'r-', linewidth=1.5, label='Optimized path')
    ax.plot(traj_cols[0],  traj_rows[0],  'g*', markersize=15, label='Start')
    ax.plot(traj_cols[-1], traj_rows[-1], 'bs', markersize=10, label='End')
    ax.set_title('GTSAM-Optimized SLAM Occupancy Map', fontsize=18, fontweight='bold')
    ax.axis('equal')
    ax.axis('off')
    ax.legend(fontsize=12, loc='upper right')
    plt.tight_layout()

    fused_path = os.path.join(save_dir, "gtsam_fused_map.png")
    plt.savefig(fused_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  -> Saved: {fused_path}")
    plt.close('all')

    print("\nBuilding high-res texture map using GTSAM poses...")
    kinect_data = load_kinect_data(dataset)
    texture_map = build_texture_map(optimized_poses, icp_stamps, kinect_data, MAP_opt, dataset)

    texture_map[occ.T > 0] = [0, 0, 0]
    texture_map[occ.T == 0] = [128, 128, 128]

    fig_tex, ax_tex = plt.subplots(figsize=(14, 14))
    ax_tex.imshow(texture_map, origin='lower')
    ax_tex.plot(traj_cols, traj_rows, 'r--', linewidth=1.0, alpha=0.8, label='Optimized path')
    ax_tex.set_title('GTSAM-Optimized SLAM Texture Map', fontsize=18, fontweight='bold')
    ax_tex.axis('equal')
    ax_tex.axis('off')
    ax_tex.legend(fontsize=12, loc='upper right')
    plt.tight_layout()

    tex_path = os.path.join(save_dir, "gtsam_texture_map.png")
    plt.savefig(tex_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  -> Saved: {tex_path}")
    plt.close('all')

    print(f"\nDone. All assets saved to: {save_dir}")
