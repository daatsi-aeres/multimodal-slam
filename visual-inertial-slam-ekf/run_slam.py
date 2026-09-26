import os
import sys
import time
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from geometry_utils import load_data
from imu_prediction import predict_imu_poses
from landmark_mapping import map_landmarks
from visual_inertial_slam import run_visual_inertial_slam
from visualization import generate_all_plots, plot_timing_summary_bar, ensure_dir


def main():
    parser = argparse.ArgumentParser(description="Visual-Inertial SLAM")
    parser.add_argument('--dataset', type=str, default='00',
                        choices=['00', '01', '02'])
    parser.add_argument('--max_features', type=int, default=150)
    parser.add_argument('--v_noise', type=float, default=200.0)
    parser.add_argument('--w_noise_v', type=float, default=1e-3)
    parser.add_argument('--w_noise_w', type=float, default=1e-4)
    parser.add_argument('--mahal_thresh', type=float, default=30.0)
    parser.add_argument('--innov_thresh', type=float, default=80.0)
    parser.add_argument('--init_cov', type=float, default=100.0)
    parser.add_argument('--min_disparity', type=float, default=5.0)
    parser.add_argument('--max_disparity', type=float, default=150.0)
    parser.add_argument('--skip_mapping', action='store_true')
    parser.add_argument('--skip_slam', action='store_true')
    args = parser.parse_args()

    W_noise = np.diag([args.w_noise_v]*3 + [args.w_noise_w]*3)
    V_noise = np.eye(4) * args.v_noise

    ds = args.dataset
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data',
                             f'dataset{ds}', f'dataset{ds}.npy')
    data_path = os.path.abspath(data_path)
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results', f'dataset{ds}')
    results_dir = os.path.abspath(results_dir)
    plots_dir = os.path.join(results_dir, 'plots')
    ensure_dir(results_dir)
    ensure_dir(plots_dir)

    print(f"{'='*60}")
    print(f" Visual-Inertial SLAM")
    print(f" Dataset: {ds}")
    print(f" Data:    {data_path}")
    print(f" Output:  {results_dir}")
    print(f"{'='*60}\n")

    print("[1/5] Loading data...")
    t0 = time.time()
    v_t, w_t, timestamps, features, K_l, K_r, extL_T_imu, extR_T_imu = load_data(data_path)
    t_load = time.time() - t0
    print(f"  Loaded in {t_load:.1f}s")
    print(f"  Timesteps: {len(timestamps)}")
    print(f"  Features:  {features.shape[1]} total landmarks")
    print(f"  dt range:  [{np.diff(timestamps).min():.4f}, {np.diff(timestamps).max():.4f}]s")
    print()

    print("[2/5] Task 1: IMU Localization (EKF Prediction)...")
    t0 = time.time()
    imu_poses, imu_covs = predict_imu_poses(v_t, w_t, timestamps)
    t_task1 = time.time() - t0
    print(f"  Completed in {t_task1:.2f}s")
    print(f"  Final position: ({imu_poses[-1,0,3]:.2f}, {imu_poses[-1,1,3]:.2f}, {imu_poses[-1,2,3]:.2f})")
    print()

    mapping_landmarks = None
    mapping_stats = None
    if not args.skip_mapping:
        print(f"[3/5] Task 3: Landmark Mapping (max {args.max_features} features/step)...")
        t0 = time.time()
        mapping_landmarks, mapping_cov, mapping_stats = map_landmarks(
            imu_poses, features, K_l, K_r, extL_T_imu, extR_T_imu,
            V_noise=V_noise,
            max_features_per_step=args.max_features,
            min_disparity=args.min_disparity,
            max_disparity=args.max_disparity
        )
        t_task3 = time.time() - t0
        n_mapped = np.sum(~np.any(np.isnan(mapping_landmarks), axis=1))
        print(f"  Completed in {t_task3:.2f}s")
        print(f"  Mapped landmarks: {n_mapped} / {features.shape[1]}")
        print()
    else:
        t_task3 = 0
        print("[3/5] Task 3: Skipped\n")

    slam_poses = None
    slam_landmarks = None
    slam_stats = None
    if not args.skip_slam:
        print(f"[4/5] Task 4: Visual-Inertial SLAM (max {args.max_features} features/step)...")
        t0 = time.time()
        slam_poses, slam_landmarks, slam_stats = run_visual_inertial_slam(
            v_t, w_t, timestamps, features,
            K_l, K_r, extL_T_imu, extR_T_imu,
            W_noise=W_noise, V_noise=V_noise,
            max_features_per_step=args.max_features,
            mahal_thresh=args.mahal_thresh,
            innov_thresh=args.innov_thresh,
            init_landmark_cov=args.init_cov,
            min_disparity=args.min_disparity,
            max_disparity=args.max_disparity
        )
        t_task4 = time.time() - t0
        n_slam_lm = np.sum(~np.any(np.isnan(slam_landmarks), axis=1))
        print(f"  Completed in {t_task4:.2f}s")
        print(f"  SLAM landmarks: {n_slam_lm} / {features.shape[1]}")
        print(f"  Final SLAM pos: ({slam_poses[-1,0,3]:.2f}, {slam_poses[-1,1,3]:.2f}, {slam_poses[-1,2,3]:.2f})")
        print()
    else:
        t_task4 = 0
        print("[4/5] Task 4: Skipped\n")

    print("[5/5] Saving results and generating plots...")
    np.save(os.path.join(results_dir, 'imu_poses.npy'), imu_poses)
    if mapping_landmarks is not None:
        np.save(os.path.join(results_dir, 'mapping_landmarks.npy'), mapping_landmarks)
    if slam_poses is not None:
        np.save(os.path.join(results_dir, 'slam_poses.npy'), slam_poses)
    if slam_landmarks is not None:
        np.save(os.path.join(results_dir, 'slam_landmarks.npy'), slam_landmarks)

    if slam_poses is None:
        slam_poses = imu_poses
    generate_all_plots(
        dataset_name=f"Dataset {ds}",
        pred_poses=imu_poses,
        slam_poses=slam_poses,
        pred_landmarks=mapping_landmarks,
        slam_landmarks=slam_landmarks,
        slam_stats=slam_stats if slam_stats else {},
        mapping_stats=mapping_stats if mapping_stats else {},
        output_dir=plots_dir
    )

    timing = {
        'Data Loading': t_load,
        'Task 1\n(IMU Pred)': t_task1,
        'Task 3\n(Mapping)': t_task3,
        'Task 4\n(SLAM)': t_task4,
    }
    plot_timing_summary_bar(
        timing,
        title=f"Dataset {ds} — Total Timing",
        save_path=os.path.join(plots_dir, 'timing_summary.png')
    )

    total_time = t_load + t_task1 + t_task3 + t_task4
    print(f"\n{'='*60}")
    print(f" DONE — Total time: {total_time:.1f}s")
    print(f" Results saved to: {results_dir}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
