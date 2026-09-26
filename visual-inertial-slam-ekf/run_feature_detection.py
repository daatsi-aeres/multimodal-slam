import os
import sys
import time
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from geometry_utils import load_data
from feature_detection import build_feature_matrix
from imu_prediction import predict_imu_poses
from landmark_mapping import map_landmarks
from visual_inertial_slam import run_visual_inertial_slam
from visualization import generate_all_plots, plot_timing_summary_bar, ensure_dir


def main():
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'dataset02')
    data_dir = os.path.abspath(data_dir)
    data_path = os.path.join(data_dir, 'dataset02.npy')
    imgs_path = os.path.join(data_dir, 'dataset02_imgs.npy')
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results', 'dataset02_extra_credit')
    results_dir = os.path.abspath(results_dir)
    plots_dir = os.path.join(results_dir, 'plots')
    ensure_dir(results_dir)
    ensure_dir(plots_dir)

    print(f"{'='*60}")
    print(f" Stereo Feature Detection")
    print(f" Dataset: 02")
    print(f"{'='*60}\n")

    print("[1/6] Loading sensor data...")
    t0 = time.time()
    v_t, w_t, timestamps, features_given, K_l, K_r, extL_T_imu, extR_T_imu = load_data(data_path)
    t_load_data = time.time() - t0
    print(f"  Loaded in {t_load_data:.1f}s")
    print(f"  Timesteps: {len(timestamps)}")

    print("[2/6] Loading stereo images from video...")
    t0 = time.time()
    vid_l_path = os.path.join(data_dir, 'dataset02_l.mp4')
    vid_r_path = os.path.join(data_dir, 'dataset02_r.mp4')
    T_steps = len(timestamps)

    print("[3/6] Running feature detection & tracking...")
    t0 = time.time()
    detected_features = build_feature_matrix(
        vid_l_path, vid_r_path, n_timesteps=T_steps,
        max_corners=100, quality_level=0.01, min_distance=30,
        detect_interval=20, verbose=True
    )
    t_detect = time.time() - t0
    print(f"  Feature detection completed in {t_detect:.1f}s")
    print(f"  Detected features shape: {detected_features.shape}")

    np.save(os.path.join(results_dir, 'detected_features.npy'), detected_features)

    import gc; gc.collect()

    print("[4/6] Task 1: IMU Prediction...")
    t0 = time.time()
    imu_poses, _ = predict_imu_poses(v_t, w_t, timestamps)
    t_imu = time.time() - t0
    print(f"  IMU prediction: {t_imu:.1f}s")

    print("[5/6] Task 3: Landmark Mapping with detected features...")
    t0 = time.time()
    V_noise = np.eye(4) * 400.0
    W_noise = np.diag([5e-4]*3 + [1e-4]*3)

    mapping_landmarks, _, mapping_stats = map_landmarks(
        imu_poses, detected_features, K_l, K_r, extL_T_imu, extR_T_imu,
        V_noise=V_noise,
        max_features_per_step=50,
        min_disparity=7.0, max_disparity=20.0
    )
    t_mapping = time.time() - t0
    n_mapped = np.sum(~np.any(np.isnan(mapping_landmarks), axis=1))
    print(f"  Mapped {n_mapped} landmarks in {t_mapping:.1f}s")

    print("[6/6] Task 4: Visual-Inertial SLAM with detected features...")
    t0 = time.time()
    slam_poses, slam_landmarks, slam_stats = run_visual_inertial_slam(
        v_t, w_t, timestamps, detected_features,
        K_l, K_r, extL_T_imu, extR_T_imu,
        W_noise=W_noise, V_noise=V_noise,
        max_features_per_step=50,
        mahal_thresh=15.0,
        innov_thresh=20.0,
        init_landmark_cov=30.0,
        min_disparity=7.0, max_disparity=20.0
    )
    t_slam = time.time() - t0
    print(f"  SLAM completed in {t_slam:.1f}s")

    np.save(os.path.join(results_dir, 'imu_poses.npy'), imu_poses)
    np.save(os.path.join(results_dir, 'slam_poses.npy'), slam_poses)
    np.save(os.path.join(results_dir, 'slam_landmarks.npy'), slam_landmarks)

    generate_all_plots(
        dataset_name="Dataset 02 (Extra Credit)",
        pred_poses=imu_poses,
        slam_poses=slam_poses,
        pred_landmarks=mapping_landmarks,
        slam_landmarks=slam_landmarks,
        slam_stats=slam_stats,
        mapping_stats=mapping_stats,
        output_dir=plots_dir
    )

    timing = {
        'Data Loading': t_load_data,
        'Feature\nDetection': t_detect,
        'IMU Pred': t_imu,
        'Mapping': t_mapping,
        'SLAM': t_slam,
    }
    plot_timing_summary_bar(timing, title="Dataset 02 Extra Credit — Timing",
                            save_path=os.path.join(plots_dir, 'timing_summary.png'))

    print(f"\n{'='*60}")
    print(f" DONE — Results saved to: {results_dir}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
