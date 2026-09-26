import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from transforms3d.euler import mat2euler


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def plot_trajectory_2d(poses, title="IMU Trajectory", save_path=None,
                       show_ori=True, label="Trajectory", ax=None, color='r'):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    else:
        fig = ax.figure

    n_pose = poses.shape[0]
    ax.plot(poses[:, 0, 3], poses[:, 1, 3], color=color, linewidth=1.5, label=label)
    ax.scatter(poses[0, 0, 3], poses[0, 1, 3], marker='s', s=80, c='green',
               zorder=5, label="Start")
    ax.scatter(poses[-1, 0, 3], poses[-1, 1, 3], marker='*', s=120, c='red',
               zorder=5, label="End")

    if show_ori:
        step = max(int(n_pose / 50), 1)
        select = list(range(0, n_pose, step))
        yaw_list = []
        for i in select:
            _, _, yaw = mat2euler(poses[i, :3, :3])
            yaw_list.append(yaw)
        dx = np.cos(yaw_list)
        dy = np.sin(yaw_list)
        norm = np.sqrt(dx**2 + dy**2)
        dx, dy = dx / norm, dy / norm
        ax.quiver(poses[select, 0, 3], poses[select, 1, 3], dx, dy,
                  color='blue', units='xy', width=0.3, scale=2,
                  headlength=3, headaxislength=2.5, alpha=0.6)

    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.axis('equal')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)

    if save_path:
        ensure_dir(os.path.dirname(save_path))
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  [SAVED] {save_path}")

    return fig, ax


def plot_trajectory_comparison(pred_poses, slam_poses, title="Trajectory Comparison",
                               save_path=None):
    fig, ax = plt.subplots(figsize=(10, 8))

    ax.plot(pred_poses[:, 0, 3], pred_poses[:, 1, 3],
            color='orange', linewidth=1.5, alpha=0.7, label='IMU Only (Prediction)')
    ax.plot(slam_poses[:, 0, 3], slam_poses[:, 1, 3],
            color='blue', linewidth=1.5, label='SLAM Corrected')

    ax.scatter(slam_poses[0, 0, 3], slam_poses[0, 1, 3],
               marker='s', s=80, c='green', zorder=5, label='Start')
    ax.scatter(slam_poses[-1, 0, 3], slam_poses[-1, 1, 3],
               marker='*', s=120, c='red', zorder=5, label='End')

    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.axis('equal')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)

    if save_path:
        ensure_dir(os.path.dirname(save_path))
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  [SAVED] {save_path}")

    plt.close(fig)
    return fig, ax


def plot_landmarks_2d(landmark_mean, poses=None, slam_poses=None,
                      title="Estimated Landmarks", save_path=None):
    fig, ax = plt.subplots(figsize=(12, 10))

    valid = ~np.any(np.isnan(landmark_mean), axis=1)
    lm = landmark_mean[valid]

    ax.scatter(lm[:, 0], lm[:, 1], s=2, alpha=0.3, c='gray', label=f'Landmarks ({len(lm)})')

    if poses is not None:
        ax.plot(poses[:, 0, 3], poses[:, 1, 3],
                color='orange', linewidth=1.5, alpha=0.7, label='IMU Only')

    if slam_poses is not None:
        ax.plot(slam_poses[:, 0, 3], slam_poses[:, 1, 3],
                color='blue', linewidth=2.0, label='SLAM')

    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.axis('equal')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)

    if save_path:
        ensure_dir(os.path.dirname(save_path))
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  [SAVED] {save_path}")

    plt.close(fig)
    return fig, ax


def plot_covariance_trace(cov_traces, title="Pose Covariance Trace",
                          save_path=None):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(cov_traces, color='purple', linewidth=1.0)
    ax.set_xlabel('Timestep', fontsize=12)
    ax.set_ylabel('Trace(Σ_pose)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

    if save_path:
        ensure_dir(os.path.dirname(save_path))
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  [SAVED] {save_path}")

    plt.close(fig)
    return fig, ax


def plot_innovation_stats(innovations, title="EKF Innovation Magnitude",
                          save_path=None):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(innovations, color='teal', linewidth=0.8, alpha=0.7)
    ax.set_xlabel('Update Step', fontsize=12)
    ax.set_ylabel('Mean Innovation (px)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    if save_path:
        ensure_dir(os.path.dirname(save_path))
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  [SAVED] {save_path}")

    plt.close(fig)
    return fig, ax


def plot_landmark_count(n_observed, n_new=None,
                        title="Features per Timestep", save_path=None):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(n_observed, color='steelblue', linewidth=0.8, alpha=0.7,
            label='Observed')
    if n_new is not None:
        ax.plot(n_new, color='coral', linewidth=0.8, alpha=0.7,
                label='Newly Initialized')
    ax.set_xlabel('Timestep', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)

    if save_path:
        ensure_dir(os.path.dirname(save_path))
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  [SAVED] {save_path}")

    plt.close(fig)
    return fig, ax


def plot_timing_breakdown(prediction_times, update_times,
                          title="EKF Computation Time", save_path=None):
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    steps = np.arange(len(prediction_times))

    axes[0].fill_between(steps, prediction_times, alpha=0.7, color='skyblue',
                         label='Prediction')
    axes[0].set_ylabel('Time (s)', fontsize=11)
    axes[0].set_title(title, fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    axes[1].fill_between(steps, update_times, alpha=0.7, color='salmon',
                         label='Update')
    axes[1].set_xlabel('Timestep', fontsize=12)
    axes[1].set_ylabel('Time (s)', fontsize=11)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    total_pred = sum(prediction_times)
    total_upd = sum(update_times)
    fig.text(0.99, 0.01,
             f'Total: Pred={total_pred:.1f}s, Update={total_upd:.1f}s, '
             f'Total={total_pred+total_upd:.1f}s',
             ha='right', fontsize=9, style='italic')

    plt.tight_layout()

    if save_path:
        ensure_dir(os.path.dirname(save_path))
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  [SAVED] {save_path}")

    plt.close(fig)
    return fig, axes


def plot_timing_summary_bar(timing_dict, title="Task Timing Summary",
                            save_path=None):
    fig, ax = plt.subplots(figsize=(8, 5))
    tasks = list(timing_dict.keys())
    times = list(timing_dict.values())

    bars = ax.bar(tasks, times, color=['#4e79a7', '#f28e2b', '#e15759', '#76b7b2'],
                  edgecolor='black', linewidth=0.5)

    for bar, t in zip(bars, times):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f'{t:.1f}s', ha='center', fontsize=11, fontweight='bold')

    ax.set_ylabel('Time (seconds)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, axis='y', alpha=0.3)

    if save_path:
        ensure_dir(os.path.dirname(save_path))
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  [SAVED] {save_path}")

    plt.close(fig)
    return fig, ax


def generate_all_plots(dataset_name, pred_poses, slam_poses,
                       pred_landmarks, slam_landmarks,
                       slam_stats, mapping_stats,
                       output_dir):
    ensure_dir(output_dir)
    print(f"\n{'='*60}")
    print(f"Generating plots for {dataset_name}")
    print(f"{'='*60}")

    plot_trajectory_2d(
        pred_poses,
        title=f"{dataset_name} — IMU-Only Trajectory (Task 1)",
        save_path=os.path.join(output_dir, "trajectory_imu_only.png")
    )
    plt.close('all')

    plot_trajectory_2d(
        slam_poses,
        title=f"{dataset_name} — SLAM Trajectory (Task 4)",
        save_path=os.path.join(output_dir, "trajectory_slam.png")
    )
    plt.close('all')

    plot_trajectory_comparison(
        pred_poses, slam_poses,
        title=f"{dataset_name} — IMU vs SLAM Trajectory",
        save_path=os.path.join(output_dir, "trajectory_comparison.png")
    )

    if pred_landmarks is not None:
        plot_landmarks_2d(
            pred_landmarks, poses=pred_poses,
            title=f"{dataset_name} — Landmark Map (Task 3, IMU poses)",
            save_path=os.path.join(output_dir, "landmarks_mapping.png")
        )

    if slam_landmarks is not None:
        plot_landmarks_2d(
            slam_landmarks, poses=pred_poses, slam_poses=slam_poses,
            title=f"{dataset_name} — Landmark Map (Task 4 SLAM)",
            save_path=os.path.join(output_dir, "landmarks_slam.png")
        )

    if slam_stats:
        if 'pose_cov_trace' in slam_stats and len(slam_stats['pose_cov_trace']) > 0:
            plot_covariance_trace(
                slam_stats['pose_cov_trace'],
                title=f"{dataset_name} — Pose Covariance Trace",
                save_path=os.path.join(output_dir, "covariance_trace.png")
            )

        if 'innovations' in slam_stats and len(slam_stats['innovations']) > 0:
            plot_innovation_stats(
                slam_stats['innovations'],
                title=f"{dataset_name} — Innovation Magnitude",
                save_path=os.path.join(output_dir, "innovation_stats.png")
            )

        if 'n_observed' in slam_stats:
            n_new = slam_stats.get('n_new', None)
            plot_landmark_count(
                slam_stats['n_observed'], n_new,
                title=f"{dataset_name} — Features per Step",
                save_path=os.path.join(output_dir, "feature_count.png")
            )

        if ('prediction_times' in slam_stats and
                'update_times' in slam_stats and
                len(slam_stats['prediction_times']) > 0):
            plot_timing_breakdown(
                slam_stats['prediction_times'],
                slam_stats['update_times'],
                title=f"{dataset_name} — Prediction vs Update Timing",
                save_path=os.path.join(output_dir, "timing_breakdown.png")
            )

    print(f"[✓] All plots saved to {output_dir}\n")
