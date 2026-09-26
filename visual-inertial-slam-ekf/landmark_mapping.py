import numpy as np
import time
from geometry_utils import inversePose, projection, projectionJacobian
from slam_utils import (
    get_cam_T_imu, build_stereo_projection_matrix, compute_baseline,
    project_landmarks_to_stereo, observation_jacobian_landmark,
    get_valid_features, subsample_features, stereo_triangulate
)


def map_landmarks(poses, features, K_l, K_r, imu_T_camL, imu_T_camR,
                  V_noise=None, max_features_per_step=200,
                  min_disparity=15.0, max_disparity=150.0):
    T = poses.shape[0]
    N = features.shape[1]

    if V_noise is None:
        V_noise = np.eye(4) * 10.0

    cam_T_imu_L = get_cam_T_imu(imu_T_camL)
    baseline = compute_baseline(imu_T_camL, imu_T_camR)
    Ks = build_stereo_projection_matrix(K_l, K_r, baseline)

    landmark_mean = np.full((N, 3), np.nan)
    landmark_cov = np.zeros((N, 3, 3))
    landmark_initialized = np.zeros(N, dtype=bool)

    stats = {
        'n_observed_per_step': [],
        'n_new_per_step': [],
        'update_times': [],
    }

    rng = np.random.default_rng(42)

    for t in range(T):
        t_start = time.time()

        valid_idx, z_valid = get_valid_features(features, t,
                                                  min_disparity=min_disparity,
                                                  max_disparity=max_disparity)
        if len(valid_idx) == 0:
            stats['n_observed_per_step'].append(0)
            stats['n_new_per_step'].append(0)
            stats['update_times'].append(0)
            continue

        valid_idx, z_valid = subsample_features(
            valid_idx, z_valid, max_features_per_step, rng
        )

        n_new = 0

        new_mask = ~landmark_initialized[valid_idx]
        if np.any(new_mask):
            new_idx = valid_idx[new_mask]
            z_new = z_valid[new_mask]
            m_new = stereo_triangulate(
                z_new, K_l, K_r, cam_T_imu_L, poses[t], baseline
            )
            landmark_mean[new_idx] = m_new
            landmark_cov[new_idx] = np.eye(3)[np.newaxis, :, :] * 100.0
            landmark_initialized[new_idx] = True
            n_new = len(new_idx)

        old_mask = landmark_initialized[valid_idx] & ~new_mask
        if np.any(old_mask):
            old_idx = valid_idx[old_mask]
            z_obs = z_valid[old_mask]
            m_old = landmark_mean[old_idx]

            imu_T_world = inversePose(poses[t].reshape(1, 4, 4)).squeeze(0)
            cam_T_world = cam_T_imu_L @ imu_T_world

            m_hom = np.hstack([m_old, np.ones((len(m_old), 1))])
            q = (cam_T_world @ m_hom.T).T

            valid_depth = q[:, 2] > 0.1
            if not np.any(valid_depth):
                stats['n_observed_per_step'].append(len(valid_idx))
                stats['n_new_per_step'].append(n_new)
                stats['update_times'].append(time.time() - t_start)
                continue

            old_idx = old_idx[valid_depth]
            z_obs = z_obs[valid_depth]
            q = q[valid_depth]
            m_old = m_old[valid_depth]

            q_proj = projection(q)
            z_pred = (Ks @ q_proj.T).T

            H = observation_jacobian_landmark(q, cam_T_world, Ks)

            for i in range(len(old_idx)):
                j = old_idx[i]
                H_j = H[i]
                Sigma_j = landmark_cov[j]

                innov = z_obs[i] - z_pred[i]

                S = H_j @ Sigma_j @ H_j.T + V_noise
                K_gain = Sigma_j @ H_j.T @ np.linalg.inv(S)

                landmark_mean[j] += K_gain @ innov

                I_KH = np.eye(3) - K_gain @ H_j
                landmark_cov[j] = I_KH @ Sigma_j

        stats['n_observed_per_step'].append(len(valid_idx))
        stats['n_new_per_step'].append(n_new)
        stats['update_times'].append(time.time() - t_start)

    return landmark_mean, landmark_cov, stats
