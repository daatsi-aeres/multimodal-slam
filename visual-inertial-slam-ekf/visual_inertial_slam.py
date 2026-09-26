import numpy as np
import time as time_module
from scipy.linalg import expm
from geometry_utils import (
    axangle2twist, twist2pose, inversePose,
    projection, projectionJacobian
)
from slam_utils import (
    get_cam_T_imu, build_stereo_projection_matrix, compute_baseline,
    get_valid_features, subsample_features, stereo_triangulate,
    observation_jacobian_landmark, observation_jacobian_pose
)


def run_visual_inertial_slam(v_t, w_t, timestamps, features,
                              K_l, K_r, imu_T_camL, imu_T_camR,
                              W_noise=None, V_noise=None,
                              max_features_per_step=200,
                              mahal_thresh=30.0, innov_thresh=80.0,
                              init_landmark_cov=100.0,
                              min_disparity=15.0, max_disparity=150.0):
    T_steps = len(timestamps)
    N = features.shape[1]

    if W_noise is None:
        W_noise = np.diag([1e-3, 1e-3, 1e-3, 1e-4, 1e-4, 1e-4])
    if V_noise is None:
        V_noise = np.eye(4) * 200.0

    cam_T_imu_L = get_cam_T_imu(imu_T_camL)
    baseline = compute_baseline(imu_T_camL, imu_T_camR)
    Ks = build_stereo_projection_matrix(K_l, K_r, baseline)

    pose = np.eye(4)
    poses = np.zeros((T_steps, 4, 4))
    poses[0] = pose.copy()

    landmark_mean = np.full((N, 3), np.nan)
    landmark_init = np.zeros(N, dtype=bool)

    Sigma_pp = np.eye(6) * 1e-2
    Sigma_pm = np.zeros((6, N, 3))
    Sigma_mm = np.zeros((N, 3, 3))

    stats = {
        'prediction_times': [],
        'update_times': [],
        'n_observed': [],
        'n_new': [],
        'pose_cov_trace': [],
        'innovations': [],
    }

    rng = np.random.default_rng(42)

    for t in range(T_steps - 1):
        t_pred_start = time_module.time()

        dt = timestamps[t + 1] - timestamps[t]
        xi = np.concatenate([v_t[t], w_t[t]])

        xi_hat = axangle2twist((dt * xi).reshape(1, 6))
        exp_xi = twist2pose(xi_hat).squeeze(0)
        pose = pose @ exp_xi

        ad_xi = _adjoint_se3(xi)
        F_t = expm(-dt * ad_xi)

        Sigma_pp = F_t @ Sigma_pp @ F_t.T + dt * W_noise
        Sigma_pm = np.einsum('ij,jkl->ikl', F_t, Sigma_pm)

        t_pred = time_module.time() - t_pred_start
        stats['prediction_times'].append(t_pred)

        t_upd_start = time_module.time()

        valid_idx, z_valid = get_valid_features(features, t + 1,
                                                  min_disparity=min_disparity,
                                                  max_disparity=max_disparity)
        if len(valid_idx) == 0:
            poses[t + 1] = pose.copy()
            stats['update_times'].append(0)
            stats['n_observed'].append(0)
            stats['n_new'].append(0)
            stats['pose_cov_trace'].append(np.trace(Sigma_pp))
            continue

        valid_idx, z_valid = subsample_features(
            valid_idx, z_valid, max_features_per_step, rng
        )

        new_mask = ~landmark_init[valid_idx]
        n_new = 0
        if np.any(new_mask):
            new_lm_idx = valid_idx[new_mask]
            z_new = z_valid[new_mask]
            m_new = stereo_triangulate(
                z_new, K_l, K_r, cam_T_imu_L, pose, baseline
            )
            landmark_mean[new_lm_idx] = m_new
            Sigma_mm[new_lm_idx] = np.eye(3)[np.newaxis] * init_landmark_cov
            landmark_init[new_lm_idx] = True
            n_new = len(new_lm_idx)

        old_mask = landmark_init[valid_idx] & ~new_mask
        if np.any(old_mask):
            obs_idx = valid_idx[old_mask]
            z_obs = z_valid[old_mask]

            _do_slam_update(
                pose, landmark_mean, Sigma_pp, Sigma_pm, Sigma_mm,
                obs_idx, z_obs, cam_T_imu_L, Ks, V_noise, stats,
                mahal_thresh=mahal_thresh, innov_thresh=innov_thresh
            )

        poses[t + 1] = pose.copy()
        stats['update_times'].append(time_module.time() - t_upd_start)
        stats['n_observed'].append(len(valid_idx))
        stats['n_new'].append(n_new)
        stats['pose_cov_trace'].append(np.trace(Sigma_pp))

    return poses, landmark_mean, stats


def _do_slam_update(pose, landmark_mean, Sigma_pp, Sigma_pm, Sigma_mm,
                    obs_idx, z_obs, cam_T_imu_L, Ks, V_noise, stats,
                    mahal_thresh=30.0, innov_thresh=80.0):
    K_obs = len(obs_idx)
    m_obs = landmark_mean[obs_idx]

    imu_T_world = inversePose(pose.reshape(1, 4, 4)).squeeze(0)
    cam_T_world = cam_T_imu_L @ imu_T_world

    m_hom = np.hstack([m_obs, np.ones((K_obs, 1))])
    q = (cam_T_world @ m_hom.T).T

    valid = q[:, 2] > 0.1
    if not np.any(valid):
        return

    obs_idx = obs_idx[valid]
    z_obs = z_obs[valid]
    q = q[valid]
    m_obs = m_obs[valid]
    K_obs = len(obs_idx)

    q_proj = projection(q)
    z_pred = (Ks @ q_proj.T).T

    H_landmark = observation_jacobian_landmark(q, cam_T_world, Ks)
    H_pose = observation_jacobian_pose(q, cam_T_imu_L, Ks)

    innovations = []
    I6 = np.eye(6)
    I3 = np.eye(3)
    IKH_total = I6.copy()
    observed_pm_exact = {}

    for i in range(K_obs):
        j = obs_idx[i]
        H_p = H_pose[i]
        H_m = H_landmark[i]

        innov = z_obs[i] - z_pred[i]
        innovations.append(np.linalg.norm(innov))

        Sigma_pj = Sigma_pm[:, j, :]
        Sigma_jj = Sigma_mm[j]

        S = (H_p @ Sigma_pp @ H_p.T +
             H_p @ Sigma_pj @ H_m.T +
             H_m @ Sigma_pj.T @ H_p.T +
             H_m @ Sigma_jj @ H_m.T +
             V_noise)

        if np.isnan(S).any() or np.isinf(S).any():
            continue

        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            try:
                S_inv = np.linalg.inv(S + np.eye(4) * 1e-3)
            except np.linalg.LinAlgError:
                continue

        m_dist = innov @ S_inv @ innov
        if m_dist > mahal_thresh or m_dist < 0.0 or np.linalg.norm(innov) > innov_thresh:
            continue

        K_pose = (Sigma_pp @ H_p.T + Sigma_pj @ H_m.T) @ S_inv
        K_lm = (Sigma_pj.T @ H_p.T + Sigma_jj @ H_m.T) @ S_inv

        delta_pose = K_pose @ innov
        delta_lm = K_lm @ innov

        delta_hat = axangle2twist(delta_pose.reshape(1, 6))
        pose[:] = pose @ twist2pose(delta_hat).squeeze(0)

        landmark_mean[j] += delta_lm

        IKH_pp = I6 - K_pose @ H_p

        Sigma_pp[:] = (IKH_pp @ Sigma_pp @ IKH_pp.T +
                       K_pose @ V_noise @ K_pose.T)
        Sigma_pp[:] = (Sigma_pp + Sigma_pp.T) / 2.0

        IKH_mm = I3 - K_lm @ H_m
        Sigma_mm[j] = (IKH_mm @ Sigma_jj @ IKH_mm.T +
                       K_lm @ V_noise @ K_lm.T)

        IKH_total = IKH_pp @ IKH_total

        exact_pj = IKH_pp @ Sigma_pj - K_pose @ H_m @ Sigma_jj
        Sigma_pm[:, j, :] = exact_pj
        observed_pm_exact[j] = exact_pj

    if not np.allclose(IKH_total, I6):
        Sigma_pm[:] = np.einsum('ij,jkl->ikl', IKH_total, Sigma_pm)
        for j, exact_pj in observed_pm_exact.items():
            Sigma_pm[:, j, :] = exact_pj

    eigvals, eigvecs = np.linalg.eigh(Sigma_pp)
    min_eig = 1e-6
    if np.any(eigvals < min_eig):
        eigvals = np.maximum(eigvals, min_eig)
        Sigma_pp[:] = eigvecs @ np.diag(eigvals) @ eigvecs.T
        Sigma_pp[:] = (Sigma_pp + Sigma_pp.T) / 2.0

    if innovations:
        stats['innovations'].append(np.mean(innovations))


def _adjoint_se3(xi):
    v, w = xi[:3], xi[3:]
    w_hat = np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])
    v_hat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    ad = np.zeros((6, 6))
    ad[:3, :3] = w_hat
    ad[:3, 3:] = v_hat
    ad[3:, 3:] = w_hat
    return ad
