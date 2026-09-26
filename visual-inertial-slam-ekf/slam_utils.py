import numpy as np
from geometry_utils import inversePose, projection, projectionJacobian


def build_stereo_projection_matrix(K_l, K_r, baseline):
    fsu_l, fsu_r = K_l[0, 0], K_r[0, 0]
    fsv_l, fsv_r = K_l[1, 1], K_r[1, 1]
    cu_l, cu_r   = K_l[0, 2], K_r[0, 2]
    cv_l, cv_r   = K_l[1, 2], K_r[1, 2]

    Ks = np.array([
        [fsu_l,  0,     cu_l,  0              ],
        [0,      fsv_l, cv_l,  0              ],
        [fsu_r,  0,     cu_r, -fsu_r * baseline],
        [0,      fsv_r, cv_r,  0              ],
    ], dtype=np.float64)
    return Ks


def compute_baseline(imu_T_camL, imu_T_camR):
    origin_L = imu_T_camL[:3, 3]
    origin_R = imu_T_camR[:3, 3]
    return np.linalg.norm(origin_L - origin_R)


def get_cam_T_imu(imu_T_cam):
    cam_T_imu = inversePose(imu_T_cam.reshape(1, 4, 4)).squeeze(0)

    oTr = np.array([
        [ 0, -1,  0,  0],
        [ 0,  0, -1,  0],
        [ 1,  0,  0,  0],
        [ 0,  0,  0,  1]
    ], dtype=np.float64)

    return oTr @ cam_T_imu


def project_landmarks_to_stereo(landmarks_world, T_imu, cam_T_imu, Ks):
    M = landmarks_world.shape[0]

    imu_T_world = inversePose(T_imu.reshape(1, 4, 4)).squeeze(0)
    m_hom = np.hstack([landmarks_world, np.ones((M, 1))])
    cam_T_world = cam_T_imu @ imu_T_world
    q = (cam_T_world @ m_hom.T).T

    q_proj = projection(q)
    z_pred = (Ks @ q_proj.T).T

    return z_pred, q


def observation_jacobian_landmark(q, cam_T_world, Ks):
    dpi = projectionJacobian(q)
    P = cam_T_world[:, :3]
    dpi_P = dpi @ P
    H = np.einsum('ij,mjk->mik', Ks, dpi_P)
    return H


def observation_jacobian_pose(q, cam_T_imu, Ks):
    M = q.shape[0]

    dpi = projectionJacobian(q)

    imu_T_cam = inversePose(cam_T_imu.reshape(1, 4, 4)).squeeze(0)
    p_imu = (imu_T_cam @ q.T).T

    odot = np.zeros((M, 4, 6))
    odot[:, :3, :3] = np.eye(3)
    odot[:, 0, 3] =  0
    odot[:, 0, 4] =  p_imu[:, 2]
    odot[:, 0, 5] = -p_imu[:, 1]
    odot[:, 1, 3] = -p_imu[:, 2]
    odot[:, 1, 4] =  0
    odot[:, 1, 5] =  p_imu[:, 0]
    odot[:, 2, 3] =  p_imu[:, 1]
    odot[:, 2, 4] = -p_imu[:, 0]
    odot[:, 2, 5] =  0

    neg_odot = -odot
    cam_odot = np.einsum('ij,mjk->mik', cam_T_imu, neg_odot)
    dpi_cam_odot = np.einsum('mij,mjk->mik', dpi, cam_odot)
    H_pose = np.einsum('ij,mjk->mik', Ks, dpi_cam_odot)

    return H_pose


def get_valid_features(features, t, min_disparity=15.0, max_disparity=150.0):
    z_t = features[:, :, t]
    valid_mask = np.all(z_t != -1, axis=0)

    if min_disparity > 0 or max_disparity < np.inf:
        disparity = z_t[0, :] - z_t[2, :]
        valid_mask &= (disparity >= min_disparity) & (disparity <= max_disparity)

    valid_idx = np.where(valid_mask)[0]
    z_valid = z_t[:, valid_idx].T
    return valid_idx, z_valid


def subsample_features(valid_idx, z_valid, max_features=200, rng=None):
    K = len(valid_idx)
    if K <= max_features:
        return valid_idx, z_valid

    if rng is None:
        rng = np.random.default_rng(42)
    chosen = rng.choice(K, max_features, replace=False)
    chosen.sort()
    return valid_idx[chosen], z_valid[chosen]


def stereo_triangulate(z, K_l, K_r, cam_T_imu_L, T_imu, baseline):
    fsu = K_l[0, 0]
    cu_l = K_l[0, 2]
    cv_l = K_l[1, 2]
    fsv = K_l[1, 1]

    uL, vL, uR, vR = z[:, 0], z[:, 1], z[:, 2], z[:, 3]

    disparity = uL - uR
    disparity = np.maximum(disparity, 0.1)

    depth = fsu * baseline / disparity

    x_cam = (uL - cu_l) * depth / fsu
    y_cam = (vL - cv_l) * depth / fsv
    z_cam = depth

    p_cam = np.stack([x_cam, y_cam, z_cam, np.ones_like(x_cam)], axis=1)

    imu_T_cam = inversePose(cam_T_imu_L.reshape(1, 4, 4)).squeeze(0)
    p_imu = (imu_T_cam @ p_cam.T).T
    p_world = (T_imu @ p_imu.T).T

    return p_world[:, :3]


def hat(v):
    if v.ndim == 1:
        return np.array([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0]
        ])
    S = np.zeros((*v.shape[:-1], 3, 3))
    S[..., 0, 1] = -v[..., 2]
    S[..., 0, 2] =  v[..., 1]
    S[..., 1, 0] =  v[..., 2]
    S[..., 1, 2] = -v[..., 0]
    S[..., 2, 0] = -v[..., 1]
    S[..., 2, 1] =  v[..., 0]
    return S


def curly_hat(xi):
    v = xi[:3]
    w = xi[3:]
    Xi = np.zeros((4, 4))
    Xi[:3, :3] = hat(w)
    Xi[:3, 3] = v
    return Xi
