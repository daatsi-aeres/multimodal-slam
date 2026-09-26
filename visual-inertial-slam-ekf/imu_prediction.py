import numpy as np
from scipy.linalg import expm
from geometry_utils import axangle2twist, twist2pose


def predict_imu_poses(v_t, w_t, timestamps, W_noise=None):
    T = len(timestamps)

    if W_noise is None:
        W_noise = np.diag([1e-3, 1e-3, 1e-3, 1e-4, 1e-4, 1e-4])

    poses = np.zeros((T, 4, 4))
    poses[0] = np.eye(4)
    covariances = np.zeros((T, 6, 6))
    covariances[0] = np.eye(6) * 1e-2

    for t in range(T - 1):
        dt = timestamps[t + 1] - timestamps[t]
        xi = np.concatenate([v_t[t], w_t[t]])

        xi_hat = axangle2twist((dt * xi).reshape(1, 6))
        exp_xi = twist2pose(xi_hat).squeeze(0)
        poses[t + 1] = poses[t] @ exp_xi

        ad_xi = _adjoint_se3(xi)
        F_t = expm(-dt * ad_xi)
        covariances[t + 1] = F_t @ covariances[t] @ F_t.T + dt * W_noise

    return poses, covariances


def _adjoint_se3(xi):
    v = xi[:3]
    w = xi[3:]
    w_hat = _skew(w)
    v_hat = _skew(v)

    ad = np.zeros((6, 6))
    ad[:3, :3] = w_hat
    ad[:3, 3:] = v_hat
    ad[3:, 3:] = w_hat
    return ad


def _skew(v):
    return np.array([
        [0,    -v[2],  v[1]],
        [v[2],  0,    -v[0]],
        [-v[1], v[0],  0   ]
    ])
