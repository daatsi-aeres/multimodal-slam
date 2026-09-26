import numpy as np
from scipy.spatial import KDTree


def find_nearest_neighbors(source, target):
    tree = KDTree(target)
    distances, indices = tree.query(source)
    return distances, indices


def compute_optimal_transform(source_pts, target_pts):
    centroid_source = np.mean(source_pts, axis=0)
    centroid_target = np.mean(target_pts, axis=0)
    source_centered = source_pts - centroid_source
    target_centered = target_pts - centroid_target
    H = source_centered.T @ target_centered
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    t = centroid_target - R @ centroid_source
    return R, t


def build_transform(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def icp_3d(source, target, init_pose=np.eye(4), max_iter=50, tolerance=1e-4):
    current_source = (init_pose[:3, :3] @ source.T).T + init_pose[:3, 3]
    total_transform = init_pose.copy()
    prev_mse = float('inf')
    for i in range(max_iter):
        distances, indices = find_nearest_neighbors(current_source, target)
        R, t = compute_optimal_transform(current_source, target[indices])
        current_transform = build_transform(R, t)
        total_transform = current_transform @ total_transform
        current_source = (current_transform[:3, :3] @ current_source.T).T + current_transform[:3, 3]
        mse = np.mean(distances**2)
        if abs(prev_mse - mse) < tolerance:
            break
        print(f"    iter {i:3d}: MSE={mse:.6f}  Δ={abs(prev_mse-mse):.2e}")
        prev_mse = mse
    return total_transform, mse


def icp_with_yaw_init(source, target, num_angles=36):
    best_pose = None
    best_mse = float('inf')

    for k, angle in enumerate(np.linspace(0, 2*np.pi, num_angles, endpoint=False)):
        print(f"  [{k+1}/{num_angles}] yaw={np.degrees(angle):6.1f}°", end='', flush=True)
        R_z = np.array([[np.cos(angle), -np.sin(angle), 0],
                        [np.sin(angle),  np.cos(angle), 0],
                        [0,              0,             1]])
        init_pose = build_transform(R_z, np.zeros(3))
        pose, mse = icp_3d(source, target, init_pose=init_pose)
        print(f"  → MSE={mse:.6f}  {'✓ best' if mse < best_mse else ''}")
        if mse < best_mse:
            best_mse = mse
            best_pose = pose
    return best_pose, best_mse
