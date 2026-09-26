import numpy as np
import matplotlib.pyplot as plt
from transforms3d.quaternions import qmult, axangle2quat
from transforms3d.euler import quat2euler 
from transforms3d.quaternions import mat2quat
import pickle
import torch
import os

# ================= CONFIG =================
BASE_PATH = "data"
RESULTS_DIR = "results"
VISUALIZE = False        # Turn ON for debugging, OFF for batch runs
USE_VICON_FOR_PANO = False
STEP_PANO = 50
NUM_ITERS = 1000
LR = 0.01
use_integrated_init = True 
# ========================================

def make_result_dir(set_name, num):
    result_dir = os.path.join(RESULTS_DIR, f"{set_name}_{num}")
    os.makedirs(result_dir, exist_ok=True)
    return result_dir

def save_imu_vicon_plot(result_dir, imu_ts, rpy_array, vicon_ts, vicon_rpy):
    labels = ['Roll', 'Pitch', 'Yaw']
    rpy_array_plot = np.unwrap(rpy_array, axis=0)
    vicon_rpy_plot = np.unwrap(vicon_rpy, axis=0)
    plt.figure(figsize=(14, 8))
    for i in range(3):
        plt.subplot(3, 1, i+1)
        plt.plot(imu_ts, rpy_array_plot[:, i], label='IMU')
        plt.plot(vicon_ts, vicon_rpy_plot[:, i], label='VICON')
        plt.ylabel(labels[i])
        plt.legend()
        plt.grid(True)
    plt.xlabel("Time (s)")
    plt.suptitle("IMU vs VICON Orientation")
    plt.tight_layout()
    plt.savefig(f"{result_dir}/imu_vs_vicon.png", dpi=300)
    plt.close()

def save_acc_plot(result_dir, imu_ts, calibrated_acc, acc_pred):
    labels = ['Acc X', 'Acc Y', 'Acc Z']
    plt.figure(figsize=(12, 8))
    for k in range(3):
        plt.subplot(3, 1, k+1)
        plt.plot(imu_ts, calibrated_acc[k], label='Measured')
        plt.plot(imu_ts, acc_pred[:, k], '--', label='Predicted Gravity')
        plt.ylabel(labels[k])
        plt.legend()
        plt.grid(True)
    plt.xlabel("Time (s)")
    plt.suptitle("Accelerometer vs Gravity Prediction")
    plt.tight_layout()
    plt.savefig(f"{result_dir}/acc_vs_gravity.png", dpi=300)
    plt.close()

def save_loss_plot(result_dir, loss_log):
    plt.figure()
    plt.plot(loss_log)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Optimization Loss")
    plt.grid(True)
    plt.savefig(f"{result_dir}/optimization_loss.png", dpi=300)
    plt.close()

def save_opt_plot(result_dir, imu_ts, rpy_opt, vicon_ts, vicon_rpy):
    labels = ['Roll', 'Pitch', 'Yaw']
    rpy_opt_plot = np.unwrap(rpy_opt, axis=0)
    vicon_rpy_plot = np.unwrap(vicon_rpy, axis=0)
    plt.figure(figsize=(14, 8))
    for i in range(3):
        plt.subplot(3,1,i+1)
        plt.plot(imu_ts, rpy_opt_plot[:,i], label="Optimized")
        plt.plot(vicon_ts, vicon_rpy_plot[:,i], '--', label="VICON")
        plt.legend(); plt.grid()
        plt.ylabel(labels[i])
    plt.xlabel("Time")
    plt.tight_layout()
    plt.savefig(f"{result_dir}/optimized_vs_vicon.png", dpi=300)
    plt.close()

def save_combined_plot(result_dir, imu_ts, rpy_imu, rpy_opt, vicon_ts, vicon_rpy):
    labels = ['Roll', 'Pitch', 'Yaw']
    rpy_imu_plot = np.unwrap(rpy_imu, axis=0)
    rpy_opt_plot = np.unwrap(rpy_opt, axis=0)
    vicon_rpy_plot = np.unwrap(vicon_rpy, axis=0)
    plt.figure(figsize=(14, 10))
    for i in range(3):
        plt.subplot(3, 1, i+1)
        plt.plot(vicon_ts, vicon_rpy_plot[:, i], color='black', linestyle='--', linewidth=2, label="VICON (Ground Truth)", alpha=0.8)
        plt.plot(imu_ts, rpy_imu_plot[:, i], color='blue', linestyle='-', linewidth=1, label="Raw IMU Integration", alpha=0.6)
        plt.plot(imu_ts, rpy_opt_plot[:, i], color='red', linestyle='-', linewidth=1.5, label="Optimized (PGD)")
        plt.ylabel(f"{labels[i]} (rad)", fontsize=12)
        plt.grid(True, which='both', linestyle='--', alpha=0.5)
        if i == 0:
            plt.legend(loc="upper right", fontsize=10)
            plt.title("Orientation Tracking: Raw vs. Optimized vs. Ground Truth", fontsize=14)
    plt.xlabel("Time (s)", fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{result_dir}/combined_orientation_plot.png", dpi=300)
    plt.close()

def run_dataset(set, num):
    print(f"\n===== Running dataset {set} {num} =====")
    result_dir = make_result_dir(set, num)
    imu_file = f"{BASE_PATH}/{set}/imu/imuRaw{num}.p"
    vicon_file = f"{BASE_PATH}/{set}/vicon/viconRot{num}.p"
    cam_path  = f"{BASE_PATH}/{set}/cam/cam{num}.p"
    visualize_plots = VISUALIZE

    with open(imu_file.format(set=set, num=num), 'rb') as f:
        imu_arr = pickle.load(f)
    with open(vicon_file.format(set=set, num=num), 'rb') as f:
        vicon_dict = pickle.load(f, encoding='latin1')

    imu_ts = imu_arr[0, :]
    gyro_sens = 3.33 * 180.0 / np.pi
    acc_sens = 330 
    gyro_raw = imu_arr[4:7, :]
    acc_raw = imu_arr[1:4, :]
    gyro_bias = np.mean(gyro_raw[:, :100], axis=1)

    Vref = 3300.0
    adc_max = 1023.0
    gyro_scale_factor= (Vref / (adc_max * gyro_sens)) 
    acc_scale_factor= (Vref / (adc_max * acc_sens)) 
    calibrated_omega = (gyro_raw - gyro_bias[:, None]) * gyro_scale_factor
    acc_raw_physical = acc_raw * acc_scale_factor
    acc_initial_mean = np.mean(acc_raw_physical[:, :100], axis=1)
    acc_bias = acc_initial_mean - np.array([0, 0, 1]) 
    calibrated_acc = acc_raw_physical - acc_bias[:, None]

    N = imu_ts.shape[0]
    q = np.array([1.0, 0.0, 0.0, 0.0])
    q_calculated = [q.copy()]

    def exp_map(theta_vec):
        angle = np.linalg.norm(theta_vec)
        if angle < 1e-8:
            return np.array([1.0, 0.0, 0.0, 0.0])
        else:
            half_angle = angle / 2
            unit_axis = theta_vec / angle
            return np.array([np.cos(half_angle), unit_axis[0]*np.sin(half_angle), unit_axis[1]*np.sin(half_angle), unit_axis[2]*np.sin(half_angle)])

    def q_inv(q):
        return np.array([q[0], -q[1], -q[2], -q[3]])

    def quaternion_integration(calibrated_omega, imu_ts):
        q = np.array([1.0, 0.0, 0.0, 0.0])
        q_calculated = [q.copy()]
        prev_ts = imu_ts[0]
        for t in range(1, N):
            dt = imu_ts[t] - prev_ts
            prev_ts = imu_ts[t]
            theta_vec = calibrated_omega[:, t-1] * dt 
            dq = exp_map(theta_vec)
            q = qmult(q, dq)
            q = q / np.linalg.norm(q)  
            q_calculated.append(q.copy())
        return q_calculated

    q_calculated = quaternion_integration(calibrated_omega, imu_ts)
    rpy_list = [quat2euler(qi, axes='sxyz') for qi in q_calculated]
    rpy_array = np.array(rpy_list)
    vicon_rots = vicon_dict['rots']
    vicon_ts = vicon_dict['ts'].flatten()

    vicon_rpy = []
    vicon_q_list = []
    last_valid_rpy = np.zeros(3)
    last_valid_q = np.array([1.0, 0.0, 0.0, 0.0])

    for i in range(vicon_rots.shape[2]):
        R = vicon_rots[:, :, i]
        if not np.isfinite(R).all():
            vicon_rpy.append(last_valid_rpy)
            vicon_q_list.append(last_valid_q)
            continue
        U, _, Vt = np.linalg.svd(R)
        R_fixed = U @ Vt
        if np.linalg.det(R_fixed) < 0:
            U[:, -1] *= -1
            R_fixed = U @ Vt
        q_vicon = mat2quat(R_fixed)
        rpy = quat2euler(q_vicon, axes='sxyz')
        vicon_rpy.append(rpy)
        vicon_q_list.append(q_vicon)
        last_valid_rpy = rpy
        last_valid_q = q_vicon

    vicon_rpy = np.array(vicon_rpy)
    q_vicon = np.array(vicon_q_list)

    if visualize_plots:
        plt.figure(figsize=(14, 8))
        for i in range(3):
            plt.subplot(3, 1, i+1)
            plt.plot(imu_ts, rpy_array[:, i], label='IMU', linewidth=1)
            plt.plot(vicon_ts, vicon_rpy[:, i], label='VICON', linewidth=2)
            plt.ylabel(labels[i] + ' (rad)')
            plt.legend()
            plt.grid(True)
        plt.xlabel('Time (s)')
        plt.suptitle('IMU Orientation vs VICON Ground Truth')
        plt.tight_layout()
        plt.show()    
        plt.close('all')

    acc_pred = []
    for q in q_calculated:
        v_world = np.array([0, 0, 0, 1])
        v_body = qmult(qmult(q_inv(q), v_world), q)
        acc_pred.append(v_body[1:])
    acc_pred = np.array(acc_pred)

    if visualize_plots:
        plt.figure(figsize=(12, 8))
        labels = ['Acc X', 'Acc Y', 'Acc Z']
        for k in range(3):
            plt.subplot(3, 1, k+1)
            plt.plot(imu_ts, calibrated_acc[k], label='IMU Measured', alpha=0.7)
            plt.plot(imu_ts, acc_pred[:, k], label='Predicted Gravity', linestyle='--', linewidth=2)
            plt.ylabel(f'{labels[k]} (m/s^2)')
            plt.legend()
            plt.grid(True)
        plt.xlabel('Time (s)')
        plt.suptitle('Measured Acceleration vs Predicted Gravity from Orientation')
        plt.tight_layout()
        plt.show()
        plt.close('all')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    dt_np = np.diff(imu_ts)
    omega_np = calibrated_omega.T[:len(dt_np)]
    omega_t = torch.tensor(omega_np, dtype=torch.float32, device=device)
    dt_t = torch.tensor(dt_np, dtype=torch.float32, device=device)
    acc_t = torch.tensor(calibrated_acc.T, dtype=torch.float32, device=device)
    T = acc_t.shape[0]
    q_init_tensor = torch.tensor(np.array(q_calculated), dtype=torch.float32, device=device)

    if use_integrated_init:
        q_opt = q_init_tensor.clone().detach().requires_grad_(True)
    else:
        q_simple = torch.zeros_like(q_init_tensor)
        q_simple[:, 0] = 1.0
        q_simple[0] = q_init_tensor[0] 
        q_opt = q_simple.clone().detach().requires_grad_(True)

    def qmul_t(q, r):
        w1, x1, y1, z1 = q.unbind(-1)
        w2, x2, y2, z2 = r.unbind(-1)
        return torch.stack([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ], dim=-1)

    def qinv_t(q):
        return torch.cat([q[..., :1], -q[..., 1:]], dim=-1)

    def qexp_t(v):
        alpha = torch.norm(v, dim=-1, keepdim=True)
        mask = (alpha < 1e-7).to(v.dtype)
        safe_alpha = alpha + 1e-12 
        factor = (1.0 - mask) * (torch.sin(alpha) / safe_alpha) + (mask * 1.0)
        q_w = torch.cos(alpha)
        q_v = factor * v
        return torch.cat([q_w, q_v], dim=-1)

    def qlog_t(q):
        w = q[:, 0:1]
        v = q[:, 1:]
        norm_v = torch.norm(v, dim=-1, keepdim=True)
        w_clamped = torch.clamp(w, -1.0 + 1e-7, 1.0 - 1e-7)
        theta = torch.acos(w_clamped)
        mask = (norm_v < 1e-7).to(v.dtype)
        safe_norm = norm_v + 1e-12
        factor = (1.0 - mask) * (2 * theta / safe_norm) + (mask * 2.0)
        return factor * v

    def cost_fn(q):
        delta = 0.5 * dt_t.unsqueeze(1) * omega_t
        dq = qexp_t(delta)
        q_pred = qmul_t(q[:-1], dq)
        q_err = qmul_t(qinv_t(q[1:]), q_pred)
        motion_err_vec = qlog_t(q_err) 
        motion_loss = 0.5 * torch.sum(motion_err_vec**2)
        g = torch.zeros((q.shape[0], 4), device=device)
        g[:, 3] = 1.0 
        g_hat = qmul_t(qinv_t(q), qmul_t(g, q))[:, 1:] 
        obs_loss = 0.5 * torch.sum((acc_t - g_hat)**2)
        return motion_loss + obs_loss

    def run_hyperparameter_sweep(alphas=[0.1, 0.05, 0.01, 0.005, 0.001]):
            print("\n--- Starting Hyperparameter Sweep ---")
            plt.figure(figsize=(10, 6))
            colors = plt.cm.viridis(np.linspace(0, 1, len(alphas)))
            for i, lr in enumerate(alphas):
                q_sweep = q_init_tensor.clone().detach().requires_grad_(True)
                sweep_losses = []
                for _ in range(1000):
                    loss = cost_fn(q_sweep)
                    loss.backward()
                    sweep_losses.append(loss.item())
                    with torch.no_grad():
                        q_sweep -= lr * q_sweep.grad
                        q_sweep /= torch.norm(q_sweep, dim=1, keepdim=True)
                        q_sweep.grad = None
                plt.plot(sweep_losses, label=f'$\\alpha={lr}$', color=colors[i], linewidth=1.5)
                print(f"LR {lr}: Final Loss = {sweep_losses[-1]:.4f}")
            plt.xlabel("Iteration")
            plt.ylabel("Loss (Log Scale)")
            plt.yscale('log')
            plt.title("Convergence Analysis for Different Learning Rates")
            plt.legend()
            plt.grid(True, which="both", linestyle='--', alpha=0.3)
            plt.tight_layout()
            plt.savefig(f"{result_dir}/hyperparam_sweep.png", dpi=300)
            plt.close()
            print("Sweep plot saved.\n")

    # run_hyperparameter_sweep()

    lr = LR
    num_iters = NUM_ITERS
    loss_log = []
    for it in range(num_iters):
        loss = cost_fn(q_opt)
        loss.backward()
        loss_log.append(loss.item())
        with torch.no_grad():
            q_opt -= lr * q_opt.grad
            q_opt /= torch.norm(q_opt, dim=1, keepdim=True)
            q_opt.grad = None
        if it % 10 == 0:
            print(f"Iter {it}, cost = {loss.item():.4f}")

    q_opt_np = q_opt.detach().cpu().numpy().astype(np.float64)
    rpy_opt_list = []
    for i in range(q_opt_np.shape[0]):
        qi = q_opt_np[i]
        rpy = quat2euler(qi, axes='sxyz')
        rpy_opt_list.append(rpy)
    rpy_opt = np.array(rpy_opt_list)

    if visualize_plots:
        labels = ['Roll', 'Pitch', 'Yaw']
        colors = ['r', 'g', 'b']
        plt.figure(figsize=(12, 10))
        for i in range(3):
            plt.subplot(3, 1, i+1)
            plt.plot(vicon_ts, vicon_rpy[:, i], label='Ground Truth (Vicon)', color='black', linestyle='--', linewidth=1.5, alpha=0.8)
            plt.plot(imu_ts, rpy_opt[:, i], label='Estimated (PGD)', color=colors[i], linewidth=1.5)
            plt.ylabel(f'{labels[i]} (rad)', fontsize=12)
            plt.legend(loc='upper right')
            plt.grid(True, which='both', linestyle='--', alpha=0.5)
            if i == 0:
                plt.title('Orientation Tracking Results: Optimized vs Ground Truth', fontsize=14)
        plt.xlabel('Time (s)', fontsize=12)
        plt.tight_layout()
        plt.savefig('orientation_results.png', dpi=300)
        plt.show()
        plt.close('all')

    # -------- Vectorized Panorama Stitching --------
    cam_file = cam_path.format(set=set, num=num)
    has_cam = num in [1, 2, 8, 9]

    if not has_cam:
        print("No camera for this dataset")
    else:
        with open(cam_path.format(set=set, num=num), "rb") as f:
            cam_data = pickle.load(f, encoding="latin1")
        frames = cam_data["cam"]
        cam_ts = cam_data["ts"].flatten()
        H, W, _, N_frames = frames.shape
        pano_H, pano_W = 800, 1600
        panorama = np.zeros((pano_H, pano_W, 3), dtype=np.uint32)
        pano_count = np.zeros((pano_H, pano_W), dtype=np.int32)
        fx = fy = 250.0
        cx = W / 2
        cy = H / 2
        R_cb = np.array([
            [0, 0, 1],
            [-1, 0, 0],
            [0, -1, 0]
        ])
        u_grid, v_grid = np.meshgrid(np.arange(W), np.arange(H))
        x_cam = (u_grid - cx) / fx
        y_cam = (v_grid - cy) / fy
        z_cam = np.ones_like(x_cam)
        rays_cam = np.stack([x_cam, y_cam, z_cam], axis=-1).reshape(-1, 3)
        rays_cam /= np.linalg.norm(rays_cam, axis=-1, keepdims=True)
        rays_imu = (R_cb @ rays_cam.T).T
        rays_flat = rays_imu.reshape(-1, 3)
        rays_flat /= np.linalg.norm(rays_flat, axis=-1, keepdims=True)
        
        def rotate_vectors(q, v):
            w, x, y, z = q
            t = 2 * np.cross([x, y, z], v)
            return v + w * t + np.cross([x, y, z], t)
        
        step = STEP_PANO
        use_vicon = USE_VICON_FOR_PANO
        for k in range(0, N_frames, step):
            t = cam_ts[k]
            if use_vicon:
                idx = np.searchsorted(vicon_ts, t, side='right') - 1
                idx = np.clip(idx, 0, len(vicon_ts) - 1)
                q_curr = q_vicon[idx]
            else:
                idx = np.searchsorted(imu_ts, t, side='right') - 1
                idx = np.clip(idx, 0, len(imu_ts) - 1)
                q_curr = q_opt_np[idx]
            img = frames[:, :, :, k]
            img_flat = img.reshape(-1, 3)
            rays_world = rotate_vectors(q_curr, rays_flat)
            xw, yw, zw = rays_world[:, 0], rays_world[:, 1], rays_world[:, 2]
            lon = np.arctan2(xw, yw)
            lat = np.arcsin(np.clip(zw, -1.0, 1.0))
            u_p = ((lon + np.pi) / (2*np.pi) * pano_W).astype(int)
            v_p = ((np.pi/2 - lat) / np.pi * pano_H).astype(int)
            u_p = np.clip(u_p, 0, pano_W-1)
            v_p = np.clip(v_p, 0, pano_H-1)
            panorama[v_p, u_p] += img_flat
            pano_count[v_p, u_p] += 1

        mask = pano_count > 0
        panorama[mask] = (panorama[mask] / pano_count[mask][:, None])
        panorama_final = panorama.astype(np.uint8)
        
        if(visualize_plots):
            plt.figure(figsize=(12, 6))
            plt.imshow(panorama_final)
            plt.title("Vectorized Panorama (Orientation-based Stitching)")
            plt.axis("off")
            plt.show()
            plt.close()

    # save_imu_vicon_plot(result_dir, imu_ts, rpy_array, vicon_ts, vicon_rpy)
    # save_acc_plot(result_dir, imu_ts, calibrated_acc, acc_pred)
    # save_loss_plot(result_dir, loss_log)
    # save_opt_plot(result_dir, imu_ts, rpy_opt, vicon_ts, vicon_rpy)
    save_combined_plot(result_dir, imu_ts, rpy_array, rpy_opt, vicon_ts, vicon_rpy)

    if has_cam and "panorama" in locals():
        plt.imsave(f"{result_dir}/panorama.png", panorama_final)

    # np.save(f"{result_dir}/q_opt.npy", q_opt_np)
    # np.save(f"{result_dir}/rpy_opt.npy", rpy_opt)
    # np.save(f"{result_dir}/loss.npy", np.array(loss_log))
    
    print(f"Saved results to {result_dir}")

def run_all_datasets():
    for num in range(1, 10):
        try:
            run_dataset("trainset", num)
        except Exception as e:
            print(f"Dataset {num} failed: {e}")

run_all_datasets()