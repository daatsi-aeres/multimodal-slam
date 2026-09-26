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
    # axis=0 ensures we unwrap along the time dimension
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
    """
    Plots Roll, Pitch, and Yaw for:
    1. VICON Ground Truth (Black Dashed)
    2. Raw IMU Integration (Blue)
    3. Optimized PGD Estimate (Red)
    """
    labels = ['Roll', 'Pitch', 'Yaw']
    
    # Unwrap all angles to prevent +/- Pi jumps
    rpy_imu_plot = np.unwrap(rpy_imu, axis=0)
    rpy_opt_plot = np.unwrap(rpy_opt, axis=0)
    vicon_rpy_plot = np.unwrap(vicon_rpy, axis=0)
    
    plt.figure(figsize=(14, 10))
    
    for i in range(3):
        plt.subplot(3, 1, i+1)
        
        # 1. Plot Ground Truth (VICON) - Reference
        plt.plot(vicon_ts, vicon_rpy_plot[:, i], 
                 color='black', linestyle='--', linewidth=2, label="VICON (Ground Truth)", alpha=0.8)
        
        # 2. Plot Raw IMU Integration - To show drift
        plt.plot(imu_ts, rpy_imu_plot[:, i], 
                 color='blue', linestyle='-', linewidth=1, label="Raw IMU Integration", alpha=0.6)
        
        # 3. Plot Optimized Result - To show correction
        plt.plot(imu_ts, rpy_opt_plot[:, i], 
                 color='red', linestyle='-', linewidth=1.5, label="Optimized (PGD)")
        
        plt.ylabel(f"{labels[i]} (rad)", fontsize=12)
        plt.grid(True, which='both', linestyle='--', alpha=0.5)
        
        # Only put legend on the first subplot to reduce clutter
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

    # 1. Load IMU and VICON data
    # naming it in a way that i can easily change the last number to test on different datasets (1-10)

    imu_file = f"{BASE_PATH}/{set}/imu/imuRaw{num}.p"
    vicon_file = f"{BASE_PATH}/{set}/vicon/viconRot{num}.p"
    cam_path  = f"{BASE_PATH}/{set}/cam/cam{num}.p"


    visualize_plots = VISUALIZE

    with open(imu_file.format(set=set, num=num), 'rb') as f:
        imu_arr = pickle.load(f)

    with open(vicon_file.format(set=set, num=num), 'rb') as f:
        vicon_dict = pickle.load(f, encoding='latin1')

    imu_ts = imu_arr[0, :]
    print("shape of the imu_ts: ",imu_ts.shape)

    gyro_sens = 3.33 * 180.0 / np.pi  # mV/rad/sec (not degrees)
    acc_sens = 330  # mV/g
    gyro_raw = imu_arr[4:7, :]
    acc_raw = imu_arr[1:4, :]
    print("shape of the gyro_raw: ", gyro_raw.shape)
    print("shape of the acc_raw: ", acc_raw.shape)


    # compute bias from first 100 samples
    gyro_bias = np.mean(gyro_raw[:, :100], axis=1)

    Vref = 3300.0  # mV
    adc_max = 1023.0
    gyro_scale_factor= (Vref / (adc_max * gyro_sens)) 
    acc_scale_factor= (Vref / (adc_max * acc_sens)) 
    # convert to rad/sec
    calibrated_omega = (gyro_raw - gyro_bias[:, None]) * gyro_scale_factor # rad/sec

    # Get the mean of the first 100 stationary samples in physical units (g)
    acc_raw_physical = acc_raw * acc_scale_factor

    acc_initial_mean = np.mean(acc_raw_physical[:, :100], axis=1)


    # The bias is the deviation from the expected gravity vector [0, 0, 1]
    # my assumption: the IMU starts perfectly level.
    acc_bias = acc_initial_mean - np.array([0, 0, 1]) 

    # Calibrate the entire sequence
    calibrated_acc = acc_raw_physical - acc_bias[:, None]

    print("Mean calibrated acc:", np.mean(calibrated_acc[:, :100], axis=1))
    print("Norm:", np.linalg.norm(np.mean(calibrated_acc[:, :100], axis=1)))

    print("Mean calibrated omega:", np.mean(calibrated_omega[:, :100], axis=1))
    print("Norm omagea:", np.linalg.norm(np.mean(calibrated_omega[:, :100], axis=1)))

    print("shape of the calibrated_omega: ", calibrated_omega.shape)
    print("shape of the calibrated_acc: ", calibrated_acc.shape)


    N = imu_ts.shape[0]
    print(f"Loaded IMU data: {N} samples")

    # Initialize quaternion storage
    q = np.array([1.0, 0.0, 0.0, 0.0])  # initial orientation (identity)
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

    # Loop over all timesteps 
    def quaternion_integration(calibrated_omega, imu_ts):
        q = np.array([1.0, 0.0, 0.0, 0.0])  # initial orientation (identity)
        q_calculated = [q.copy()]
        prev_ts = imu_ts[0]
        for t in range(1, N):
            # Compute timestep delta t
            dt = imu_ts[t] - prev_ts # this is my delta time
            prev_ts = imu_ts[t]

            theta_vec = calibrated_omega[:, t-1] * dt 
            dq = exp_map(theta_vec)
            # Update orientation
            q = qmult(q, dq)  # q_{t+1} = q_t ◦ dq
            q = q / np.linalg.norm(q)  
            q_calculated.append(q.copy())
        return q_calculated

    q_calculated = quaternion_integration(calibrated_omega, imu_ts) #this is the calculated rotation based on calibrated imu data

    print("length of q_calculated: ", len(q_calculated))

    # Convert quaternions to roll-pitch-yaw
    rpy_list = [quat2euler(qi, axes='sxyz') for qi in q_calculated]  # radians
    rpy_array = np.array(rpy_list)  # shape (N, 3)

    vicon_rots = vicon_dict['rots']   # shape (3,3,M)
    vicon_ts = vicon_dict['ts'].flatten()  # shape (M,)

    #Convert rotation matrices → RPY
    vicon_rpy = []
    vicon_q_list = []
    last_valid_rpy = np.zeros(3)
    last_valid_q = np.array([1.0, 0.0, 0.0, 0.0])

    #VICON glitch handling: data was not clean had NaNs that were crashing the code
    for i in range(vicon_rots.shape[2]):
        R = vicon_rots[:, :, i]

        if not np.isfinite(R).all():
            # VICON glitch → hold last valid value
            vicon_rpy.append(last_valid_rpy)
            vicon_q_list.append(last_valid_q)
            continue

        # Project onto SO(3)
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

    print(f"Loaded Vicon RPY: {vicon_rpy}")
    print(f"Loaded IMU RPY: {rpy_array}")


    labels = ['Roll', 'Pitch', 'Yaw']
    # here we will compare the rotation purely based on imu gyroscope with the ground truth
    if visualize_plots:
        # Plot IMU vs VICON
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
  


    # We will verify predicted gravity against calibrated accelerometer (observation model)
    acc_pred = []
    for q in q_calculated:
        # Rotate World Gravity (0,0,1) into Body Frame
        # q_inv o [0,0,0,1] o q
        v_world = np.array([0, 0, 0, 1])  # Pure quaternion for z-axis
        
        v_body = qmult(qmult(q_inv(q), v_world), q) #tells us where gravity is in the body frame
        acc_pred.append(v_body[1:]) # extract vector part and scale by gravity

    acc_pred = np.array(acc_pred) #acc pred is in g's
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


    #calibration and verification over
    #shifting to torch for optimization form this point onwards

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True


    dt_np = np.diff(imu_ts)
    omega_np = calibrated_omega.T[:len(dt_np)]

    omega_t = torch.tensor(omega_np, dtype=torch.float32, device=device)
    dt_t = torch.tensor(dt_np, dtype=torch.float32, device=device)

    acc_t = torch.tensor(calibrated_acc.T, dtype=torch.float32, device=device)

    T = acc_t.shape[0]

    # Convert your pre-calculated integration results to a tensor
    q_init_tensor = torch.tensor(np.array(q_calculated), dtype=torch.float32, device=device)


    # Initialize q_opt with these values
    # This is a testthat I did to see how initialization affects optimization. Option to test simple initialization vs integrated initialization

    if use_integrated_init:
        q_opt = q_init_tensor.clone().detach().requires_grad_(True)
    else:
        # Initialize with identity quaternions everywhere except the first one
        # This tests the optimizer's ability to converge from a "flat" start
        q_simple = torch.zeros_like(q_init_tensor)
        q_simple[:, 0] = 1.0 # Set w=1 for all (identity)
        q_simple[0] = q_init_tensor[0] 
        q_opt = q_simple.clone().detach().requires_grad_(True)

    #essential forward operations

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
        return torch.cat([q[..., :1], -q[..., 1:]], dim=-1) #assuming unit quaternions

    def qexp_t(v):
        # v: (T, 3) representing (omega * dt / 2)
        alpha = torch.norm(v, dim=-1, keepdim=True) # (T, 1)
        
        # Use a small epsilon to avoid division by zero
        # For very small angles, we use the Taylor expansion: 
        # sin(a)/a ≈ 1 and cos(a) ≈ 1
        mask = (alpha < 1e-7).to(v.dtype)
        
        # Sinc-like scaling factor
        # When alpha is small, factor is 1.0
        # When alpha is large, factor is sin(alpha)/alpha
        safe_alpha = alpha + 1e-12 
        factor = (1.0 - mask) * (torch.sin(alpha) / safe_alpha) + (mask * 1.0)
        
        q_w = torch.cos(alpha)
        q_v = factor * v
        
        return torch.cat([q_w, q_v], dim=-1)


    def qlog_t(q):
        # q: (T, 4) -> [w, x, y, z]
        w = q[:, 0:1]
        v = q[:, 1:]
        norm_v = torch.norm(v, dim=-1, keepdim=True) # (T, 1)
        
        # Clamp w to avoid NaN in acos gradient at 1.0 or -1.0
        w_clamped = torch.clamp(w, -1.0 + 1e-7, 1.0 - 1e-7)
        theta = torch.acos(w_clamped) # Half angle
        
        # If angle is small, 2*acos(w)/norm(v) ≈ 2
        mask = (norm_v < 1e-7).to(v.dtype)
        safe_norm = norm_v + 1e-12
        
        # The multiplier is (2 * theta / sin(theta)) or (2 * theta / norm_v)
        # depending on if q is a unit quaternion. 
        # Using 2 * theta / safe_norm is standard for unit quaternions.
        factor = (1.0 - mask) * (2 * theta / safe_norm) + (mask * 2.0)
        
        return factor * v # Returns (T, 3) vector part of log map

    #defining the cost function

    def cost_fn(q):
        # 1. Motion Model Loss (Vectorized)
        # q[:-1] is all quaternions except the last one (q_t)
        # q[1:] is all quaternions except the first one (q_t+1)
        delta = 0.5 * dt_t.unsqueeze(1) * omega_t # Shape (T-1, 3)
        dq = qexp_t(delta) # Shape (T-1, 4)
        
        q_pred = qmul_t(q[:-1], dq) # Predicted next states
        q_err = qmul_t(qinv_t(q[1:]), q_pred) # Difference between q_t+1 and pred
        
        # qlog_t returns the 3D vector part of the log map
        motion_err_vec = qlog_t(q_err) 
        motion_loss = 0.5 * torch.sum(motion_err_vec**2)

        # 2. Observation Model Loss (Vectorized)
        # We need a batch of gravity vectors [0, 0, 0, 1] for every time step
        g = torch.zeros((q.shape[0], 4), device=device)
        g[:, 3] = 1.0 
        
        # Rotate gravity into body frame: g_hat = q_inv * g * q
        # Extract index 1: (the x, y, z parts)
        g_hat = qmul_t(qinv_t(q), qmul_t(g, q))[:, 1:] 
        obs_loss = 0.5 * torch.sum((acc_t - g_hat)**2)

        return motion_loss + obs_loss

    def run_hyperparameter_sweep(alphas=[0.1, 0.05, 0.01, 0.005, 0.001]):
            print("\n--- Starting Hyperparameter Sweep ---")
            plt.figure(figsize=(10, 6))
            
            colors = plt.cm.viridis(np.linspace(0, 1, len(alphas)))

            for i, lr in enumerate(alphas):
                # 1. Re-initialize q_opt for fair comparison
                # We use the same initialization strategy (Warm Start) for all
                q_sweep = q_init_tensor.clone().detach().requires_grad_(True)
                
                sweep_losses = []
                
                # 2. Run optimization for fixed iterations
                for _ in range(1000): # Fixed at 1000 as requested
                    loss = cost_fn(q_sweep)
                    loss.backward()
                    sweep_losses.append(loss.item())
                    
                    with torch.no_grad():
                        q_sweep -= lr * q_sweep.grad
                        q_sweep /= torch.norm(q_sweep, dim=1, keepdim=True)
                        q_sweep.grad = None
                
                # 3. Plot
                plt.plot(sweep_losses, label=f'$\\alpha={lr}$', color=colors[i], linewidth=1.5)
                print(f"LR {lr}: Final Loss = {sweep_losses[-1]:.4f}")

            plt.xlabel("Iteration")
            plt.ylabel("Loss (Log Scale)")
            plt.yscale('log') # Log scale is best for seeing convergence rates
            plt.title("Convergence Analysis for Different Learning Rates")
            plt.legend()
            plt.grid(True, which="both", linestyle='--', alpha=0.3)
            plt.tight_layout()
            plt.savefig(f"{result_dir}/hyperparam_sweep.png", dpi=300)
            plt.close()
            print("Sweep plot saved.\n")

        # UNCOMMENT THIS LINE TO RUN THE SWEEP
    # run_hyperparameter_sweep()

    #gradient descent loop
    lr = LR
    num_iters = NUM_ITERS
    loss_log = []
    for it in range(num_iters):
        loss = cost_fn(q_opt)
        loss.backward()
        loss_log.append(loss.item())


        with torch.no_grad():
            q_opt -= lr * q_opt.grad
            q_opt /= torch.norm(q_opt, dim=1, keepdim=True)  # projection
            q_opt.grad = None

        if it % 10 == 0:
            print(f"Iter {it}, cost = {loss.item():.4f}")

    q_opt_np = q_opt.detach().cpu().numpy().astype(np.float64)

    #ensuring each quaternion is a clean numpy array
    rpy_opt_list = []
    for i in range(q_opt_np.shape[0]):
        qi = q_opt_np[i]
        # ensuring qi is a 1D array of 4 elements
        rpy = quat2euler(qi, axes='sxyz')
        rpy_opt_list.append(rpy)

    rpy_opt = np.array(rpy_opt_list)

    if visualize_plots:
        labels = ['Roll', 'Pitch', 'Yaw']
        colors = ['r', 'g', 'b']

        plt.figure(figsize=(12, 10))
        for i in range(3):
            plt.subplot(3, 1, i+1)
        
            # Plot Ground Truth (VICON)
            plt.plot(vicon_ts, vicon_rpy[:, i], label='Ground Truth (Vicon)', 
                color='black', linestyle='--', linewidth=1.5, alpha=0.8)
        
            # Plot Optimized Estimate (PGD)
            plt.plot(imu_ts, rpy_opt[:, i], label='Estimated (PGD)', 
                color=colors[i], linewidth=1.5)

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


    #solving the panoramic motion problem
    #only enter if camera data is available
    # -------- Vectorized Panorama Stitching --------
    cam_file = cam_path.format(set=set, num=num)
    has_cam = num in [1, 2, 8, 9]

    if not has_cam:
        print("No camera for this dataset")
    else:
        with open(cam_path.format(set=set, num=num), "rb") as f:
            cam_data = pickle.load(f, encoding="latin1")
            
        frames = cam_data["cam"]  # (H, W, 3, N)
        cam_ts = cam_data["ts"].flatten()
        H, W, _, N_frames = frames.shape
        
        pano_H, pano_W = 800, 1600
        panorama = np.zeros((pano_H, pano_W, 3), dtype=np.uint32)  # use uint32 to avoid overflow when summing
        pano_count = np.zeros((pano_H, pano_W), dtype=np.int32)

        # Camera intrinsics
        fx = fy = 250.0
        cx = W / 2
        cy = H / 2

        # Rotation matrix from Camera to IMU frame
        R_cb = np.array([
            [0, 0, 1], # IMU X is Cam Z
            [-1, 0, 0], # IMU Y is -Cam X 
            [0, -1, 0]  # IMU Z is -Cam Y 
        ])

        # Precompute camera rays for all pixels
        u_grid, v_grid = np.meshgrid(np.arange(W), np.arange(H))  # pixel coords
        x_cam = (u_grid - cx) / fx
        y_cam = (v_grid - cy) / fy
        z_cam = np.ones_like(x_cam)

        # Stack and normalize in Camera Frame
        rays_cam = np.stack([x_cam, y_cam, z_cam], axis=-1).reshape(-1, 3)
        rays_cam /= np.linalg.norm(rays_cam, axis=-1, keepdims=True)

        # Transform all rays into the IMU Frame once
        rays_imu = (R_cb @ rays_cam.T).T

        # normalize
        rays_flat = rays_imu.reshape(-1, 3)
        rays_flat /= np.linalg.norm(rays_flat, axis=-1, keepdims=True)

        
        def rotate_vectors(q, v):
            """Rotate N vectors v (N,3) by quaternion q (4,)"""
            w, x, y, z = q
            t = 2 * np.cross([x, y, z], v)
            return v + w * t + np.cross([x, y, z], t)
        
        # Loop over frames, but rotate all pixels at once
        step = STEP_PANO
        use_vicon = USE_VICON_FOR_PANO

        for k in range(0, N_frames, step):
            t = cam_ts[k]
            
            if use_vicon:
                # Sync with Vicon timestamps
                idx = np.searchsorted(vicon_ts, t, side='right') - 1
                idx = np.clip(idx, 0, len(vicon_ts) - 1)
                q_curr = q_vicon[idx]
            else:
                # Sync with IMU timestamps 
                idx = np.searchsorted(imu_ts, t, side='right') - 1
                idx = np.clip(idx, 0, len(imu_ts) - 1)
                q_curr = q_opt_np[idx]
                
            img = frames[:, :, :, k]  # (H,W,3)
            img_flat = img.reshape(-1, 3)
            
            # Rotate all rays
            rays_world = rotate_vectors(q_curr, rays_flat)  # (H*W, 3)
            
            xw, yw, zw = rays_world[:, 0], rays_world[:, 1], rays_world[:, 2]
            
            # Spherical coordinates
            lon = np.arctan2(xw, yw)
            lat = np.arcsin(np.clip(zw, -1.0, 1.0))
            
            # Map to panorama
            u_p = ((lon + np.pi) / (2*np.pi) * pano_W).astype(int)
            v_p = ((np.pi/2 - lat) / np.pi * pano_H).astype(int)
            
            # Clamp
            u_p = np.clip(u_p, 0, pano_W-1)
            v_p = np.clip(v_p, 0, pano_H-1)
            
            # Paint panorama (vectorized)
            panorama[v_p, u_p] += img_flat
            pano_count[v_p, u_p] += 1

        # Final averaging
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


    # Save panorama only if available
    if has_cam and "panorama" in locals():
        plt.imsave(f"{result_dir}/panorama.png", panorama_final)

    # # Save raw data for me to check
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
