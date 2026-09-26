import numpy as np
import matplotlib.pyplot as plt
from transforms3d.quaternions import qmult, axangle2quat
from transforms3d.euler import quat2euler 
from transforms3d.quaternions import mat2quat
import pickle
import torch


# Load IMU
# naming it in a way that i can easily change the last number to test on different datasets (1-10)

imu_file = "data/{set}/imu/imuRaw{num}.p"
cam_path = "data/{set}/cam/cam{num}.p"

set="testset"
num = 11
visualize_plots = True

with open(imu_file.format(set=set, num=num), 'rb') as f:
    imu_arr = pickle.load(f)

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

g_dir = acc_initial_mean / np.linalg.norm(acc_initial_mean)


# The bias is the deviation from the expected gravity vector [0, 0, 1]
# assuming the IMU starts perfectly level.
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
prev_ts = imu_ts[0]

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

q_calculated = quaternion_integration(calibrated_omega, imu_ts)

print("length of q_calculated: ", len(q_calculated))

# Convert quaternions to roll-pitch-yaw
rpy_list = [quat2euler(qi, axes='sxyz') for qi in q_calculated]  # radians
rpy_array = np.array(rpy_list)  # shape (N, 3)


print(f"Loaded IMU RPY: {rpy_array}")


labels = ['Roll', 'Pitch', 'Yaw']

if visualize_plots:
    # Plot IMU 
    plt.figure(figsize=(14, 8))
    for i in range(3):
        plt.subplot(3, 1, i+1)
        plt.plot(imu_ts, rpy_array[:, i], label='IMU', linewidth=1)
        plt.ylabel(labels[i] + ' (rad)')
        plt.legend()
        plt.grid(True)

    plt.xlabel('Time (s)')
    plt.suptitle('IMU Orientation')
    plt.tight_layout()
    plt.show()      


# Verify predicted gravity against accelerometer (observation model)
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

#calibration and verification over
#shifting to torch for optimization form this point onwards

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

omega_t = torch.tensor(calibrated_omega.T[:-1], dtype=torch.float32, device=device)
dt_t = torch.tensor(np.diff(imu_ts), dtype=torch.float32, device=device)
acc_t = torch.tensor(calibrated_acc.T, dtype=torch.float32, device=device)

T = acc_t.shape[0]

q_init_tensor = torch.tensor(np.array(q_calculated), dtype=torch.float32, device=device)

# Initialize q_opt with these values
# Option to test simple initialization vs integrated initialization
use_integrated_init = True 

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
    
    mask = (alpha < 1e-7).to(v.dtype)
    

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

    # Observation Model Loss (Vectorized)
    # We need a batch of gravity vectors [0, 0, 0, 1] for every time step
    g = torch.zeros((q.shape[0], 4), device=device)
    g[:, 3] = 1.0 
    
    # Rotate gravity into body frame: g_hat = q_inv * g * q
    # Extract index 1: (the x, y, z parts)
    g_hat = qmul_t(qinv_t(q), qmul_t(g, q))[:, 1:] 
    obs_loss = 0.5 * torch.sum((acc_t - g_hat)**2)

    return motion_loss + obs_loss

#gradient descent loop
lr = 0.01
num_iters = 100

for it in range(num_iters):
    loss = cost_fn(q_opt)
    loss.backward()

    with torch.no_grad():
        q_opt -= lr * q_opt.grad
        q_opt /= torch.norm(q_opt, dim=1, keepdim=True)  # projection
        q_opt.grad.zero_()

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
    
        # Plot IMU Raw
        plt.plot(imu_ts, rpy_array[:, i], label='Raw (IMU)', 
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

#solving the panoramic motion problem
#only enter if camera data is available
# -------- Vectorized Panorama Stitching --------
if cam_path is None:
    print("no camera images to process")
else:
    with open(cam_path.format(set=set,num=num), "rb") as f:
        cam_data = pickle.load(f, encoding="latin1")
        
    frames = cam_data["cam"]  # (H, W, 3, N)
    cam_ts = cam_data["ts"].flatten()
    H, W, _, N_frames = frames.shape
    
    pano_H, pano_W = 800, 1600
    panorama = np.zeros((pano_H, pano_W, 3), dtype=np.uint32)  
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

    # Transform all rays into the IMU Frame ONCE
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
    step = 1
    for k in range(0, N_frames, step):
        t = cam_ts[k]
 
        # Sync with IMU timestamps (for optimized result)
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
    # panorama_final = np.zeros_like(panorama, dtype=np.uint8)
    panorama[mask] = (panorama[mask] / pano_count[mask][:, None]).astype(np.uint8)

    print(panorama.shape)
    print("Non-zero pixels:", np.count_nonzero(panorama))
    
    plt.figure(figsize=(12, 6))
    plt.imshow(panorama)
    plt.title("Vectorized Panorama (Orientation-based Stitching)")
    plt.axis("off")
    plt.show()