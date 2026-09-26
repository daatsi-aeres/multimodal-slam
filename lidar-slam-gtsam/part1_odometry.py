import numpy as np
import matplotlib.pyplot as plt
import os


def load_sensor_data(dataset=20):
    """Load encoder and IMU data. Returns dicts with arrays and timestamps."""
    encoder_data = np.load("data/Encoders%d.npz"%dataset)
    imu_data = np.load("data/Imu%d.npz"%dataset)
    return encoder_data, imu_data


def compute_wheel_velocities(encoder_counts, encoder_stamps):
    delta_t = np.diff(encoder_stamps)
    distance_per_tick = 0.0022  # meters per encoder tick
    FR = encoder_counts[0, 1:]
    FL = encoder_counts[1, 1:]
    RR = encoder_counts[2, 1:]
    RL = encoder_counts[3, 1:]
    v_R = (FR + RR) / 2 * distance_per_tick / delta_t
    v_L = (FL + RL) / 2 * distance_per_tick / delta_t
    velocities = {"v_R": v_R, "v_L": v_L, "time_stamps": encoder_stamps[1:]}
    return velocities


def motion_model_step(state, v, omega, dt):
    """
    Propagate the robot state one step forward.

    state: [x, y, theta]
    v:     linear velocity (m/s)
    omega: yaw rate (rad/s)
    dt:    time step (s)

    Returns: new state [x, y, theta]
    """
    x_new = state[0] + v * np.cos(state[2]) * dt
    y_new = state[1] + v * np.sin(state[2]) * dt
    theta_new = state[2] + omega * dt
    return np.array([x_new, y_new, theta_new])


def integrate_odometry(encoder_data, imu_data):
    """
    Walk through encoder timestamps, sync with nearest IMU reading,
    and integrate the differential-drive motion model step by step.

    Returns: poses (N, 3) array of [x, y, theta], timestamps (N,)
    """
    encoder_stamps = encoder_data["time_stamps"]
    imu_stamps = imu_data["time_stamps"]
    poses = []
    state = np.array([0.0, 0.0, 0.0])
    wheel_baseline = 0.3937
    imu_omega_array = []
    encoder_omega_array = []
    velocities = compute_wheel_velocities(encoder_data["counts"][:, :], encoder_data["time_stamps"][:])
    for i, t in enumerate(encoder_stamps[1:]):
        imu_idx = np.argmin(np.abs(imu_stamps - t))
        v = np.mean([velocities["v_R"][i], velocities["v_L"][i]])
        omega = imu_data["angular_velocity"][2, imu_idx]
        imu_omega_array.append(omega)
        omega_enc = (velocities["v_R"][i] - velocities["v_L"][i]) / wheel_baseline  # encoder-derived yaw, not used for integration
        encoder_omega_array.append(omega_enc)
        dt = t - encoder_stamps[len(poses)]
        state = motion_model_step(state, v, omega, dt)
        poses.append(state.copy())
    if __name__ == "__main__":
        plt.figure()
        plt.plot(imu_omega_array[:500], label='IMU yaw rate')
        plt.plot(encoder_omega_array[:500], label='Encoder yaw rate')
        plt.legend()
        plt.title('omega comparison')
        plt.savefig("plots/part1_odometry/omega_comparison.png", dpi=300, bbox_inches="tight")
        plt.show()

    return np.array(poses), encoder_stamps[1:]


def plot_trajectory(poses):
    """Plot the (x, y) robot trajectory."""
    x = poses[:, 0]
    y = poses[:, 1]

    plt.figure(figsize=(8, 8))
    plt.plot(x, y, linewidth=2)
    plt.scatter(x[0], y[0], c='green', s=100, label="Start")
    plt.scatter(x[-1], y[-1], c='red', s=100, label="End")

    plt.xlabel("X position (m)")
    plt.ylabel("Y position (m)")
    plt.title("Robot Trajectory from Odometry")
    plt.legend()
    plt.axis("equal")
    plt.grid(True)

    save_dir = "plots/part1_odometry"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "odometry_trajectory_21.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Plot saved to: {save_path}")


if __name__ == "__main__":
    encoder_data, imu_data = load_sensor_data(dataset=21)
    poses, timestamps = integrate_odometry(encoder_data, imu_data)
    plot_trajectory(poses)
    plt.show()
