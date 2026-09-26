# Orientation Tracking & Panorama Stitching

Orientation estimation using IMU data, plus panoramic stitching. The approach uses **Projected Gradient Descent (PGD)** to optimize quaternions by minimizing the error between a gyro-based motion model and an accelerometer-based observation model.

---

## Setup & Running the Code

1. **Dependencies**: You'll need `numpy`, `matplotlib`, `torch`, `pickle`, and `transforms3d`.
2. **Data Path**: Update the `BASE_PATH` variable in the `CONFIG` section of `main.py` to point to your data directory.
3. **Run**: Execute the script using:
```bash
python main.py

```


4. **Outputs**: All plots and panoramas are saved in a generated `results/` directory, organized by dataset name and number.

---

## Features

### Configuration Toggles

The `# ================= CONFIG =================` block at the top allows you to switch between different modes:

* `VISUALIZE`: Set to `True` for interactive `plt.show()` debugging.
* `use_integrated_init`: Toggle between a "Warm Start" (using gyro integration) or a "Cold Start" (identity quaternions) for the optimizer.
* `USE_VICON_FOR_PANO`: Choose whether to stitch the panorama using VICON ground truth or my optimized PGD estimates.
* `STEP_PANO`: Skips frames to speed up stitching (default is 50).

### Additional Tools (Uncomment to enable)

I have left several functional blocks commented out at the bottom of the `run_dataset` function for the TA/Professor to use if they want to dig deeper:

* `run_hyperparameter_sweep()`: Generates a convergence plot comparing different learning rates ().
* `save_acc_plot`: Saves a comparison of measured acceleration vs. predicted gravity.
* `save_loss_plot`: Exports the optimization loss curve as a `.png`.
* `np.save(...)`: Exports raw `q_opt` and `rpy` data as `.npy` files for verification.

---
