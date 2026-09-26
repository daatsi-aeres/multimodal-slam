
import numpy as np
from utils import read_canonical_model, load_pc, visualize_icp_result
from icp_3d import icp_3d, icp_with_yaw_init

if __name__ == "__main__":
  obj_name = 'liq_container' # drill or liq_container
  num_pc = 4 # number of point clouds

  source_pc = read_canonical_model(obj_name)

  for i in range(num_pc):
    target_pc = load_pc(obj_name, i)

    pose = np.eye(4)
    pose, mse = icp_with_yaw_init(source_pc, target_pc)
    print(f"PC {i}: MSE = {mse:.6f}")
    visualize_icp_result(source_pc, target_pc, pose)
    