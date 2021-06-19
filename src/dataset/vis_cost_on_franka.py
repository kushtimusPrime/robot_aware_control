import os
import numpy as np
import imageio
import h5py
from numpy.lib.type_check import imag
from scipy.spatial.transform.rotation import Rotation
from tqdm import tqdm
import matplotlib.pyplot as plt

from src.env.robotics.masks.franka_mask_env import FrankaMaskEnv
from src.prediction.losses import RobotWorldCost
from src.utils.state import State


def compute_all_rews(imgs, masks, cf):
    cost = RobotWorldCost(cf)
    goal_img = imgs[-1]
    goal_mask = masks[-1]

    all_rew = []
    all_relative_rew = []
    for i in range(len(imgs)):
        if i == 1 or i == 3:
            continue
        if "dontcare" in cf.reward_type:
            goal_state = State(img=goal_img, mask=goal_mask)
            curr_state = State(img=imgs[i], mask=masks[i])
        else:
            goal_state = State(img=goal_img)
            curr_state = State(img=imgs[i])

        rew = cost(curr_state, goal_state)
        if i == 0:
            ref_rew = rew
        relative_rew = rew / ref_rew
        # print(relative_rew)

        all_rew.append(rew)
        all_relative_rew.append(relative_rew)
    return all_rew, all_relative_rew


if __name__ == "__main__":
    from src.config import create_parser

    parser = create_parser()
    cf, unparsed = parser.parse_known_args()

    VISUALIZE = True
    DATA_ROOT = "/home/pallab/locobot_ws/src/eef_control/data/franka_views/vis_cost"
    camera_extrinsics = np.array(
        [
            [-0.00589602, 0.76599739, -0.64281664, 1.11131074],
            [0.9983059, -0.03270131, -0.04812437, 0.07869842 - 0.01],
            [-0.05788409, -0.64201138, -0.76450691, 0.59455265 + 0.02],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    offset = [0.0, 0.0, 0.0]
    rot_matrix = camera_extrinsics[:3, :3]
    cam_pos = camera_extrinsics[:3, 3]
    rel_rot = Rotation.from_quat([0, 1, 0, 0])  # calculated
    cam_rot = Rotation.from_matrix(rot_matrix)

    env = FrankaMaskEnv()
    env.set_opencv_camera_pose("main_cam", camera_extrinsics, offset)

    # load the franka qpos data and generate mask
    files = []
    for d in os.scandir(DATA_ROOT):
        if d.is_file() and d.path.endswith("hdf5"):
            files.append(d.path)

    plt.figure(dpi=300)
    count = 0
    linestype = ["", "o", "x", "-", "."]

    for fp in tqdm(files):
        imgs = None
        with h5py.File(fp, "r") as hf:
            qpos = hf["qpos"][:]
            states = hf["states"][:]
            imgs = hf["observations"][:]

        masks = None
        if VISUALIZE:
            masks = env.generate_masks(qpos, 640, 480)
            gif = []
            for i in range(len(imgs)):
                img = np.copy(imgs[i])
                imageio.imwrite(fp + "_" + str(i) + ".png", img)
                mask = masks[i]
                img[mask] = (0, 255, 0)
                gif.append(img)
            imageio.mimwrite(f"{fp}.gif", gif, fps=2)
            imageio.mimwrite(f"{fp}_imgs.gif", imgs, fps=2)
        else:
            masks = env.generate_masks(qpos, 64, 48)
            with h5py.File(fp, "a") as hf:
                if "masks" in hf.keys():
                    hf["masks"][:] = masks
                else:
                    hf.create_dataset("masks", data=masks)

        obj_name = None
        if "cost_vis_data_2021-05-28_02 16 14" in fp:
            obj_name = "bear"
        elif "cost_vis_data_2021-05-28_03_16_16" in fp:
            obj_name = "octopus"
        elif "cost_vis_data_2021-05-28_03_16_48" in fp:
            obj_name = "box"

        cf.reward_type = "dontcare"
        all_rew, all_relative_rew = compute_all_rews(imgs, masks, cf)
        plt.plot(
            all_relative_rew[:-1],
            "r-" + linestype[count % len(linestype)],
            label=obj_name + "_RA-C",
        )

        cf.reward_type = "dense"
        all_rew, all_relative_rew = compute_all_rews(imgs, masks, cf)
        plt.plot(
            all_relative_rew[:-1],
            "b-" + linestype[count % len(linestype)],
            label=obj_name + "_V-C",
        )

        count += 1

    plt.legend()
    plt.xlabel("time step")
    plt.ylabel("relative reward")
    plt.ylim([0, 1.1])
    plt.savefig(DATA_ROOT + "/cost_vis.png")
    plt.show()