import os
from scipy.spatial.transform.rotation import Rotation
import numpy as np
import imageio
import h5py
from pupil_apriltags import Detector
import cv2

from src.env.robotics.masks.base_mask_env import MaskEnv
from src.env.robotics.masks.locobot_analytical_ik import AnalyticInverseKinematics as AIK


class WX250sMaskEnv(MaskEnv):
    def __init__(self):
        self.thick = True
        model_path = os.path.join("wx250s", "model.xml")
        initial_qpos = None
        n_actions = 1
        n_substeps = 1
        seed = None
        super().__init__(model_path, initial_qpos, n_actions, n_substeps, seed=seed)
        self._img_width = 64
        self._img_height = 48
        self._camera_name = "main_cam"
        self._joints = ['waist', 'shoulder', 'elbow', 'forearm_roll', 'wrist_angle', 'wrist_rotate']
        # self._joints.append("gripper_revolute_joint")
        self._joint_references = [self.sim.model.get_joint_qpos_addr(x) for x in self._joints]

    def compare_traj(self, traj_name, qpos_data, eef_data, real_imgs):
        joint_references = [self.sim.model.get_joint_qpos_addr(x) for x in self._joints]
        # run qpos trajectory
        gif = []
        for i, qpos in enumerate(qpos_data):
            self.sim.data.qpos[joint_references] = qpos
            self.sim.forward()
            # self.render("human")
            # img = self.render("rgb_array")
            # eef_pos = eef_data[i][:3]
            # eef_site = self.sim.model.body_name2id("eef_body")
            # self.sim.model.body_pos[eef_site] = eef_pos
            # self.sim.forward()
            mask = self.get_robot_mask()
            real_img = real_imgs[i]
            mask_img = real_img.copy()
            mask_img[mask] = (0, 255, 255)
            # mask_img = mask_img.astype(int)
            # mask_img[mask] += (100, 0, 0)
            # mask_img = mask_img.astype(np.uint8)
            comparison = mask_img
            # comparison = np.concatenate([img, real_img, mask_img], axis=1)
            mask_img = cv2.cvtColor(mask_img, cv2.COLOR_BGR2RGB)
            # cv2.imwrite(f"{traj_name}_mask_" + str(i) + ".png", mask_img)
            gif.append(comparison)
        imageio.mimwrite(f"{traj_name}_mask.gif", gif)

    def get_robot_mask(self, width=None, height=None):
        """
        Return binary img mask where 1 = robot and 0 = world pixel.
        robot_mask_with_obj means the robot mask is computed with object occlusions.
        """
        # returns a binary mask where robot pixels are True
        seg = self.render("rgb_array", segmentation=True, width=width, height=height)  # flip the camera
        types = seg[:, :, 0]
        ids = seg[:, :, 1]
        geoms = types == self.mj_const.OBJ_GEOM
        geoms_ids = np.unique(ids[geoms])
        if width is None or height is None:
            mask_dim = [self._img_height, self._img_width]
        else:
            mask_dim = [height, width]
        mask = np.zeros(mask_dim, dtype=np.bool)
        # TODO: change these to include the robot base
        # ignore_parts = {"finger_r_geom", "finger_l_geom"}
        ignore_parts = {}
        # ignore_parts = {}
        for i in geoms_ids:
            if self.thick:
                mask[ids == i] = True
                continue

            name = self.sim.model.geom_id2name(i)
            if name is not None:
                if name in ignore_parts:
                    continue
                mask[ids == i] = True
        return mask

    def get_gripper_pos(self, qpos):
        self.sim.data.qpos[self._joint_references] = qpos
        self.sim.forward()
        return self.sim.data.get_body_xpos("gripper_link").copy()

    def generate_masks(self,  qpos_data, width=None, height=None):
        joint_references = [self.sim.model.get_joint_qpos_addr(x) for x in self._joints]
        # finger_references = [self.sim.model.get_joint_qpos_addr(x) for x in ["joint_6", "joint_7"]]
        masks = []
        for qpos in qpos_data:
            self.sim.data.qpos[joint_references] = qpos
            # self.sim.data.qpos[finger_references] = [-0.025, 0.025]
            self.sim.forward()
            mask = self.get_robot_mask(width, height)
            masks.append(mask)
        masks = np.asarray(masks, dtype=np.bool)
        return masks

def load_data(filename):
    with h5py.File(filename, "r") as f:
        # List all groups
        all_keys = [key for key in f.keys()]

        qposes, imgs, eef_states, actions = None, None, None, None
        if 'qpos' in all_keys:
            qposes = np.array(f['qpos'])
        else:
            print("ERROR! No qpos")

        if 'observations' in all_keys:
            imgs = np.array(f['observations'])
        else:
            print("ERROR! No observations")

        if 'states' in all_keys:
            eef_states = np.array(f['states'])
        else:
            print("ERROR! No states")

        if 'actions' in all_keys:
            actions = np.array(f['actions'])
        else:
            print("ERROR! No actions")
    return qposes, imgs, eef_states, actions

def load_states(filename):
    with h5py.File(filename, "r") as f:
        if 'states' in f:
            eef_states = np.array(f['states'])
        else:
            print("ERROR! No states")
    return  eef_states


def get_camera_pose_from_apriltag(image, detector=None):
    if detector is None:
        detector = Detector(families='tag36h11',
                            nthreads=1,
                            quad_decimate=1.0,
                            quad_sigma=0.0,
                            refine_edges=1,
                            decode_sharpening=0.25,
                            debug=0)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    results = []
    results = detector.detect(gray,
                              estimate_tag_pose=True,
                              camera_params=[612.45,
                                             612.45,
                                             330.55,
                                             248.61],
                              tag_size=0.0353)
    print("[INFO] {} total AprilTags detected".format(len(results)))

    if len(results) == 0:
        return None, None

    # loop over the AprilTag detection results
    for r in results:
        pose_t = r.pose_t
        pose_R = r.pose_R
        # print("pose_t", r.pose_t)
        # print("pose_R", r.pose_R)
    return pose_t, pose_R


def predict_next_qpos(eef_curr, qpos_curr, action):
    """
    eef_curr: (3, ) 3d position of eef
    qpos_curr: (5, )
    action: (2, ) planar action
    """
    # TODO: record pitch/roll in eef pose in the future
    PUSH_HEIGHT = 0.15
    DEFAULT_PITCH = 1.3
    DEFAULT_ROLL = 0.0
    eef_next = np.zeros(3)
    eef_next[0:2] = eef_curr[0:2] + action
    eef_next[2] = PUSH_HEIGHT

    ik_solver = AIK()

    qpos_next = np.zeros(5)
    qpos_next[0:4] = ik_solver.ik(eef_next, alpha=-DEFAULT_PITCH, cur_arm_config=qpos_curr[0:4])
    qpos_next[4] = DEFAULT_ROLL
    return qpos_next


def overlay_trajs(traj_path1, traj_path2):
    with h5py.File(traj_path1 + ".hdf5", "r") as f:
        imgs1 = np.array(f['observations'])
    with h5py.File(traj_path2 + ".hdf5", "r") as f:
        imgs2 = np.array(f['observations'])
    avg_img = np.zeros(imgs1[0].shape)
    for t in range(imgs1.shape[0]):
        avg_img += imgs1[t]
    for t in range(imgs2.shape[0]):
        avg_img += imgs2[t]
    avg_img /= (imgs1.shape[0] + imgs2.shape[0])

    avg_img = avg_img.astype(np.uint8)

    avg_img = cv2.cvtColor(avg_img, cv2.COLOR_BGR2RGB)
    cv2.imwrite(f"{traj_path1}_overlay.png", avg_img)

def get_gripper_pos(self, qpos):
    raise NotImplementedError