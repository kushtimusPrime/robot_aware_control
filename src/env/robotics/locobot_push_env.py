import copy
import os
from collections import defaultdict

import numpy as np
from gym import spaces
from src.env.robotics.masks.base_mask_env import MaskEnv
from src.env.robotics.utils import (ctrl_set_action, mocap_set_action,
                                    reset_mocap2body_xpos, reset_mocap_welds)

DEBUG = False

class LocobotPushEnv(MaskEnv):
    def __init__(self, config):
        self._config = config
        modified =  config.modified
        model_path = f"locobot_push.xml"
        if modified:
            model_path = "locobot_push_fetch.xml"
        model_path = os.path.join("locobot", model_path)

        initial_qpos = None
        n_actions = 4
        n_substeps = 20
        seed = config.seed
        np.random.seed(seed)
        self._img_width = 84
        self._img_height = 84
        self._render_device = config.render_device
        if modified:
            self._joints = [f"joint_{i}" for i in range(1, 8)]
            self._gripper_joints = ['robot0:r_gripper_finger_joint', 'robot0:l_gripper_finger_joint']
        else:
            self._joints = [f"joint_{i}" for i in range(1, 8)]

        self._geoms = {
            # "robot-geom-0",
            # "robot-geom-1",
            # "robot-geom-2",
            # "robot-geom-3",
            # "robot-geom-4",
            # "robot-geom-5",
            "robot-geom-6",
            "shoulder_link_geom",
            "elbow_link_geom",
            "forearm_link_geom",
            "wrist_link_geom",
            "wrist_hole_geom",
            "gripper_link_geom",
            "ar_tag_geom",
            "gripper_hole_geom",
            "finger_r_geom",
            "finger_l_geom",
        }

        super().__init__(model_path, initial_qpos, n_actions, n_substeps, seed=seed)

        self._camera_name = "main_cam"
        # self._joints.append("gripper_revolute_joint")
        self._joint_references = [
            self.sim.model.get_joint_qpos_addr(x) for x in self._joints
        ]
        self._joint_vel_references = [
            self.sim.model.get_joint_qvel_addr(x) for x in self._joints
        ]
        self.action_space = spaces.Box(-1.0, 1.0, shape=(n_actions,), dtype="float32")

        self._objects = ["object1"]

        # workspace boundaries for eef
        # self._ws_low = [0.24, -0.17, 0.05]
        self._ws_low = [0.24, -0.17, 0.05]
        self._ws_high = [0.42, 0.17, 0.3]
        self.initial_sim_state = None


    def get_robot_mask(self, width=None, height=None):
        """
        Return binary img mask where 1 = robot and 0 = world pixel.
        robot_mask_with_obj means the robot mask is computed with object occlusions.
        """
        # returns a binary mask where robot pixels are True
        seg = self.render(
            "rgb_array", segmentation=True, width=width, height=height
        )  # flip the camera
        types = seg[:, :, 0]
        ids = seg[:, :, 1]
        geoms = types == self.mj_const.OBJ_GEOM
        geoms_ids = np.unique(ids[geoms])
        if width is None or height is None:
            mask_dim = [self._img_height, self._img_width]
        else:
            mask_dim = [height, width]
        mask = np.zeros(mask_dim, dtype=bool)
        # TODO: change these to include the robot base
        # ignore_parts = {"finger_r_geom", "finger_l_geom"}

        ignore_parts = {}
        for i in geoms_ids:
            name = self.sim.model.geom_id2name(i)
            if name is not None:
                if name in ignore_parts:
                    continue
                if name in self._geoms or "robot0" in name:
                    mask[ids == i] = True
        return mask

    def generate_masks(self, qpos_data, width=None, height=None):
        joint_references = [self.sim.model.get_joint_qpos_addr(x) for x in self._joints]
        masks = []
        for qpos in qpos_data:
            self.sim.data.qpos[joint_references] = qpos
            self.sim.forward()
            mask = self.get_robot_mask(width, height)
            masks.append(mask)
        masks = np.asarray(masks, dtype=np.bool)
        return masks

    def reset(self, initial_state=None, init_robot_qpos=True):
        """Reset the robot and block pose

        Args:
            initial_state ([type], optional): dictionary containing the robot / block poses. Defaults to None.
            init_robot_qpos (bool, optional): initialize qpos from initial_state if true. else use eef pos.

        Returns:
            [type]: [description]
        """
        if self.initial_sim_state is None:
            if self._config.modified:
                self.sim.data.qpos[self._joint_references] = [-0.25862757, -1.20163741,  0.32891832,  1.42506277, -0.10650079,  1.43468923, 0.06129823]
            else:
                # first move the arm above to avoid object collision
                # robot_above_qpos = [0.0, 0.43050715, 0.2393125, 0.63018035, 0.0, 0, 0]
                robot_above_qpos = [0.0, 0.1, 0.2393125, 0.63018035, 0, 0, 0]
                self.sim.data.qpos[self._joint_references] = robot_above_qpos
            self.sim.forward()
            self.initial_sim_state = copy.deepcopy(self.sim.get_state())
        else:
            self.sim.set_state(self.initial_sim_state)
        reset_mocap_welds(self.sim)
        reset_mocap2body_xpos(self.sim)
        # then sample object initialization
        self._sample_objects()

        eef_target_pos = [0.3, 0.0, 0.05]
        # some noise to the x/y of the eef initial pos
        noise = np.random.uniform(-0.03, 0.03, size = 2)
        eef_target_pos[:2] += noise
        self._move(eef_target_pos, threshold=0.01, max_time=100, speed=10, clip=False)
        self.set_gripper_val([0,0])

        if initial_state is not None:
            if init_robot_qpos:
                self.sim.data.qpos[self._joint_references] = initial_state["qpos"].copy()
            else:
                self._move(initial_state["states"][:3], threshold=0.01, max_time=100, speed=10, clip=False)
            self.sim.data.set_joint_qpos("object1:joint", initial_state["obj_qpos"].copy())
            self.sim.forward()
        return self._get_obs()

    def step(self, action, clip=True):
        action = np.asarray(action)
        action = np.clip(action, self.action_space.low, self.action_space.high)
        if clip:
            # disable z and gripper action space for pushing
            action[2] = 0
            action[3] = 0
            # check if applying action will violate the workspace boundary, if so, clip it.
            curr_eef_state = self.get_gripper_world_pos()
            next_eef_state = curr_eef_state + (action[:3] * 0.05)

            next_eef_state = np.clip(next_eef_state, self._ws_low, self._ws_high)
            clipped_ac = (next_eef_state - curr_eef_state) / 0.05
            action[:3] = clipped_ac
        self._set_action(action)
        # gravity compensation
        self.sim.data.qfrc_applied[
            self._joint_vel_references
        ] = self.sim.data.qfrc_bias[self._joint_vel_references]
        self.sim.step()
        self._step_callback()
        obs = self._get_obs()

        done = False
        info = {}
        reward = 0
        info["reward"] = reward
        return obs, reward, done, info

    def _set_action(self, action):
        # TODO: set joint action from end effector action using IK
        # use mocap to do it? since gripper position is in world coordinates
        action = (
            action.copy()
        )  # ensure that we don't change the action outside of this scope
        pos_ctrl, gripper_ctrl = action[:3], action[3]

        pos_ctrl *= 0.05  # limit maximum change in position
        rot_ctrl = [
            1.0,
            0.0,
            1.0,
            0.0,
        ]  # fixed rotation of the end effector, expressed as a quaternion
        # default_rot = Quaternion(self.sim.data.mocap_quat[0].copy())
        # y_rot = Quaternion(axis=[1, 0, 0], degrees=10) # Rotate 5 deg about X
        # rot_ctrl = list(default_rot * y_rot)
        # gripper_ctrl = np.array([gripper_ctrl, gripper_ctrl])
        # print(gripper_ctrl)
        # assert gripper_ctrl.shape == (2,)
        action = np.concatenate([pos_ctrl, rot_ctrl, [gripper_ctrl, gripper_ctrl]])
        # Apply action to simulation.
        ctrl_set_action(self.sim, action)
        mocap_set_action(self.sim, action)

    def _get_obs(self):
        """
        Return image, mask, robot state
        """
        if not hasattr(self, "_joint_references"):
            self._joint_references = [
                self.sim.model.get_joint_qpos_addr(x) for x in self._joints
            ]
            self._joint_vel_references = [
                self.sim.model.get_joint_qvel_addr(x) for x in self._joints
            ]
        if DEBUG:
            img = np.zeros((48,64,3))
            masks = np.zeros((48,64,1))
        else:
            img = self.render("rgb_array")
            masks = self.get_robot_mask()
        gripper_xpos = self.get_gripper_world_pos()
        # assume 0 for z, rotation, gripper force
        states = np.array([*gripper_xpos[:2],0, 0, 0])
        qpos = self.sim.data.qpos[self._joint_references].copy()
        # object qpos
        obj_qpos = self.sim.data.get_joint_qpos("object1:joint").copy()
        return {"observation": img, "masks": masks, "states": states, "qpos": qpos, "obj_qpos": obj_qpos}

    def render(self, mode="rgb_array", camera_name=None, segmentation=False, width=None, height=None):
        if width is None or height is None:
            width, height = self._img_width, self._img_height
        if camera_name is None:
            camera_name = "main_cam"
        if mode == "rgb_array":
            data = self.sim.render(
                width,
                height,
                camera_name=camera_name,
                segmentation=segmentation,
                device_id=self._render_device,
            )
            # original image is upside-down, so flip it
            return data[::-1]
        elif mode == "human":
            self._get_viewer(mode).render()

    def _sample_objects(self):
        # set objects in radius around spawn
        center = self.sim.data.get_site_xpos("blockspawn")[:2]
        spawn_id = self.sim.model.site_name2id("blockspawn")
        radius = self.sim.model.site_size[spawn_id][0]
        failed = False
        sampled_points = []
        for obj in self._objects:
            # reject sample if it overlaps with previous objects
            # reject sample if it's too close to the spawn point where the robot is
            for i in range(1000):
                no_overlap = True
                xy = self._sample_from_circle(center, radius)
                if np.linalg.norm(xy - center) < 0.08:
                    continue

                for other_xy in sampled_points:
                    if np.linalg.norm(xy - other_xy) < 0.07:
                        no_overlap = False
                        break
                if no_overlap:
                    sampled_points.append(xy)
                    break
            joint = obj + ":joint"
            pose = self.sim.data.get_joint_qpos(joint)
            z = pose[2]
            if no_overlap:
                obj_quat = [1,0,0,0]
                obj_pose = [xy[0], xy[1], z, *obj_quat]
                self.sim.data.set_joint_qpos(joint, obj_pose)
            else:
                failed = True
        # use default qpose if failed
        if failed:
            print("using default qpose since sampling failed")

    def _sample_from_circle(self, center, radius):
        """
        https://stackoverflow.com/questions/30564015/how-to-generate-random-points-in-a-circular-distribution
        """
        alpha = 2 * 3.1415 * np.random.uniform()
        r = radius * np.sqrt(np.random.uniform())
        # calculating coordinates
        x = r * np.cos(alpha) + center[0]
        y = r * np.sin(alpha) + center[1]
        return np.array([x, y])

    def _move(
        self,
        target,
        history=None,
        target_type="gripper",
        max_time=100,
        threshold=0.01,
        speed=10,
        noise=0,
        gripper=0.0,
        clip=True
    ):
        if target_type == "gripper":
            gripper_xpos = self.get_gripper_world_pos()
            d = target - gripper_xpos
        elif "object" in target_type:
            object_xpos = self.sim.data.get_site_xpos(target_type).copy()
            d = target - object_xpos
        step = 0
        while np.linalg.norm(d) > threshold and step < max_time:
            # add some random noise to ac
            if noise > 0:
                d[:3] = d[:3] + np.random.uniform(-noise, noise, size=2)
            ac = np.clip(d[:3] * speed, -1, 1)
            if clip:
                pad_ac = [*ac[:2], 0, gripper]
            else:
                pad_ac = [*ac, gripper]
            if history is not None:
                history["ac"].append(pad_ac)

            obs, _, _, info = self.step(pad_ac, clip=clip)
            if history is not None:
                history["obs"].append(obs)
                for k, v in info.items():
                    history[k].append(v)
            if target_type == "gripper":
                gripper_xpos = self.get_gripper_world_pos()
                d = target - gripper_xpos
            elif "object" in target_type:
                object_xpos = self.sim.data.get_site_xpos(target_type).copy()
                d = target - object_xpos
            step += 1

    def generate_demo(self, behavior):
        """
        Runs a hard coded behavior and stores the episode
        Returns a dictionary with observation, action
        """
        self._behavior = behavior
        obs = self.reset()
        # self.render("human")
        history = defaultdict(list)
        history["obs"].append(obs)
        ep_len = self._config.demo_length
        beta = self._config.temporal_beta
        if behavior == "temporal_random_robot":
            self.temporal_random_robot(history, ep_len, beta)
        elif behavior == "straight_push":
            self.straight_push(history)
        else:
            raise ValueError(behavior)

        return history

    def straight_push(self, history, object="object1", noise=0):
        # move gripper behind the block and oriented for a goal push
        block_xpos = self.sim.data.get_site_xpos(object).copy()
        spawn_xpos = self.sim.data.get_site_xpos("spawn").copy()
        goal_dir = (block_xpos - spawn_xpos) / np.linalg.norm(block_xpos - spawn_xpos)
        gripper_target = block_xpos - 0.05 * goal_dir
        self._move(gripper_target, history, speed=20, max_time=3)
        # push the block
        obj_target = block_xpos + 0.12 * goal_dir
        self._move(
            obj_target,
            history,
            target_type=object,
            speed=5,
            threshold=0.025,
            max_time=10,
            noise=noise,
        )

    def temporal_random_robot(self, history, ep_len, beta=1):
        """
        first moves robot near a random object, then
        generate temporally correlated actions
        """
        obj = np.random.choice(self._objects)
        history["pushed_obj"] = obj
        # move gripper behind the block and oriented for a goal push
        block_xpos = self.sim.data.get_site_xpos(obj).copy()
        spawn_xpos = self.sim.data.get_site_xpos("spawn").copy()
        goal_dir = (block_xpos - spawn_xpos) / np.linalg.norm(block_xpos - spawn_xpos)
        gripper_target = block_xpos - 0.05 * goal_dir
        # move robot near an object, record the actions
        self._move(gripper_target, history, speed=100, max_time=3, clip=True)
        past_acs = len(history["ac"])
        # generate temporally correlated noise
        u = np.zeros((ep_len - 1, *self.action_space.shape))
        actions = np.zeros_like(u)
        if past_acs > 0:
            actions[:past_acs] = history["ac"]
        for i in range(past_acs, ep_len - 1):
            u[i] = self.action_space.sample()
            u[i][2:4] = 0
            actions[i] = beta * u[i] + (1 - beta) * actions[i - 1]
        history["ac"] = actions

        for i in range(past_acs, ep_len - 1):
            obs, _, _, info = self.step(actions[i])
            history["obs"].append(obs)
            for k, v in info.items():
                history[k].append(v)

    def get_gripper_world_pos(self):
        return self.sim.data.get_site_xpos("robot0:grip").copy()

    def get_gripper_val(self):
        if self._config.modified:
            gripper_joints = self._gripper_joints
        else:
            gripper_joints = self._joints[-2:]

        return np.array([self.sim.data.get_joint_qpos(g).copy() for g in gripper_joints])

    def set_gripper_val(self, values):
        if self._config.modified:
            gripper_joints = self._gripper_joints
        else:
            gripper_joints = self._joints[-2:]
        # assumes right, then left gripper value
        self.sim.data.set_joint_qpos(gripper_joints[0], values[0])
        self.sim.data.set_joint_qpos(gripper_joints[1], values[1])
        self.sim.forward()



if __name__ == "__main__":
    import sys

    import imageio
    from src.config import argparser
    from src.utils.mujoco import init_mjrender_device

    config, _ = argparser()
    init_mjrender_device(config)
    config.gpu = 0
    config.modified = False

    DEBUG = False
    env = LocobotPushEnv(config)
    obs  = env.reset()
    # while True:
    #     env.render("human")
        # for i in range(5):
        #     env.render("human")
        #     obs, *_ = env.step([0,0,-0, 0.005])
        #     print(obs["states"][-2:])
        # for i in range(5):
        #     env.render("human")
        #     obs, *_ = env.step([-0,-0,0, -0.005])
        #     print(obs["states"][-2:])
        # break
    # img = env.render("rgb_array", camera_name="main_cam", width=640, height=480)
    # imageio.imwrite("side.png", img)
    # sys.exit(0)
    for i in range(10):
        history = env.generate_demo("straight_push")
        gif = []
        for o in history["obs"]:
            img = o["observation"]
            # mask = o["masks"]
            # img[mask] = (0, 255, 255)
            gif.append(img)
        imageio.mimwrite(f"test{i}.gif", gif)
    sys.exit(0)

    # env.get_robot_mask()
    # try locobot analytical ik
    env.reset()
    # while True:
    # #     x, y, z = np.random.uniform(low=-1, high=1, size=3)
    # #     # x, y = 0, 0
    # #     # print(x,y,z)
    # #     obs = env.step([x, y, 0])
    #     env.render("human")
    # env.render("human")
    # gif = []
    # obs = env.reset()
    # print(env.get_gripper_world_pos())
    # gif.append(obs["observation"])
    # for i in range(20):
    #     x,y,z = [0,1,0]
    #     obs, _, _, _ = env.step([x, y, z])
    #     gif.append(obs["observation"])
    # # for i in range(10):
    # #     x,y,z = [0,0,1]
    # #     obs, _, _, _ = env.step([x, y, z])
    # #     gif.append(obs["observation"])

    # print(env.get_gripper_world_pos())
    # imageio.mimwrite("test2.gif", gif)
