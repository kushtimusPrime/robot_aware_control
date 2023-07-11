import os
import copy
import mujoco_py
import numpy as np
import gym
from gym import error, spaces
from PIL import Image as im
import h5py
import io

DEFAULT_SIZE = 500
DEVICE_ID = -1

class KukaRobotPixel():
    def __init__(self,model_path,initial_qpos,n_actions,n_substeps,seed=None):
        print("Start Robot Env Init")
        model = mujoco_py.load_model_from_path(model_path)
        
        self.mj_const = mujoco_py.const
        self.sim = mujoco_py.MjSim(model, nsubsteps=n_substeps)
        self.viewer = None
        self._viewers = {}

        self.metadata = {
            "render.modes": ["human", "rgb_array"],
            "video.frames_per_second": int(np.round(1.0 / self.dt)),
        }

        self._seed = self.seed(seed)
        self._env_setup(initial_qpos=initial_qpos)
        self.initial_state = copy.deepcopy(self.sim.get_state())

        self.goal = self._sample_goal()
        obs = self._get_obs()
        self.action_space = spaces.Box(-1.0, 1.0, shape=(n_actions,), dtype="float32")

        self.observation_space = spaces.Dict(
            dict(
                observation=spaces.Box(
                    -np.inf, np.inf, shape=obs["observation"].shape, dtype="float32"
                ),
            )
        )
        self._img_width = 160
        self._img_height = 120
        self._camera_name = "main_cam"
        self._joints = [f"joint_{i}" for i in range(1,8)]

    def baseRender(
        self,
        mode="human",
        width=DEFAULT_SIZE,
        height=DEFAULT_SIZE,
        camera_name=None,
        segmentation=False,
    ):
        if mode == "rgb_array":
            data = self.sim.render(
                width,
                height,
                camera_name=camera_name,
                segmentation=segmentation,
                device_id=DEVICE_ID,
            )
            # original image is upside-down, so flip it
            return data[::-1, :, :]
        elif mode == "human":
            self._get_viewer(mode).render()

    def render(self, mode, segmentation=False):
        if mode == "rgb_array":
            out = self.baseRender(
                mode,
                width=self._img_width,
                height=self._img_height,
                camera_name=self._camera_name,
                segmentation=segmentation,
            )
            return out[::-1, ::-1]
        elif mode == "human":
            self.baseRender(mode)

    def get_robot_mask(self):
        """
        Return binary img mask where 1 = robot and 0 = world pixel.
        robot_mask_with_obj means the robot mask is computed with object occlusions.
        """
        # returns a binary mask where robot pixels are True
        seg = self.render("rgb_array", segmentation=True)  # flip the camera
        types = seg[:, :, 0]
        ids = seg[:, :, 1]
        geoms = types == self.mj_const.OBJ_GEOM
        geoms_ids = np.unique(ids[geoms])
        mask_dim = [self._img_height, self._img_width]
        mask = np.zeros(mask_dim, dtype=np.bool)
        ignore_parts = {"base_link_vis", "base_link_col", "head_vis"}
        for i in geoms_ids:
            name = self.sim.model.geom_id2name(i)
            if name is not None:
                if name in ignore_parts:
                    continue
                a = "vis" in name
                b = "col" in name
                c = "gripper" in name
                d = "finger" in name
                if any([a, b, c, d]):
                    mask[ids == i] = True
        return mask
    
    @property
    def dt(self):
        return self.sim.model.opt.timestep * self.sim.nsubsteps
    
    def seed(self, seed=None):
        self.np_random = np.random.RandomState(seed)
        return seed
    
    def _env_setup(self, initial_qpos):
        """Initial configuration of the environment. Can be used to configure initial state
        and extract information from the simulation.
        """
        print(initial_qpos.shape)
        for name, value in initial_qpos.items():
            self.sim.data.set_joint_qpos(name, value)
        self.sim.forward()

    def _sample_goal(self):
        """Samples a new goal and returns it."""
        pass
    
    def _get_obs(self):
        return {"observation": np.array([0])}

def main():
    print("Hello World!")
    qpos_file_path = "hdf5/penn_kuka_traj1.hdf5"
    
    the_qpos = None
    with open(qpos_file_path, 'rb') as f:
        buf = f.read()

    with h5py.File(io.BytesIO(buf)) as hf:
        the_qpos = hf['env']['qpos'][:]
    print(the_qpos)
    print("Earl xit")
    model_path = "src/env/robotics/assets/kuka/robot.xml"
    initial_qpos = the_qpos
    n_actions = 1
    n_substeps = 1
    seed = None
    kuka_robot_pixel_test = KukaRobotPixel(model_path, initial_qpos, n_actions, n_substeps, seed=seed)
    maybe_kuka_photo = kuka_robot_pixel_test.get_robot_mask()
    data = im.fromarray(maybe_kuka_photo)
    data.save('test.png')

if __name__ == "__main__":
    main()
