import os
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from pyrep.const import RenderMode
from pyrep.errors import IKError
from rlbench.environment import Environment
from rlbench.task_environment import InvalidActionError
from rlbench.action_modes import ArmActionMode, ActionMode
from rlbench.observation_config import ObservationConfig, CameraConfig
from utils import task_switch, euler_to_quaternion

os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = os.environ["COPPELIASIM_ROOT"]


class GripperPlot:
    def __init__(self, headless):
        self.headless = headless
        if headless:
            return
        self.displayed_gripper = 0.9
        self.fig = plt.figure()
        ax = self.fig.add_subplot(111)
        ax.set_xlim(-1.25, 1.25)
        ax.set_ylim(-1.25, 1.25)
        horizontal_patch = plt.Rectangle((-1, 0), 2, 0.6)
        self.left_patch = plt.Rectangle((-0.9, -1), 0.4, 1, color="black")
        self.right_patch = plt.Rectangle((0.5, -1), 0.4, 1, color="black")
        ax.add_patch(horizontal_patch)
        ax.add_patch(self.left_patch)
        ax.add_patch(self.right_patch)
        self.fig.canvas.draw()
        plt.show(block=False)
        plt.pause(0.1)
        for _ in range(2):
            self.set_data(0)
            plt.pause(0.1)
            self.set_data(1)
            plt.pause(0.1)
        return

    def set_data(self, last_gripper_open):
        if self.headless:
            return
        if self.displayed_gripper == last_gripper_open:
            return
        if last_gripper_open == 0.9:
            self.displayed_gripper = 0.9
            self.left_patch.set_xy((-0.9, -1))
            self.right_patch.set_xy((0.5, -1))
        elif last_gripper_open == -0.9:
            self.displayed_gripper = -0.9
            self.left_patch.set_xy((-0.4, -1))
            self.right_patch.set_xy((0, -1))
        self.fig.canvas.draw()
        plt.pause(0.01)
        return

    def reset(self):
        self.set_data(1)


class CustomEnv:
    def __init__(self, config):
        # image_size=(128, 128)
        obs_config = ObservationConfig(
            left_shoulder_camera=CameraConfig(rgb=False, depth=False, mask=False),
            right_shoulder_camera=CameraConfig(rgb=False, depth=False, mask=False),
            front_camera=CameraConfig(rgb=False, depth=False, mask=False),
            wrist_camera=CameraConfig(
                rgb=True, depth=False, mask=False, render_mode=RenderMode.OPENGL
            ),
            joint_positions=True,
            joint_velocities=True,
            joint_forces=False,
            gripper_pose=False,
            task_low_dim_state=False,
        )
        action_mode = ActionMode(ArmActionMode.EE_POSE_EE_FRAME)
        self.env = Environment(
            action_mode,
            obs_config=obs_config,
            static_positions=config["static_env"],
            headless=config["headless_env"],
        )
        self.env.launch()
        self.task = self.env.get_task(task_switch[config["task"]])
        self.gripper_plot = GripperPlot(config["headless_env"])
        self.gripper_open = 0.9
        self.gripper_deque = deque([0.9] * 20, maxlen=20)
        return

    def reset(self):
        self.gripper_plot.reset()
        self.gripper_open = 0.9
        self.gripper_deque = deque([0.9] * 20, maxlen=20)
        descriptions, obs = self.task.reset()
        camera_obs, proprio_obs = obs_split(obs)
        return camera_obs, proprio_obs

    def step(self, action):
        action_delayed = self.postprocess_action(action)
        try:
            next_obs, reward, done = self.task.step(action_delayed)
        except (IKError and InvalidActionError):
            zero_action = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, action_delayed[-1]]
            next_obs, reward, done = self.task.step(zero_action)
        camera_obs, proprio_obs = obs_split(next_obs)
        return camera_obs, proprio_obs, reward, done

    def render(self):
        return

    def close(self):
        self.env.shutdown()
        return

    def postprocess_action(self, action):
        delta_position = action[:3] * 0.01
        delta_angle_quat = euler_to_quaternion(action[3:6] * 0.04)
        gripper_delayed = self.delay_gripper(action[-1])
        action_post = np.concatenate(
            (delta_position, delta_angle_quat, [gripper_delayed])
        )
        return action_post

    def delay_gripper(self, gripper_action):
        if gripper_action >= 0.0:
            gripper_action = 0.9
        elif gripper_action < 0.0:
            gripper_action = -0.9
        self.gripper_plot.set_data(gripper_action)
        self.gripper_deque.append(gripper_action)
        if all([x == 0.9 for x in self.gripper_deque]):
            self.gripper_open = 1
        elif all([x == -0.9 for x in self.gripper_deque]):
            self.gripper_open = 0
        return self.gripper_open


def obs_split(obs):
    camera_obs = obs.wrist_rgb.transpose(
        (2, 0, 1)
    )  # Transpose it into torch order (CHW)
    proprio_obs = np.append(obs.joint_positions, obs.gripper_open)
    return camera_obs, proprio_obs
