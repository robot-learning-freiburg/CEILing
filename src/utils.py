import random
import numpy as np
import wandb
import time
import torch
from torch.utils.data import Dataset
from pynput import keyboard
from functools import partial
from collections import deque
from rlbench.tasks import (
    CloseMicrowave,
    PushButton,
    TakeLidOffSaucepan,
    UnplugCharger,
)


task_switch = {
    "CloseMicrowave": CloseMicrowave,
    "PushButton": PushButton,
    "TakeLidOffSaucepan": TakeLidOffSaucepan,
    "UnplugCharger": UnplugCharger,
}
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(device)


class KeyboardObserver:
    def __init__(self):
        self.reset()
        self.hotkeys = keyboard.GlobalHotKeys(
            {
                "g": partial(self.set_label, 1),  # good
                "b": partial(self.set_label, 0),  # bad
                "c": partial(self.set_gripper, -0.9),  # close
                "v": partial(self.set_gripper, 0.9),  # open
                "f": partial(self.set_gripper, None),  # gripper free
                "x": self.reset_episode,
            }
        )
        self.hotkeys.start()
        self.direction = np.array([0, 0, 0, 0, 0, 0])
        self.listener = keyboard.Listener(
            on_press=self.set_direction, on_release=self.reset_direction
        )
        self.key_mapping = {
            "a": (1, 1),  # left
            "d": (1, -1),  # right
            "s": (0, 1),  # backward
            "w": (0, -1),  # forward
            "q": (2, 1),  # down
            "e": (2, -1),  # up
            "j": (3, -1),  # look left
            "l": (3, 1),  # look right
            "i": (4, -1),  # look up
            "k": (4, 1),  # look down
            "u": (5, -1),  # rotate left
            "o": (5, 1),  # rotate right
        }
        self.listener.start()
        return

    def set_label(self, value):
        self.label = value
        print("label set to: ", value)
        return

    def get_label(self):
        return self.label

    def set_gripper(self, value):
        self.gripper_open = value
        print("gripper set to: ", value)
        return

    def get_gripper(self):
        return self.gripper_open

    def set_direction(self, key):
        try:
            idx, value = self.key_mapping[key.char]
            self.direction[idx] = value
        except (KeyError, AttributeError):
            pass
        return

    def reset_direction(self, key):
        try:
            idx, _ = self.key_mapping[key.char]
            self.direction[idx] = 0
        except (KeyError, AttributeError):
            pass
        return

    def has_joints_cor(self):
        return self.direction.any()

    def has_gripper_update(self):
        return self.get_gripper() is not None

    def get_ee_action(self):
        return self.direction * 0.9

    def reset_episode(self):
        self.reset_button = True
        return

    def reset(self):
        self.set_label(1)
        self.set_gripper(None)
        self.reset_button = False
        return


class MetricsLogger:
    def __init__(self):
        self.total_successes = 0
        self.total_episodes = 0
        self.total_steps = 0
        self.total_cor_steps = 0
        self.total_pos_steps = 0
        self.total_neg_steps = 0
        self.episode_metrics = deque(maxlen=1)
        self.reset_episode()
        return

    def reset_episode(self):
        self.episode_reward = 0
        self.episode_steps = 0
        self.episode_cor_steps = 0
        self.episode_pos_steps = 0
        self.episode_neg_steps = 0
        return

    def log_step(self, reward, feedback):
        self.episode_reward += reward
        self.episode_steps += 1
        if feedback == -1:
            self.episode_cor_steps += 1
        elif feedback == 1:
            self.episode_pos_steps += 1
        elif feedback == 0:
            self.episode_neg_steps += 1
        else:
            raise NotImplementedError
        return

    def log_episode(self, current_episode):
        episode_metrics = {
            "reward": self.episode_reward,
            "ep_cor_rate": self.episode_cor_steps / self.episode_steps,
            "ep_pos_rate": self.episode_pos_steps / self.episode_steps,
            "ep_neg_rate": self.episode_neg_steps / self.episode_steps,
            "episode": current_episode,
        }
        self.append(episode_metrics)
        self.total_episodes += 1
        if self.episode_reward > 0:
            self.total_successes += 1
        self.total_steps += self.episode_steps
        self.total_cor_steps += self.episode_cor_steps
        self.total_pos_steps += self.episode_pos_steps
        self.total_neg_steps += self.episode_neg_steps
        self.reset_episode()
        return

    def log_session(self):
        success_rate = self.total_successes / self.total_episodes
        cor_rate = self.total_cor_steps / self.total_steps
        pos_rate = self.total_pos_steps / self.total_steps
        neg_rate = self.total_neg_steps / self.total_steps
        wandb.run.summary["success_rate"] = success_rate
        wandb.run.summary["total_cor_rate"] = cor_rate
        wandb.run.summary["total_pos_rate"] = pos_rate
        wandb.run.summary["total_neg_rate"] = neg_rate
        return

    def append(self, episode_metrics):
        self.episode_metrics.append(episode_metrics)
        return

    def pop(self):
        return self.episode_metrics.popleft()

    def empty(self):
        return len(self.episode_metrics) == 0


class TrajectoriesDataset(Dataset):
    def __init__(self, sequence_len):
        self.sequence_len = sequence_len
        self.camera_obs = []
        self.proprio_obs = []
        self.action = []
        self.feedback = []
        self.reset_current_traj()
        self.pos_count = 0
        self.cor_count = 0
        self.neg_count = 0
        return

    def __getitem__(self, idx):
        if self.cor_count < 10:
            alpha = 1
        else:
            alpha = (self.pos_count + self.neg_count) / self.cor_count
        weighted_feedback = [
            alpha if value == -1 else value for value in self.feedback[idx]
        ]
        weighted_feedback = torch.tensor(weighted_feedback).unsqueeze(1)
        return (
            self.camera_obs[idx],
            self.proprio_obs[idx],
            self.action[idx],
            weighted_feedback,
        )

    def __len__(self):
        return len(self.proprio_obs)

    def add(self, camera_obs, proprio_obs, action, feedback):
        self.current_camera_obs.append(camera_obs)
        self.current_proprio_obs.append(proprio_obs)
        self.current_action.append(action)
        self.current_feedback.append(feedback)
        if feedback[0] == 1:
            self.pos_count += 1
        elif feedback[0] == -1:
            self.cor_count += 1
        elif feedback[0] == 0:
            self.neg_count += 1
        return

    def save_current_traj(self):
        camera_obs = downsample_traj(self.current_camera_obs, self.sequence_len)
        proprio_obs = downsample_traj(self.current_proprio_obs, self.sequence_len)
        action = downsample_traj(self.current_action, self.sequence_len)
        feedback = downsample_traj(self.current_feedback, self.sequence_len)
        camera_obs_th = torch.tensor(camera_obs, dtype=torch.float32)
        proprio_obs_th = torch.tensor(proprio_obs, dtype=torch.float32)
        action_th = torch.tensor(action, dtype=torch.float32)
        feedback_th = torch.tensor(feedback, dtype=torch.float32)
        self.camera_obs.append(camera_obs_th)
        self.proprio_obs.append(proprio_obs_th)
        self.action.append(action_th)
        self.feedback.append(feedback_th)
        self.reset_current_traj()
        return

    def reset_current_traj(self):
        self.current_camera_obs = []
        self.current_proprio_obs = []
        self.current_action = []
        self.current_feedback = []
        return

    def sample(self, batch_size):
        batch_size = min(batch_size, len(self))
        indeces = random.sample(range(len(self)), batch_size)
        batch = zip(*[self[i] for i in indeces])
        camera_batch = torch.stack(next(batch), dim=1)
        proprio_batch = torch.stack(next(batch), dim=1)
        action_batch = torch.stack(next(batch), dim=1)
        feedback_batch = torch.stack(next(batch), dim=1)
        return camera_batch, proprio_batch, action_batch, feedback_batch


def downsample_traj(traj, target_len):
    if len(traj) == target_len:
        return traj
    elif len(traj) < target_len:
        return traj + [traj[-1]] * (target_len - len(traj))
    else:
        indeces = np.linspace(start=0, stop=len(traj) - 1, num=target_len)
        indeces = np.round(indeces).astype(int)
        return np.array([traj[i] for i in indeces])


def loop_sleep(start_time):
    dt = 0.05
    sleep_time = dt - (time.time() - start_time)
    if sleep_time > 0.0:
        time.sleep(sleep_time)
    return


def euler_to_quaternion(euler_angle):
    roll, pitch, yaw = euler_angle
    qx = np.sin(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) - np.cos(
        roll / 2
    ) * np.sin(pitch / 2) * np.sin(yaw / 2)
    qy = np.cos(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2) + np.sin(
        roll / 2
    ) * np.cos(pitch / 2) * np.sin(yaw / 2)
    qz = np.cos(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2) - np.sin(
        roll / 2
    ) * np.sin(pitch / 2) * np.cos(yaw / 2)
    qw = np.cos(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) + np.sin(
        roll / 2
    ) * np.sin(pitch / 2) * np.sin(yaw / 2)
    return [qx, qy, qz, qw]


def set_seeds(seed=0):
    """Sets all seeds."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
