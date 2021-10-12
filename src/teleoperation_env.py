import os
import time
import torch
import numpy as np
from argparse import ArgumentParser
from human_feedback import correct_action
from utils import KeyboardObserver, TrajectoriesDataset, loop_sleep
from custom_env import CustomEnv


def main(config):
    save_path = "data/" + config["task"] + "/"
    assert os.path.exists(save_path)
    env = CustomEnv(config)
    keyboard_obs = KeyboardObserver()
    replay_memory = TrajectoriesDataset(config["sequence_len"])
    camera_obs, proprio_obs = env.reset()
    gripper_open = 0.9
    time.sleep(5)
    print("Go!")
    episodes_count = 0
    while episodes_count < config["episodes"]:
        start_time = time.time()
        action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, gripper_open])
        if keyboard_obs.has_joints_cor() or keyboard_obs.has_gripper_update():
            action = correct_action(keyboard_obs, action)
            gripper_open = action[-1]
        next_camera_obs, next_proprio_obs, reward, done = env.step(action)
        replay_memory.add(camera_obs, proprio_obs, action, [1])
        camera_obs, proprio_obs = next_camera_obs, next_proprio_obs
        if keyboard_obs.reset_button:
            replay_memory.reset_current_traj()
            camera_obs, proprio_obs = env.reset()
            gripper_open = 0.9
            keyboard_obs.reset()
        elif done:
            replay_memory.save_current_traj()
            camera_obs, proprio_obs = env.reset()
            gripper_open = 0.9
            episodes_count += 1
            keyboard_obs.reset()
            done = False
        else:
            loop_sleep(start_time)
    file_name = "demos_" + str(config["episodes"]) + ".dat"
    if config["save_demos"]:
        torch.save(replay_memory, save_path + file_name)
    return


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "-t",
        "--task",
        dest="task",
        default="CloseMicrowave",
        help="options: CloseMicrowave, PushButton, TakeLidOffSaucepan, UnplugCharger",
    )
    args = parser.parse_args()
    config = {
        "task": args.task,
        "static_env": False,
        "headless_env": False,
        "save_demos": True,
        "episodes": 10,
        "sequence_len": 150,
    }
    main(config)
