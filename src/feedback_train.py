import threading
import time
import wandb
import torch
from argparse import ArgumentParser
from custom_env import CustomEnv
from models import Policy
from human_feedback import human_feedback
from utils import TrajectoriesDataset  # noqa: F401
from utils import (
    device,
    KeyboardObserver,
    MetricsLogger,
    loop_sleep,
    set_seeds,
)


def train_step(policy, replay_memory, metrics_logger, config, stop_flag):
    while not stop_flag.isSet():
        batch = replay_memory.sample(config["batch_size"])
        camera_batch, proprio_batch, action_batch, feedback_batch = batch
        training_metrics = policy.update_params(
            camera_batch, proprio_batch, action_batch, feedback_batch
        )
        wandb.log(training_metrics)
        if not metrics_logger.empty():
            wandb.log(metrics_logger.pop())
    return


def run_env_simulation(env, policy, replay_memory, metrics_logger, config):
    keyboard_obs = KeyboardObserver()
    for episode in range(config["episodes"]):
        keyboard_obs.reset()
        done = False
        lstm_state = None
        camera_obs, proprio_obs = env.reset()
        while not done and metrics_logger.episode_steps < 300:  # i.e. 15 seconds
            start_time = time.time()
            action, lstm_state = policy.predict(camera_obs, proprio_obs, lstm_state)
            action, feedback = human_feedback(
                keyboard_obs, action, config["feedback_type"]
            )
            next_camera_obs, next_proprio_obs, reward, done = env.step(action)
            replay_memory.add(camera_obs, proprio_obs, action, [feedback])
            camera_obs, proprio_obs = next_camera_obs, next_proprio_obs
            metrics_logger.log_step(reward, feedback)
            loop_sleep(start_time)
        replay_memory.save_current_traj()
        metrics_logger.log_episode(episode)
    metrics_logger.log_session()
    return


def main(config):
    replay_memory = torch.load("data/" + config["task"] + "/demos_10.dat")
    env = CustomEnv(config)
    policy = Policy(config).to(device)
    model_path = "data/" + config["task"] + "/" + "pretraining_policy.pt"
    policy.load_state_dict(torch.load(model_path))
    policy.train()
    wandb.watch(policy, log_freq=100)
    metrics_logger = MetricsLogger()
    stop_flag = threading.Event()
    training_loop = threading.Thread(
        target=train_step,
        args=(policy, replay_memory, metrics_logger, config, stop_flag),
    )
    training_loop.start()
    time.sleep(5)
    run_env_simulation(env, policy, replay_memory, metrics_logger, config)
    time.sleep(60)
    stop_flag.set()
    file_name = "data/" + config["task"] + "/" + config["feedback_type"] + "_policy.pt"
    torch.save(policy.state_dict(), file_name)
    return


if __name__ == "__main__":
    set_seeds(1996)
    parser = ArgumentParser()
    parser.add_argument(
        "-f",
        "--feedback_type",
        dest="feedback_type",
        default="ceiling_full",
        help="options: evaluative, dagger, iwr, ceiling_full, ceiling_partial",
    )
    parser.add_argument(
        "-t",
        "--task",
        dest="task",
        default="CloseMicrowave",
        help="options: CloseMicrowave, PushButton, TakeLidOffSaucepan, UnplugCharger",
    )
    args = parser.parse_args()
    config_defaults = {
        "feedback_type": args.feedback_type,
        "task": args.task,
        "episodes": 100,
        "static_env": False,
        "headless_env": False,
        "proprio_dim": 8,
        "action_dim": 7,
        "visual_embedding_dim": 256,
        "learning_rate": 3e-4,
        "weight_decay": 3e-6,
        "batch_size": 16,
    }
    wandb.init(config=config_defaults, project="ceiling", mode="disabled")
    config = wandb.config  # important, in case the sweep gives different values
    main(config)
