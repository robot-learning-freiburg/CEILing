import torch
import wandb
import time
from models import Policy
from custom_env import CustomEnv
from utils import loop_sleep, set_seeds
from argparse import ArgumentParser

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


def run_simulation(env, policy, episodes):
    successes = 0
    time.sleep(10)
    for episode in range(episodes):
        steps = episode_reward = 0
        done = False
        camera_obs, proprio_obs = env.reset()
        lstm_state = None
        while not done and steps < 300:  # i.e. 15 seconds
            start_time = time.time()
            action, lstm_state = policy.predict(camera_obs, proprio_obs, lstm_state)
            next_camera_obs, next_proprio_obs, reward, done = env.step(action)
            camera_obs, proprio_obs = next_camera_obs, next_proprio_obs
            episode_reward += reward
            steps += 1
            loop_sleep(start_time)
        if episode_reward > 0:
            successes += 1
        wandb.log({"reward": episode_reward, "episode": episode})
    success_rate = successes / episodes
    wandb.run.summary["success_rate"] = success_rate
    return


def main(config):
    policy = Policy(config).to(device)
    model_path = "data/" + config["task"] + "/" + config["feedback_type"] + "_policy.pt"
    policy.load_state_dict(torch.load(model_path))
    policy.eval()
    env = CustomEnv(config)
    run_simulation(env, policy, config["episodes"])
    return


if __name__ == "__main__":
    set_seeds(1996)
    parser = ArgumentParser()
    parser.add_argument(
        "-f",
        "--feedback_type",
        dest="feedback_type",
        default="ceiling_01",
        help="options: cloning_10, cloning_100, evaluative, dagger, iwr, ceiling_full, ceiling_partial",
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
    wandb.init(config=config_defaults, project="ceiling_eval", mode="disabled")
    config = wandb.config  # important, in case the sweep gives different values
    main(config)
