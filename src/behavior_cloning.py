import wandb
import torch
from models import Policy
from utils import TrajectoriesDataset  # noqa: F401
from utils import device, set_seeds
from argparse import ArgumentParser


def train_step(policy, replay_memory, config):
    for _ in range(config["steps"]):
        batch = replay_memory.sample(config["batch_size"])
        camera_batch, proprio_batch, action_batch, feedback_batch = batch
        training_metrics = policy.update_params(
            camera_batch, proprio_batch, action_batch, feedback_batch
        )
        wandb.log(training_metrics)
    return


def main(config):
    if config["feedback_type"] == "pretraining":
        dataset_name = "/demos_10.dat"
        config["steps"] = 800
    elif config["feedback_type"] == "cloning_10":
        dataset_name = "/demos_10.dat"
        config["steps"] = 2000
    elif config["feedback_type"] == "cloning_100":
        dataset_name = "/demos_100.dat"
        config["steps"] = 2000
    else:
        raise NotImplementedError
    replay_memory = torch.load("data/" + config["task"] + dataset_name)
    policy = Policy(config).to(device)
    wandb.watch(policy, log_freq=100)
    train_step(policy, replay_memory, config)
    file_name = "data/" + config["task"] + "/" + config["feedback_type"] + "_policy.pt"
    torch.save(policy.state_dict(), file_name)
    return


if __name__ == "__main__":
    set_seeds(1)
    parser = ArgumentParser()
    parser.add_argument(
        "-f",
        "--feedback_type",
        dest="feedback_type",
        default="cloning_10",
        help="options: pretraining, cloning_10, cloning_100",
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
