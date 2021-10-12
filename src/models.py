import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from utils import device


class Policy(nn.Module):
    def __init__(self, config):
        super(Policy, self).__init__()
        lstm_dim = config["visual_embedding_dim"] + config["proprio_dim"]
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=2, kernel_size=3, padding=1, stride=2
        )
        self.conv2 = nn.Conv2d(
            in_channels=2, out_channels=1, kernel_size=3, padding=1, stride=2
        )
        self.conv3 = nn.Conv2d(
            in_channels=1, out_channels=1, kernel_size=3, padding=1, stride=2
        )
        self.lstm = nn.LSTM(lstm_dim, lstm_dim)  # , batch_first=True)
        self.linear_out = nn.Linear(lstm_dim, config["action_dim"])
        self.optimizer = torch.optim.Adam(
            self.parameters(),
            lr=config["learning_rate"],
            weight_decay=config["weight_decay"],
        )
        self.std = 0.1 * torch.ones(config["action_dim"], dtype=torch.float32)
        self.std = self.std.to(device)
        return

    def forward_step(self, camera_obs, proprio_obs, lstm_state):
        vis_encoding = F.elu(self.conv1(camera_obs))
        vis_encoding = F.elu(self.conv2(vis_encoding))
        vis_encoding = F.elu(self.conv3(vis_encoding))
        vis_encoding = torch.flatten(vis_encoding, start_dim=1)
        low_dim_input = torch.cat((vis_encoding, proprio_obs), dim=-1).unsqueeze(0)
        lstm_out, (h, c) = self.lstm(low_dim_input, lstm_state)
        lstm_state = (h, c)
        out = torch.tanh(self.linear_out(lstm_out))
        return out, lstm_state

    def forward(self, camera_obs_traj, proprio_obs_traj, action_traj, feedback_traj):
        losses = []
        lstm_state = None
        for idx in range(len(proprio_obs_traj)):
            mu, lstm_state = self.forward_step(
                camera_obs_traj[idx], proprio_obs_traj[idx], lstm_state
            )
            distribution = Normal(mu, self.std)
            log_prob = distribution.log_prob(action_traj[idx])
            loss = -log_prob * feedback_traj[idx]
            losses.append(loss)
        total_loss = torch.cat(losses).mean()
        return total_loss

    def update_params(
        self, camera_obs_traj, proprio_obs_traj, action_traj, feedback_traj
    ):
        camera_obs = camera_obs_traj.to(device)
        proprio_obs = proprio_obs_traj.to(device)
        action = action_traj.to(device)
        feedback = feedback_traj.to(device)
        self.optimizer.zero_grad()
        loss = self.forward(camera_obs, proprio_obs, action, feedback)
        loss.backward()
        self.optimizer.step()
        training_metrics = {"loss": loss}
        return training_metrics

    def predict(self, camera_obs, proprio_obs, lstm_state):
        camera_obs_th = torch.tensor(camera_obs, dtype=torch.float32).unsqueeze(0)
        proprio_obs_th = torch.tensor(proprio_obs, dtype=torch.float32).unsqueeze(0)
        camera_obs_th = camera_obs_th.to(device)
        proprio_obs_th = proprio_obs_th.to(device)
        with torch.no_grad():
            action_th, lstm_state = self.forward_step(
                camera_obs_th, proprio_obs_th, lstm_state
            )
            action = action_th.detach().cpu().squeeze(0).squeeze(0).numpy()
            action[-1] = binary_gripper(action[-1])
        return action, lstm_state


def binary_gripper(gripper_action):
    if gripper_action >= 0.0:
        gripper_action = 0.9
    elif gripper_action < 0.0:
        gripper_action = -0.9
    return gripper_action
