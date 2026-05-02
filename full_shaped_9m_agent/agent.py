import os
from typing import Dict
import gym
import numpy as np
import torch
import torch.nn as nn
from soccer_twos import AgentInterface

WEIGHTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../full_shaped_weights_9M.pth")

class PolicyNet(nn.Module):
    def __init__(self, obs_size=336, hidden_size=256, branches=3, actions_per_branch=3):
        super().__init__()
        self.hidden0 = nn.Linear(obs_size, hidden_size)
        self.hidden1 = nn.Linear(hidden_size, hidden_size)
        self.logits  = nn.Linear(hidden_size, branches * actions_per_branch)

    def forward(self, x):
        x = torch.relu(self.hidden0(x))
        x = torch.relu(self.hidden1(x))
        return self.logits(x)

class FullShaped9MAgent(AgentInterface):
    def __init__(self, env: gym.Env):
        super().__init__()
        raw = torch.load(WEIGHTS_PATH)
        self.model = PolicyNet()
        self.model.load_state_dict({
            "hidden0.weight": torch.tensor(np.array(raw["_hidden_layers.0._model.0.weight"])),
            "hidden0.bias":   torch.tensor(np.array(raw["_hidden_layers.0._model.0.bias"])),
            "hidden1.weight": torch.tensor(np.array(raw["_hidden_layers.1._model.0.weight"])),
            "hidden1.bias":   torch.tensor(np.array(raw["_hidden_layers.1._model.0.bias"])),
            "logits.weight":  torch.tensor(np.array(raw["_logits._model.0.weight"])),
            "logits.bias":    torch.tensor(np.array(raw["_logits._model.0.bias"])),
        })
        self.model.eval()

    def act(self, observation: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
        actions = {}
        for player_id, obs in observation.items():
            with torch.no_grad():
                obs_tensor = torch.from_numpy(obs).float().unsqueeze(0)
                logits = self.model(obs_tensor).squeeze(0)
                action = [torch.argmax(logits[i*3:(i+1)*3]).item() for i in range(3)]
            actions[player_id] = np.array(action)
        return actions
