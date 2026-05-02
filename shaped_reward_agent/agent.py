import os
from typing import Dict

import gym
import numpy as np
import torch
import torch.nn as nn
from gym_unity.envs import ActionFlattener
from soccer_twos import AgentInterface


class PPOPolicyNet(nn.Module):
    """Recreates the RLlib PPO FC network from the checkpoint weights."""
    def __init__(self, obs_size=336, hidden_size=512, action_size=27):
        super().__init__()
        self.hidden = nn.Linear(obs_size, hidden_size)
        self.logits = nn.Linear(hidden_size, action_size)

    def forward(self, x):
        x = torch.relu(self.hidden(x))
        return self.logits(x)


class ShapedRewardAgent(AgentInterface):
    """
    Agent trained with PPO using shaped rewards:
    - reward for moving closer to the ball
    - reward for moving ball closer to opponent goal
    Trained with team_vs_policy variation, single_player=True, flatten_branched=True.
    """

    def __init__(self, env: gym.Env):
        super().__init__()

        self.flattener = ActionFlattener(env.action_space.nvec)
        self.model = PPOPolicyNet(
            obs_size=env.observation_space.shape[0],
            hidden_size=512,
            action_size=self.flattener.action_space.n,
        )

        weights_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "policy_weights.pth")
        raw_weights = torch.load(weights_path)

        # map checkpoint keys to our model keys, converting numpy arrays to tensors
        state_dict = {
            "hidden.weight": torch.tensor(raw_weights["_hidden_layers.0._model.0.weight"]),
            "hidden.bias":   torch.tensor(raw_weights["_hidden_layers.0._model.0.bias"]),
            "logits.weight": torch.tensor(raw_weights["_logits._model.0.weight"]),
            "logits.bias":   torch.tensor(raw_weights["_logits._model.0.bias"]),
        }
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def act(self, observation: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
        actions = {}
        for player_id, obs in observation.items():
            with torch.no_grad():
                obs_tensor = torch.from_numpy(obs).float().unsqueeze(0)
                logits = self.model(obs_tensor)
                flat_action = torch.argmax(logits, dim=1).item()
            actions[player_id] = self.flattener.lookup_action(flat_action)
        return actions
