"""
MinimalShapedWrapper — 2-component dense reward aligned directly with scoring.

Components:
  1. Ball contact      — encourages touching the ball (exploration signal)
  2. Ball-to-goal delta — rewards ball moving toward opponent goal (causal proxy)
  3. Goal scored       — terminal ±1 (unchanged, logged as component for comparison)

Everything else removed. No proxy-of-proxy terms.
"""
from random import uniform as randfloat

import gym
import numpy as np
from ray.rllib import MultiAgentEnv
import soccer_twos


class RLLibWrapper(gym.core.Wrapper, MultiAgentEnv):
    pass


COMPONENT_NAMES = ["contact", "ball_goal_delta", "goal_scored"]


class MinimalShapedWrapper(gym.core.Wrapper):
    GOAL_POS      = np.array([18.0, 0.0])
    CONTACT_DIST  = 1.5

    def reset(self):
        obs = self.env.reset()
        self.prev_dist_to_goal = {}
        self._ep_components = {
            f"agent{i}/{n}": 0.0
            for i in [0, 1]
            for n in COMPONENT_NAMES
        }
        self._ep_components.update({f"team/{n}": 0.0 for n in COMPONENT_NAMES})
        return obs

    def _track(self, name, value, agent_idx):
        self._ep_components[f"agent{agent_idx}/{name}"] += value
        self._ep_components[f"team/{name}"] += value

    def _shape(self, reward, i, player_pos, ball_pos, ball_vel, track=False):
        # 1. Ball contact
        dist_to_ball = np.linalg.norm(player_pos - ball_pos)
        contact_r = 0.001 if dist_to_ball < self.CONTACT_DIST else 0.0
        reward += contact_r
        if track: self._track("contact", contact_r, i)

        # 2. Ball-to-goal delta — ball moving toward opponent goal
        dist_to_goal = np.linalg.norm(ball_pos - self.GOAL_POS)
        goal_delta_r = 0.0
        if i in self.prev_dist_to_goal:
            goal_delta_r = (self.prev_dist_to_goal[i] - dist_to_goal) * 0.025
        self.prev_dist_to_goal[i] = dist_to_goal
        reward += goal_delta_r
        if track: self._track("ball_goal_delta", goal_delta_r, i)

        return reward

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        if "player_info" in info:
            # single-agent mode
            player_pos = info["player_info"]["position"]
            ball_pos   = info["ball_info"]["position"]
            ball_vel   = info["ball_info"]["velocity"]
            reward = self._shape(reward, 0, player_pos, ball_pos, ball_vel, track=True)
        else:
            episode_done = isinstance(done, dict) and done.get("__all__", False)

            if episode_done:
                base_team_reward = reward.get(0, 0) + reward.get(1, 0)
                if base_team_reward > 0.5:
                    game_result = 1
                elif base_team_reward < -0.5:
                    game_result = -1
                else:
                    game_result = 0

                for i in [0, 1]:
                    if i in info:
                        info[i]["game_result"] = game_result

                # log terminal goal reward as a component (fires once per episode)
                # split equally between both agents
                goal_r = float(game_result)  # +1, -1, or 0
                for i in [0, 1]:
                    self._track("goal_scored", goal_r / 2.0, i)

                if 0 in info:
                    info[0]["reward_components"] = dict(self._ep_components)

            for i in [0, 1]:
                if i not in info or "player_info" not in info[i]:
                    continue
                player_pos = info[i]["player_info"]["position"]
                ball_pos   = info[i]["ball_info"]["position"]
                ball_vel   = info[i]["ball_info"]["velocity"]
                reward[i] = self._shape(
                    reward[i], i, player_pos, ball_pos, ball_vel, track=True
                )

        return obs, reward, done, info


def create_minimal_env(env_config: dict = {}):
    if hasattr(env_config, "worker_index"):
        env_config["worker_id"] = (
            env_config.worker_index * env_config.get("num_envs_per_worker", 1)
            + env_config.vector_index
        )
    env = soccer_twos.make(**env_config)
    env = MinimalShapedWrapper(env)
    if "multiagent" in env_config and not env_config["multiagent"]:
        return env
    return RLLibWrapper(env)


def sample_vec(range_dict):
    return [
        randfloat(range_dict["x"][0], range_dict["x"][1]),
        randfloat(range_dict["y"][0], range_dict["y"][1]),
    ]
