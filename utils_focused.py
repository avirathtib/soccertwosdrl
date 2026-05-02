"""
FocusedShapedWrapper — leaner reward shaping for SoccerTwos.

Keeps only the 5 core directional components aligned with winning:
  1. Ball proximity       — general pressure toward ball
  2. Ball contact         — reward touching the ball
  3. Approach delta       — reward moving closer to ball
  4. Ball-angle to goal   — reward ball velocity directed at opponent goal
  5. Ball-to-goal delta   — reward ball moving toward opponent goal

Each component is tracked per episode and logged as a custom metric
(reward_component/<name>) so individual curves are visible in TensorBoard.
"""
from collections import defaultdict
from random import uniform as randfloat

import gym
import numpy as np
from ray.rllib import MultiAgentEnv
import soccer_twos


class RLLibWrapper(gym.core.Wrapper, MultiAgentEnv):
    pass


def _angle_between(v1, v2):
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)
    if denom < 1e-8:
        return 0.0
    cos_angle = np.clip(np.dot(v1, v2) / denom, -1.0, 1.0)
    return np.arccos(cos_angle)


class FocusedShapedWrapper(gym.core.Wrapper):
    GOAL_POS       = np.array([18.0, 0.0])
    CONTACT_DIST   = 1.5
    MAX_FIELD_DIST = 30.0

    COMPONENT_NAMES = [
        "proximity",
        "contact",
        "approach_delta",
        "ball_angle",
        "ball_goal_delta",
    ]

    def reset(self):
        obs = self.env.reset()
        self.prev_dist_to_ball = {}
        self.prev_dist_to_goal = {}
        self._ep_components = {
            f"agent{i}/{n}": 0.0
            for i in [0, 1]
            for n in self.COMPONENT_NAMES
        }
        self._ep_components.update({f"team/{n}": 0.0 for n in self.COMPONENT_NAMES})
        return obs

    def _track(self, name, value, agent_idx):
        self._ep_components[f"agent{agent_idx}/{name}"] += value
        self._ep_components[f"team/{name}"] += value

    def _shape(self, reward, i, player_pos, ball_pos, ball_vel, track=False):
        dist_to_ball = np.linalg.norm(player_pos - ball_pos)

        # 1. Ball proximity
        proximity = max(0.0, (self.MAX_FIELD_DIST - dist_to_ball) / self.MAX_FIELD_DIST)
        prox_r = proximity * 0.001
        reward += prox_r
        if track: self._track("proximity", prox_r, i)

        # 2. Ball contact
        contact_r = 0.001 if dist_to_ball < self.CONTACT_DIST else 0.0
        reward += contact_r
        if track: self._track("contact", contact_r, i)

        # 3. Approach delta
        approach_r = 0.0
        if i in self.prev_dist_to_ball:
            approach_r = (self.prev_dist_to_ball[i] - dist_to_ball) * 0.01
        reward += approach_r
        self.prev_dist_to_ball[i] = dist_to_ball
        if track: self._track("approach_delta", approach_r, i)

        # 4. Ball angle toward opponent goal
        ball_speed = np.linalg.norm(ball_vel)
        angle_r = 0.0
        if ball_speed > 1e-3:
            angle = _angle_between(self.GOAL_POS - ball_pos, ball_vel)
            normalized_angle = (np.pi / 2 - angle) / (np.pi / 2)
            angle_r = normalized_angle * ball_speed / 40.0
        reward += angle_r
        if track: self._track("ball_angle", angle_r, i)

        # 5. Ball-to-goal delta
        dist_to_goal = np.linalg.norm(ball_pos - self.GOAL_POS)
        goal_delta_r = 0.0
        if i in self.prev_dist_to_goal:
            goal_delta_r = (self.prev_dist_to_goal[i] - dist_to_goal) * 0.025
        reward += goal_delta_r
        self.prev_dist_to_goal[i] = dist_to_goal
        if track: self._track("ball_goal_delta", goal_delta_r, i)

        return reward

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        if "player_info" in info:
            player_pos = info["player_info"]["position"]
            ball_pos   = info["ball_info"]["position"]
            ball_vel   = info["ball_info"]["velocity"]
            reward = self._shape(reward, 0, player_pos, ball_pos, ball_vel, track=True)
        else:
            episode_done = isinstance(done, dict) and done.get("__all__", False)
            if episode_done:
                base_team_reward = reward.get(0, 0) + reward.get(1, 0)
                game_result = (1 if base_team_reward > 0.5
                               else -1 if base_team_reward < -0.5
                               else 0)
                for i in [0, 1]:
                    if i in info:
                        info[i]["game_result"] = game_result
                # attach component totals to agent 0's info for callback logging
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


def create_focused_env(env_config: dict = {}):
    if hasattr(env_config, "worker_index"):
        env_config["worker_id"] = (
            env_config.worker_index * env_config.get("num_envs_per_worker", 1)
            + env_config.vector_index
        )
    env = soccer_twos.make(**env_config)
    env = FocusedShapedWrapper(env)
    if "multiagent" in env_config and not env_config["multiagent"]:
        return env
    return RLLibWrapper(env)


def sample_vec(range_dict):
    return [
        randfloat(range_dict["x"][0], range_dict["x"][1]),
        randfloat(range_dict["y"][0], range_dict["y"][1]),
    ]


def sample_val(range_tpl):
    return randfloat(range_tpl[0], range_tpl[1])
