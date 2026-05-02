from random import uniform as randfloat

import gym
import numpy as np
from ray.rllib import MultiAgentEnv
import soccer_twos


class RLLibWrapper(gym.core.Wrapper, MultiAgentEnv):
    """
    A RLLib wrapper so our env can inherit from MultiAgentEnv.
    """

    pass

def _angle_between(v1, v2):
    """Returns angle in radians between two vectors."""
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)
    if denom < 1e-8:
        return 0.0
    cos_angle = np.clip(np.dot(v1, v2) / denom, -1.0, 1.0)
    return np.arccos(cos_angle)


class ShapedRewardsWrapper(gym.core.Wrapper):
    GOAL_POS     = np.array([18.0, 0.0])   # blue team attacks this goal
    OWN_GOAL_POS = np.array([-18.0, 0.0])  # blue team defends this goal
    CONTACT_DIST = 1.5                      # distance to count as ball contact
    MAX_FIELD_DIST = 30.0                   # approx max distance on the field
    MAX_STEPS = 5000                        # episode length (matches Unity env)

    # Soft role weights: agent 0 = forward-biased, agent 1 = defender-biased
    OFFENSIVE_WEIGHT = {0: 0.8, 1: 0.4}
    DEFENSIVE_WEIGHT = {0: 0.2, 1: 0.6}

    COMPONENT_NAMES = [
        "time_penalty", "proximity", "contact", "approach_delta",
        "ball_angle", "ball_goal_delta", "ball_behind", "ball_deep",
        "defensive_cover", "pressure_escape", "uncontested_clear",
        "assist", "own_goal", "anti_clustering",
    ]

    def _defensive_cover(self, player_pos, ball_pos):
        """Reward for standing on the line between ball and own goal (0-1)."""
        g2b = ball_pos - self.OWN_GOAL_POS
        g2p = player_pos - self.OWN_GOAL_POS
        dist_g2b = np.linalg.norm(g2b) + 1e-8
        t = np.clip(np.dot(g2p, g2b) / dist_g2b ** 2, 0.0, 1.0)
        perp = np.linalg.norm(player_pos - (self.OWN_GOAL_POS + t * g2b))
        return max(0.0, 1.0 - perp / 5.0)

    def reset(self):
        obs = self.env.reset()
        self.prev_dist_to_ball = {}
        self.prev_dist_to_goal = {}
        self.time_penalty = 0.0
        self.last_toucher = None
        self.last_passer = None
        # per-agent and team-level accumulators
        self._ep_components = {
            f"agent{i}/{n}": 0.0
            for i in [0, 1]
            for n in self.COMPONENT_NAMES
        }
        self._ep_components.update({f"team/{n}": 0.0 for n in self.COMPONENT_NAMES})
        return obs

    def _pressure_escape(self, player_pos, ball_pos, ball_vel, opp_positions):
        """
        Reward for hitting ball away from nearby opponents (escape pressure).
        Activates when at least one opponent is within PRESSURE_DIST of the ball.
        """
        if not opp_positions:
            return 0.0
        min_opp_dist = min(np.linalg.norm(op - ball_pos) for op in opp_positions)
        if min_opp_dist > 3.5:
            return 0.0  # no pressure
        ball_speed = np.linalg.norm(ball_vel)
        if ball_speed < 0.5:
            return 0.0
        # reward ball velocity directed away from nearest opponent
        nearest_opp = min(opp_positions, key=lambda op: np.linalg.norm(op - ball_pos))
        away_dir = ball_pos - nearest_opp
        away_norm = np.linalg.norm(away_dir) + 1e-8
        alignment = np.dot(ball_vel / ball_speed, away_dir / away_norm)
        return max(0.0, alignment) * 0.008

    def _uncontested_clear(self, ball_pos, ball_vel, opp_positions):
        """
        Reward for driving ball toward opponent goal when no opponent is near.
        Stronger than the regular ball-angle reward — activates on free ball.
        """
        if not opp_positions:
            return 0.0
        min_opp_dist = min(np.linalg.norm(op - ball_pos) for op in opp_positions)
        if min_opp_dist < 5.0:
            return 0.0  # opponents are close — don't give bonus
        ball_speed = np.linalg.norm(ball_vel)
        if ball_speed < 0.5:
            return 0.0
        # reward ball moving toward opponent goal
        to_goal = self.GOAL_POS - ball_pos
        to_goal_norm = np.linalg.norm(to_goal) + 1e-8
        alignment = np.dot(ball_vel / ball_speed, to_goal / to_goal_norm)
        return max(0.0, alignment) * ball_speed / 20.0 * 0.015

    def _track(self, name, value, agent_idx):
        """Add value to per-agent and team accumulators."""
        self._ep_components[f"agent{agent_idx}/{name}"] += value
        self._ep_components[f"team/{name}"] += value

    def _shape(self, reward, i, player_pos, ball_pos, ball_vel, opp_positions=None, track=False):
        off_w = self.OFFENSIVE_WEIGHT.get(i, 0.5)
        def_w = self.DEFENSIVE_WEIGHT.get(i, 0.5)

        # 1. Time penalty
        tp = -(self.time_penalty * 0.03)
        reward += tp
        if track: self._track("time_penalty", tp, i)

        # 2. Ball proximity
        dist_to_ball = np.linalg.norm(player_pos - ball_pos)
        proximity = max(0.0, (self.MAX_FIELD_DIST - dist_to_ball) / self.MAX_FIELD_DIST)
        prox_r = proximity * 0.001
        reward += prox_r
        if track: self._track("proximity", prox_r, i)

        # 3. Ball contact
        contact_r = 0.001 if dist_to_ball < self.CONTACT_DIST else 0.0
        reward += contact_r
        if track: self._track("contact", contact_r, i)

        # 4. Delta: agent moving closer to ball (offense-weighted)
        approach_r = 0.0
        if i in self.prev_dist_to_ball:
            approach_r = (self.prev_dist_to_ball[i] - dist_to_ball) * 0.01 * off_w
        reward += approach_r
        self.prev_dist_to_ball[i] = dist_to_ball
        if track: self._track("approach_delta", approach_r, i)

        # 5. Ball angle toward goal (offense-weighted)
        ball_speed = np.linalg.norm(ball_vel)
        angle_r = 0.0
        if ball_speed > 1e-3:
            angle = _angle_between(self.GOAL_POS - ball_pos, ball_vel)
            normalized_angle = (np.pi / 2 - angle) / (np.pi / 2)
            angle_r = normalized_angle * ball_speed / 40.0 * off_w
        reward += angle_r
        if track: self._track("ball_angle", angle_r, i)

        # 6. Delta: ball moving closer to goal (offense-weighted)
        dist_to_goal = np.linalg.norm(ball_pos - self.GOAL_POS)
        goal_delta_r = 0.0
        if i in self.prev_dist_to_goal:
            goal_delta_r = (self.prev_dist_to_goal[i] - dist_to_goal) * 0.025 * off_w
        reward += goal_delta_r
        self.prev_dist_to_goal[i] = dist_to_goal
        if track: self._track("ball_goal_delta", goal_delta_r, i)

        # 7. Ball behind penalty
        behind_r = -0.005 if player_pos[0] > ball_pos[0] else 0.0
        reward += behind_r
        if track: self._track("ball_behind", behind_r, i)

        # 8. Ball deep in own half penalty
        deep_r = -0.003 if ball_pos[0] < -12 else 0.0
        reward += deep_r
        if track: self._track("ball_deep", deep_r, i)

        # 9. Defensive cover
        ball_threatening = (ball_pos[0] < 0 or
                            np.dot(ball_vel, self.OWN_GOAL_POS - ball_pos) > 0)
        cover_r = 0.0
        if ball_threatening:
            cover_r = self._defensive_cover(player_pos, ball_pos) * 0.005 * def_w
        reward += cover_r
        if track: self._track("defensive_cover", cover_r, i)

        # 10. Pressure escape
        escape_r = 0.0
        if opp_positions:
            escape_r = self._pressure_escape(player_pos, ball_pos, ball_vel, opp_positions) * off_w
        reward += escape_r
        if track: self._track("pressure_escape", escape_r, i)

        # 11. Uncontested clear
        clear_r = 0.0
        if opp_positions:
            clear_r = self._uncontested_clear(ball_pos, ball_vel, opp_positions) * off_w
        reward += clear_r
        if track: self._track("uncontested_clear", clear_r, i)

        return reward

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.time_penalty = min(self.time_penalty + 1.0 / self.MAX_STEPS, 1.0)

        if "player_info" in info:
            # single-agent mode (TeamVsPolicyWrapper)
            player_pos = info["player_info"]["position"]
            ball_pos = info["ball_info"]["position"]
            ball_vel = info["ball_info"]["velocity"]
            reward = self._shape(reward, 0, player_pos, ball_pos, ball_vel, track=True)
        else:
            episode_done = isinstance(done, dict) and done.get("__all__", False)
            # track ball touches for assist/own-goal detection
            for i in [0, 1]:
                if i not in info or "player_info" not in info[i]:
                    continue
                dist = np.linalg.norm(
                    info[i]["player_info"]["position"] - info[i]["ball_info"]["position"]
                )
                if dist < self.CONTACT_DIST:
                    if self.last_toucher is not None and self.last_toucher != i:
                        self.last_passer = self.last_toucher
                    self.last_toucher = i

            if episode_done:
                base_team_reward = reward.get(0, 0) + reward.get(1, 0)
                if base_team_reward > 0.5:
                    game_result = 1
                    assist_r = 0.5
                    if self.last_passer is not None:
                        reward[self.last_passer] = reward.get(self.last_passer, 0) + assist_r
                        self._track("assist", assist_r, self.last_passer)
                elif base_team_reward < -0.5:
                    game_result = -1
                    og_r = -0.5
                    if self.last_toucher in [0, 1]:
                        reward[self.last_toucher] = reward.get(self.last_toucher, 0) + og_r
                        self._track("own_goal", og_r, self.last_toucher)
                else:
                    game_result = 0
                for i in [0, 1]:
                    if i in info:
                        info[i]["game_result"] = game_result
                # attach component totals for callback logging
                if 0 in info:
                    info[0]["reward_components"] = dict(self._ep_components)

            # collect opponent positions (agents 2 and 3 = orange team)
            opp_positions = [
                info[j]["player_info"]["position"]
                for j in [2, 3]
                if j in info and "player_info" in info[j]
            ]

            # anti-clustering: penalize both agents when too close to each other
            if (0 in info and "player_info" in info[0] and
                    1 in info and "player_info" in info[1]):
                p0 = info[0]["player_info"]["position"]
                p1 = info[1]["player_info"]["position"]
                teammate_dist = np.linalg.norm(p0 - p1)
                if teammate_dist < 5.0:
                    cluster_penalty = 0.003 * (1.0 - teammate_dist / 5.0)
                    reward[0] = reward.get(0, 0) - cluster_penalty
                    reward[1] = reward.get(1, 0) - cluster_penalty
                    self._ep_components["anti_clustering"] -= cluster_penalty  # track once

            # shape rewards for blue team (agents 0 and 1)
            for i in [0, 1]:
                if i not in info or "player_info" not in info[i]:
                    continue
                player_pos = info[i]["player_info"]["position"]
                ball_pos = info[i]["ball_info"]["position"]
                ball_vel = info[i]["ball_info"]["velocity"]
                reward[i] = self._shape(
                    reward[i], i, player_pos, ball_pos, ball_vel, opp_positions,
                    track=True  # track both agents; _track writes agent-specific + team keys
                )

        return obs, reward, done, info



def create_rllib_env(env_config: dict = {}):
    """
    Creates a RLLib environment and prepares it to be instantiated by Ray workers.
    Args:
        env_config: configuration for the environment.
            You may specify the following keys:
            - variation: one of soccer_twos.EnvType. Defaults to EnvType.multiagent_player.
            - opponent_policy: a Callable for your agent to train against. Defaults to a random policy.
    """
    if hasattr(env_config, "worker_index"):
        env_config["worker_id"] = (
            env_config.worker_index * env_config.get("num_envs_per_worker", 1)
            + env_config.vector_index
        )
    env = soccer_twos.make(**env_config)
    env = ShapedRewardsWrapper(env)
    # env = TransitionRecorderWrapper(env)
    if "multiagent" in env_config and not env_config["multiagent"]:
        # is multiagent by default, is only disabled if explicitly set to False
        return env
    return RLLibWrapper(env)


def sample_vec(range_dict):
    return [
        randfloat(range_dict["x"][0], range_dict["x"][1]),
        randfloat(range_dict["y"][0], range_dict["y"][1]),
    ]


def sample_val(range_tpl):
    return randfloat(range_tpl[0], range_tpl[1])


def sample_pos_vel(range_dict):
    _s = {}
    if "position" in range_dict:
        _s["position"] = sample_vec(range_dict["position"])
    if "velocity" in range_dict:
        _s["velocity"] = sample_vec(range_dict["velocity"])
    return _s


def sample_player(range_dict):
    _s = sample_pos_vel(range_dict)
    if "rotation_y" in range_dict:
        _s["rotation_y"] = sample_val(range_dict["rotation_y"])
    return _s
