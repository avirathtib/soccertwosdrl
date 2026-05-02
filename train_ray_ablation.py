import copy
import math

import numpy as np
import ray
from ray import tune
from ray.rllib.agents.callbacks import DefaultCallbacks
import soccer_twos
from ray.rllib import MultiAgentEnv
import gym


SAVE_OPPONENT_EVERY = 5
MAX_HISTORY_SIZE = 20


class RLLibWrapper(gym.core.Wrapper, MultiAgentEnv):
    pass


def create_plain_env(env_config: dict = {}):
    """Plain env with NO reward shaping — sparse ±1 goal reward only."""
    if hasattr(env_config, "worker_index"):
        env_config["worker_id"] = (
            env_config.worker_index * env_config.get("num_envs_per_worker", 1)
            + env_config.vector_index
        )
    env = soccer_twos.make(**env_config)
    return RLLibWrapper(env)


def policy_mapping_fn(agent_id, *args, **kwargs):
    if agent_id in [0, 1]:
        return "current_team"
    else:
        return "opponent_team"


class SelfPlayCallback(DefaultCallbacks):
    def __init__(self):
        super().__init__()
        self.policy_history = []
        self.iteration = 0

    def on_episode_end(self, worker, base_env, policies, episode, **kwargs):
        info = episode.last_info_for(0)
        if info and "game_result" in info:
            result = info["game_result"]
            episode.custom_metrics["win"]  = 1.0 if result == 1  else 0.0
            episode.custom_metrics["loss"] = 1.0 if result == -1 else 0.0
            episode.custom_metrics["draw"] = 1.0 if result == 0  else 0.0

    def on_train_result(self, **info):
        result = info["result"]
        trainer = info["trainer"]
        self.iteration += 1

        if self.iteration % 10 == 0:
            current_reward = result.get("policy_reward_mean", {}).get("current_team", None)
            reward_str = f"{current_reward:.3f}" if current_reward is not None else "N/A"
            print(
                f"[Ablation] iter={self.iteration} "
                f"current_team_reward={reward_str} "
                f"history_size={len(self.policy_history)}"
            )

        if self.iteration % SAVE_OPPONENT_EVERY == 0:
            snapshot = copy.deepcopy(
                trainer.get_weights(["current_team"])["current_team"]
            )
            self.policy_history.append(snapshot)
            if len(self.policy_history) > MAX_HISTORY_SIZE:
                self.policy_history.pop(0)

        if len(self.policy_history) > 0:
            if len(self.policy_history) == 1:
                idx = 0
            else:
                idx = math.floor(
                    np.random.triangular(0, len(self.policy_history) - 1, len(self.policy_history) - 1)
                )
            new_opponent_weights = self.policy_history[idx]
        else:
            new_opponent_weights = copy.deepcopy(
                trainer.get_weights(["current_team"])["current_team"]
            )

        new_weights = {"opponent_team": new_opponent_weights}
        trainer.set_weights(new_weights)
        trainer.workers.foreach_worker(lambda w: w.set_weights(new_weights))


if __name__ == "__main__":
    NUM_ENVS_PER_WORKER = 3

    ray.init()

    tune.registry.register_env("SoccerPlain", create_plain_env)
    temp_env = create_plain_env()
    obs_space = temp_env.observation_space
    act_space = temp_env.action_space
    temp_env.close()

    analysis = tune.run(
        "PPO",
        name="PPO_ablation_no_shaping",
        config={
            "num_gpus": 0,
            "num_workers": 6,
            "num_envs_per_worker": NUM_ENVS_PER_WORKER,
            "log_level": "INFO",
            "framework": "torch",
            "callbacks": SelfPlayCallback,
            "multiagent": {
                "policies": {
                    "current_team": (None, obs_space, act_space, {}),
                    "opponent_team": (None, obs_space, act_space, {}),
                },
                "policy_mapping_fn": policy_mapping_fn,
                "policies_to_train": ["current_team"],
            },
            "env": "SoccerPlain",
            "env_config": {"num_envs_per_worker": NUM_ENVS_PER_WORKER},
            "model": {
                "vf_share_layers": False,
                "fcnet_hiddens": [256, 256],
                "fcnet_activation": "relu",
            },
            "lr": 1e-4,
            "entropy_coeff": 0.01,
            "lambda": 0.95,
            "clip_param": 0.15,
            "num_sgd_iter": 15,
            "sgd_minibatch_size": 512,
            "train_batch_size": 10000,
            "rollout_fragment_length": 500,
            "batch_mode": "truncate_episodes",
        },
        stop={"timesteps_total": 9000000},  # 9M steps
        checkpoint_freq=25,
        checkpoint_at_end=True,
        local_dir="./ray_results",
    )

    best_trial = analysis.get_best_trial("episode_reward_mean", mode="max")
    best_checkpoint = analysis.get_best_checkpoint(
        trial=best_trial, metric="episode_reward_mean", mode="max"
    )
    print(f"Best checkpoint: {best_checkpoint}")
    print("Done")
