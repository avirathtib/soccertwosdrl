"""
Minimal Shaped — PPO self-play with 2 causally-aligned reward components.
  1. Ball contact (0.001 per step within 1.5 units)
  2. Ball-to-goal delta (0.025 * progress per step)
  3. Terminal ±1 unchanged

Trains to 9M steps from scratch for fair comparison.
"""
import copy
import math
import pickle

import numpy as np
import ray
from ray import tune
from ray.rllib.agents.callbacks import DefaultCallbacks
from utils_minimal import create_minimal_env


NUM_ENVS_PER_WORKER = 3
SAVE_OPPONENT_EVERY = 5
MAX_HISTORY_SIZE    = 20
BASELINE_MIX_PROB   = 0.0   # pure self-play; set to 0.4 to mix in CEIA baseline
RESTORE_CHECKPOINT  = (
    "ray_results/PPO_minimal_shaped/"
    "PPO_SoccerMinimal_ed598_00000_0_2026-04-26_10-40-53/"
    "checkpoint_000650/checkpoint-650"
)


def policy_mapping_fn(agent_id, *args, **kwargs):
    return "current_team" if agent_id in [0, 1] else "opponent_team"


class PrioritizedSelfPlayCallback(DefaultCallbacks):
    def __init__(self):
        super().__init__()
        self.policy_history = []
        self.baseline_weights = None
        self.iteration = 0

    def _load_baseline(self):
        path = (
            "/Users/apple/Desktop/gt/cs8803drl/soccer-twos-starter/"
            "ceia_baseline_agent/ray_results/PPO_selfplay_twos/"
            "PPO_Soccer_f475e_00000_0_2021-09-19_15-54-02/"
            "checkpoint_002449/checkpoint-2449"
        )
        with open(path, "rb") as f:
            data = pickle.load(f)
        worker = pickle.loads(data["worker"])
        weights = worker["state"]["default"]
        vb_shapes = {
            "_value_branch_separate.0._model.0.weight": (256, 336),
            "_value_branch_separate.0._model.0.bias":   (256,),
            "_value_branch_separate.1._model.0.weight": (256, 256),
            "_value_branch_separate.1._model.0.bias":   (256,),
        }
        for key, shape in vb_shapes.items():
            if key not in weights:
                weights[key] = np.zeros(shape, dtype=np.float32)
        weights.pop("_optimizer_variables", None)
        return weights

    def on_episode_end(self, worker, base_env, policies, episode, **kwargs):
        info = episode.last_info_for(0)
        if info and "game_result" in info:
            result = info["game_result"]
            episode.custom_metrics["win"]  = 1.0 if result == 1  else 0.0
            episode.custom_metrics["loss"] = 1.0 if result == -1 else 0.0
            episode.custom_metrics["draw"] = 1.0 if result == 0  else 0.0
        if info and "reward_components" in info:
            for name, value in info["reward_components"].items():
                episode.custom_metrics[f"reward_component/{name}"] = value

    def on_train_result(self, **info):
        result  = info["result"]
        trainer = info["trainer"]
        self.iteration += 1

        if self.iteration % 10 == 0:
            cr = result.get("policy_reward_mean", {}).get("current_team", None)
            cr_str = f"{cr:.3f}" if cr is not None else "N/A"
            print(
                f"[MinimalSP] iter={self.iteration} "
                f"reward={cr_str} "
                f"global={result['episode_reward_mean']:.3f} "
                f"history={len(self.policy_history)}"
            )

        if self.iteration % SAVE_OPPONENT_EVERY == 0:
            snapshot = copy.deepcopy(
                trainer.get_weights(["current_team"])["current_team"]
            )
            self.policy_history.append(snapshot)
            if len(self.policy_history) > MAX_HISTORY_SIZE:
                self.policy_history.pop(0)

        if self.baseline_weights is None and BASELINE_MIX_PROB > 0:
            try:
                self.baseline_weights = self._load_baseline()
                print("[MinimalSP] Loaded CEIA baseline weights")
            except Exception as e:
                print(f"[MinimalSP] Could not load baseline: {e}")

        if (self.baseline_weights is not None
                and np.random.random() < BASELINE_MIX_PROB):
            new_opp = self.baseline_weights
        elif len(self.policy_history) > 0:
            idx = 0 if len(self.policy_history) == 1 else math.floor(
                np.random.triangular(0, len(self.policy_history) - 1,
                                     len(self.policy_history) - 1)
            )
            new_opp = self.policy_history[idx]
        else:
            new_opp = copy.deepcopy(
                trainer.get_weights(["current_team"])["current_team"]
            )

        new_weights = {"opponent_team": new_opp}
        trainer.set_weights(new_weights)
        trainer.workers.foreach_worker(lambda w: w.set_weights(new_weights))


if __name__ == "__main__":
    ray.init()

    tune.registry.register_env("SoccerMinimal", create_minimal_env)
    temp_env = create_minimal_env()
    obs_space = temp_env.observation_space
    act_space  = temp_env.action_space
    temp_env.close()

    kwargs = dict(
        name="PPO_minimal_shaped",
        config={
            "num_gpus": 0,
            "num_workers": 6,
            "num_envs_per_worker": NUM_ENVS_PER_WORKER,
            "log_level": "INFO",
            "framework": "torch",
            "callbacks": PrioritizedSelfPlayCallback,
            "multiagent": {
                "policies": {
                    "current_team": (None, obs_space, act_space, {}),
                    "opponent_team": (None, obs_space, act_space, {}),
                },
                "policy_mapping_fn": policy_mapping_fn,
                "policies_to_train": ["current_team"],
            },
            "env": "SoccerMinimal",
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
        stop={"timesteps_total": 20_000_000},
        checkpoint_freq=25,
        checkpoint_at_end=True,
        local_dir="./ray_results",
    )

    if RESTORE_CHECKPOINT:
        kwargs["restore"] = RESTORE_CHECKPOINT

    analysis = tune.run("PPO", **kwargs)
    best_trial = analysis.get_best_trial("episode_reward_mean", mode="max")
    best_ckpt  = analysis.get_best_checkpoint(
        trial=best_trial, metric="episode_reward_mean", mode="max"
    )
    print(f"Best checkpoint: {best_ckpt}")
    print("Done.")
