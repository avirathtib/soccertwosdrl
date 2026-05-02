"""
Evaluation script comparing agents at comparable checkpoints against random and baseline opponents.

Match format: N_MATCHES matches, each consisting of EPISODES_PER_MATCH episodes (points).
              The team with more goals after EPISODES_PER_MATCH episodes wins the match.

Agents evaluated:
  - Basic Shaped Rewards + Self-play  (checkpoints at ~5M and ~9M steps)
  - Full Shaped Rewards + Self-play   (checkpoints at ~5M and ~9M steps)

Opponents:
  - Random agent
  - CEIA Baseline agent
"""

import sys
import os

# log to both stdout and a file
_log_file = open("evaluation_results.log", "w", buffering=1)

def log(*args, **kwargs):
    msg = " ".join(str(a) for a in args)
    print(msg, flush=True)
    _log_file.write(msg + "\n")
    _log_file.flush()

import pickle
import numpy as np
import torch
import torch.nn as nn
import soccer_twos


# ── Checkpoint paths ────────────────────────────────────────────────────────
CHECKPOINTS = {
    "Basic Shaped @ 2M":  "ray_results/PPO_selfplay_basic_shaped/PPO_Soccer_28017_00000_0_2026-04-09_08-18-03/checkpoint_000100/checkpoint-100",
    "Basic Shaped @ 5M":  "ray_results/PPO_selfplay_basic_shaped/PPO_Soccer_28017_00000_0_2026-04-09_08-18-03/checkpoint_000250/checkpoint-250",
    "Basic Shaped @ 9M":  "ray_results/PPO_selfplay_basic_shaped/PPO_Soccer_28017_00000_0_2026-04-09_08-18-03/checkpoint_000450/checkpoint-450",
    "Full Shaped @ 2M":   "ray_results/PPO_selfplay_team/PPO_Soccer_51281_00000_0_2026-04-01_19-05-46/checkpoint_000100/checkpoint-100",
    "Full Shaped @ 5M":   "ray_results/PPO_selfplay_team/PPO_Soccer_51281_00000_0_2026-04-01_19-05-46/checkpoint_000250/checkpoint-250",
    "Full Shaped @ 9M":   "ray_results/PPO_selfplay_team/PPO_Soccer_51281_00000_0_2026-04-01_19-05-46/checkpoint_000450/checkpoint-450",
}

BASELINE_CHECKPOINT = (
    "ceia_baseline_agent/ray_results/PPO_selfplay_twos/"
    "PPO_Soccer_f475e_00000_0_2021-09-19_15-54-02/"
    "checkpoint_002449/checkpoint-2449"
)

N_MATCHES = 12           # matches per agent/opponent pair
EPISODES_PER_MATCH = 10  # episodes (goal opportunities) per match


# ── Model ────────────────────────────────────────────────────────────────────
class PolicyNet(nn.Module):
    def __init__(self, obs_size=336, hidden_size=256, branches=3, actions_per_branch=3):
        super().__init__()
        self.hidden0 = nn.Linear(obs_size, hidden_size)
        self.hidden1 = nn.Linear(hidden_size, hidden_size)
        self.logits = nn.Linear(hidden_size, branches * actions_per_branch)

    def forward(self, x):
        x = torch.relu(self.hidden0(x))
        x = torch.relu(self.hidden1(x))
        return self.logits(x)


def load_model(checkpoint_path, policy_key="current_team"):
    with open(checkpoint_path, "rb") as f:
        data = pickle.load(f)
    worker = pickle.loads(data["worker"])
    state = worker["state"][policy_key]
    # handle both {weights: {...}} and direct weight dict formats
    weights = state.get("weights", state)
    model = PolicyNet()
    model.load_state_dict({
        "hidden0.weight": torch.tensor(np.array(weights["_hidden_layers.0._model.0.weight"])),
        "hidden0.bias":   torch.tensor(np.array(weights["_hidden_layers.0._model.0.bias"])),
        "hidden1.weight": torch.tensor(np.array(weights["_hidden_layers.1._model.0.weight"])),
        "hidden1.bias":   torch.tensor(np.array(weights["_hidden_layers.1._model.0.bias"])),
        "logits.weight":  torch.tensor(np.array(weights["_logits._model.0.weight"])),
        "logits.bias":    torch.tensor(np.array(weights["_logits._model.0.bias"])),
    })
    model.eval()
    return model


def get_action(model, obs):
    with torch.no_grad():
        obs_t = torch.from_numpy(obs).float().unsqueeze(0)
        logits = model(obs_t).squeeze(0)
        return np.array([torch.argmax(logits[i*3:(i+1)*3]).item() for i in range(3)])


def random_action(_obs):
    return np.array([np.random.randint(0, 3) for _ in range(3)])


# ── Single episode ────────────────────────────────────────────────────────────
def run_episode(env, our_model, opponent_fn):
    """Run one episode. Returns (our_goals, opp_goals, steps)."""
    obs = env.reset()
    done = {"__all__": False}
    cumulative = 0.0
    steps = 0

    while not done["__all__"]:
        actions = {}
        for i in [0, 1]:
            actions[i] = get_action(our_model, obs[i]) if i in obs else np.zeros(3, dtype=int)
        for i in [2, 3]:
            actions[i] = opponent_fn(obs[i]) if i in obs else np.zeros(3, dtype=int)

        obs, reward, done, _ = env.step(actions)
        cumulative += reward.get(0, 0) + reward.get(1, 0)
        steps += 1

    # Determine goals from cumulative reward (+1 = we scored, -1 = they scored)
    if cumulative > 0.5:
        our_goals, opp_goals = 1, 0
    elif cumulative < -0.5:
        our_goals, opp_goals = 0, 1
    else:
        our_goals, opp_goals = 0, 0

    return our_goals, opp_goals, steps


# ── Match (N episodes = N points) ─────────────────────────────────────────────
def play_match(env, our_model, opponent_fn, match_num, episodes=EPISODES_PER_MATCH):
    """Play one match of `episodes` episodes. Returns (match_won, our_goals, opp_goals)."""
    our_total = 0
    opp_total = 0
    total_steps = 0

    for ep in range(episodes):
        our_g, opp_g, steps = run_episode(env, our_model, opponent_fn)
        our_total += our_g
        opp_total += opp_g
        total_steps += steps
        log(f"    Ep {ep+1:2d}/{episodes}: {'goal' if our_g else ('conceded' if opp_g else 'draw')}  "
            f"| running score {our_total}-{opp_total}  | steps: {steps}")

    match_won  = our_total > opp_total
    match_lost = our_total < opp_total
    result_str = "WIN " if match_won else ("LOSS" if match_lost else "DRAW")
    log(f"  Match {match_num:2d} result: {result_str}  {our_total}-{opp_total}  (total steps: {total_steps})")
    return match_won, not match_won and not match_lost, match_lost, our_total, opp_total


# ── Evaluation loop ──────────────────────────────────────────────────────────
def evaluate(our_model, opponent_fn, n_matches=N_MATCHES, label=""):
    match_wins = match_losses = match_draws = 0
    total_our_goals = total_opp_goals = 0

    env = soccer_twos.make(render=True)
    log(f"  Environment created. Starting {n_matches} matches x {EPISODES_PER_MATCH} episodes each...")

    for m in range(n_matches):
        log(f"\n  --- Match {m+1}/{n_matches} ---")
        won, draw, lost, our_g, opp_g = play_match(env, our_model, opponent_fn, m + 1)
        if won:
            match_wins += 1
        elif lost:
            match_losses += 1
        else:
            match_draws += 1
        total_our_goals += our_g
        total_opp_goals += opp_g

    env.close()

    log(f"\n  {label} summary: {match_wins}W {match_losses}L {match_draws}D  "
        f"| goals {total_our_goals}-{total_opp_goals} "
        f"| win rate {match_wins/n_matches*100:.1f}%")
    return match_wins, match_losses, match_draws, total_our_goals, total_opp_goals


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    log("Loading baseline model...")
    baseline_model = load_model(BASELINE_CHECKPOINT, policy_key="default")
    baseline_fn = lambda obs: get_action(baseline_model, obs)

    results = {}

    for agent_name, ckpt_path in CHECKPOINTS.items():
        log(f"\n{'='*60}")
        log(f"Agent: {agent_name}")
        log(f"Checkpoint: {ckpt_path}")
        log(f"{'='*60}")
        log(f"Loading model weights...")
        our_model = load_model(ckpt_path)
        log(f"Model loaded successfully.")

        log(f"\n  vs Random ({N_MATCHES} matches x {EPISODES_PER_MATCH} eps):")
        w, l, d, og, oppg = evaluate(our_model, random_action, label="vs Random")
        results[(agent_name, "vs Random")] = (w, l, d, og, oppg)

        log(f"\n  vs Baseline ({N_MATCHES} matches x {EPISODES_PER_MATCH} eps):")
        w, l, d, og, oppg = evaluate(our_model, baseline_fn, label="vs Baseline")
        results[(agent_name, "vs Baseline")] = (w, l, d, og, oppg)

    # ── Print summary table ───────────────────────────────────────────────────
    log(f"\n\n{'='*80}")
    log(f"{'RESULTS SUMMARY':^80}")
    log(f"{'='*80}")
    log(f"{'Agent':<25} {'Opponent':<15} {'MW':>4} {'ML':>4} {'MD':>4} {'Win%':>7} {'Goals':>10}")
    log(f"{'-'*80}")
    for (agent, opponent), (w, l, d, og, oppg) in results.items():
        win_pct = w / N_MATCHES * 100
        log(f"{agent:<25} {opponent:<15} {w:>4} {l:>4} {d:>4} {win_pct:>6.1f}%  {og:>4}-{oppg:<4}")
    log(f"{'='*80}")
