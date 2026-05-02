"""
Head-to-head evaluation between all agent pairs at 9M steps.
Same format as evaluate_agents.py: 12 matches x 10 episodes each.
Alternates sides each match to control for blue/orange bias.

Matchups:
  Full Shaped vs Ablation
  Full Shaped vs Basic Shaped
  Ablation    vs Basic Shaped
"""
import numpy as np
import torch
import torch.nn as nn
import pickle
import soccer_twos

N_MATCHES          = 12
EPISODES_PER_MATCH = 10

WEIGHTS = {
    "Full Shaped @ 9M":     "ray_results/PPO_selfplay_team/PPO_Soccer_51281_00000_0_2026-04-01_19-05-46/checkpoint_000450/checkpoint-450",
    "Basic Shaped @ 9M":    "basic_shaped_weights_9M.pth",
    "Ablation @ 9M":        "ablation_weights_9M.pth",
    "Focused Shaped @ 9M":  "focused_shaped_weights_9M.pth",
    "Minimal Shaped @ 9M":  "minimal_shaped_weights_9M.pth",
    "Minimal Shaped @ 13M": "minimal_weights_13M.pth",
    "Minimal Shaped @ 19M": "minimal_weights_19M.pth",
}


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


def load_model(path, policy_key="current_team"):
    """Load from either a .pth weights file or a Ray checkpoint."""
    if path.endswith(".pth"):
        raw = torch.load(path)
    else:
        with open(path, "rb") as f:
            data = pickle.load(f)
        worker = pickle.loads(data["worker"])
        state  = worker["state"][policy_key]
        raw    = state.get("weights", state)

    model = PolicyNet()
    model.load_state_dict({
        "hidden0.weight": torch.tensor(np.array(raw["_hidden_layers.0._model.0.weight"])),
        "hidden0.bias":   torch.tensor(np.array(raw["_hidden_layers.0._model.0.bias"])),
        "hidden1.weight": torch.tensor(np.array(raw["_hidden_layers.1._model.0.weight"])),
        "hidden1.bias":   torch.tensor(np.array(raw["_hidden_layers.1._model.0.bias"])),
        "logits.weight":  torch.tensor(np.array(raw["_logits._model.0.weight"])),
        "logits.bias":    torch.tensor(np.array(raw["_logits._model.0.bias"])),
    })
    model.eval()
    return model


def get_action(model, obs):
    with torch.no_grad():
        t = torch.from_numpy(obs).float().unsqueeze(0)
        logits = model(t).squeeze(0)
        return np.array([torch.argmax(logits[i*3:(i+1)*3]).item() for i in range(3)])


def run_episode(env, blue_model, orange_model):
    """One episode. Blue = agents 0,1. Orange = agents 2,3."""
    obs  = env.reset()
    done = {"__all__": False}
    while not done["__all__"]:
        actions = {
            0: get_action(blue_model,   obs[0]) if 0 in obs else np.zeros(3, dtype=int),
            1: get_action(blue_model,   obs[1]) if 1 in obs else np.zeros(3, dtype=int),
            2: get_action(orange_model, obs[2]) if 2 in obs else np.zeros(3, dtype=int),
            3: get_action(orange_model, obs[3]) if 3 in obs else np.zeros(3, dtype=int),
        }
        obs, reward, done, _ = env.step(actions)
    blue_r = reward.get(0, 0) + reward.get(1, 0)
    if blue_r > 0.5:   return "blue"
    elif blue_r < -0.5: return "orange"
    return "draw"


def play_match(env, model_a, model_b, match_num, a_is_blue):
    """One match of EPISODES_PER_MATCH episodes, alternating perspective."""
    a_goals = b_goals = 0
    for _ in range(EPISODES_PER_MATCH):
        if a_is_blue:
            result = run_episode(env, model_a, model_b)
            if result == "blue":   a_goals += 1
            elif result == "orange": b_goals += 1
        else:
            result = run_episode(env, model_b, model_a)
            if result == "orange": a_goals += 1
            elif result == "blue":   b_goals += 1

    won  = a_goals > b_goals
    lost = a_goals < b_goals
    tag  = "WIN " if won else ("LOSS" if lost else "DRAW")
    side = "blue  " if a_is_blue else "orange"
    print(f"  Match {match_num:2d} [{side}]: {tag}  {a_goals}-{b_goals}")
    return won, lost, a_goals, b_goals


def run_matchup(name_a, model_a, name_b, model_b):
    print(f"\n{'='*60}")
    print(f"  {name_a}  vs  {name_b}")
    print(f"  ({N_MATCHES} matches x {EPISODES_PER_MATCH} episodes, alternating sides)")
    print(f"{'='*60}")

    env = soccer_twos.make(render=False)
    wins = losses = draws = 0
    total_a = total_b = 0

    for m in range(N_MATCHES):
        a_is_blue = (m % 2 == 0)
        won, lost, ag, bg = play_match(env, model_a, model_b, m + 1, a_is_blue)
        if won:    wins   += 1
        elif lost: losses += 1
        else:      draws  += 1
        total_a += ag
        total_b += bg

    env.close()

    total_ep = total_a + total_b
    ep_wr = total_a / total_ep * 100 if total_ep > 0 else 0.0
    print(f"\n  {name_a}: {wins}W {losses}L {draws}D")
    print(f"  Match win%: {wins/N_MATCHES*100:.1f}%  |  Goals: {total_a}-{total_b}  |  Episode win%: {ep_wr:.1f}%")
    return wins, losses, draws, total_a, total_b


if __name__ == "__main__":
    print("Loading models...")
    models = {name: load_model(path) for name, path in WEIGHTS.items()}
    print("All models loaded.\n")

    results = {}

    pairs = [
        ("Minimal Shaped @ 19M", "Ablation @ 9M"),
        ("Minimal Shaped @ 19M", "Minimal Shaped @ 13M"),
        ("Minimal Shaped @ 19M", "Minimal Shaped @ 9M"),
        ("Minimal Shaped @ 19M", "Full Shaped @ 9M"),
    ]

    for name_a, name_b in pairs:
        w, l, d, ga, gb = run_matchup(name_a, models[name_a], name_b, models[name_b])
        results[(name_a, name_b)] = (w, l, d, ga, gb)

    print(f"\n\n{'='*70}")
    print(f"{'HEAD-TO-HEAD SUMMARY':^70}")
    print(f"{'='*70}")
    print(f"{'Agent A':<24} {'vs':<4} {'Agent B':<24} {'W':>3} {'L':>3} {'D':>3} {'Win%':>6}  Goals")
    print(f"{'-'*70}")
    for (na, nb), (w, l, d, ga, gb) in results.items():
        print(f"{na:<24} {'vs':<4} {nb:<24} {w:>3} {l:>3} {d:>3} {w/N_MATCHES*100:>5.1f}%  {ga}-{gb}")
    print(f"{'='*70}")
