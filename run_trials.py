"""
Evaluate all reward variants at 9M steps vs Ablation @ 9M.
Ablation (sparse ±1) is the baseline opponent — CEIA is too easy for all agents.
12 matches x 10 episodes, alternating sides.
"""
import pickle
import numpy as np
import torch
import torch.nn as nn
import soccer_twos

N_MATCHES          = 12
EPISODES_PER_MATCH = 10

ABLATION_PATH = "ablation_weights_9M.pth"

# All reward variants at 9M — same compute budget
AGENTS = {
    "Basic Shaped @ 9M":   "basic_shaped_weights_9M.pth",
    "Full Shaped @ 9M":    "ray_results/PPO_selfplay_team/PPO_Soccer_51281_00000_0_2026-04-01_19-05-46/checkpoint_000450/checkpoint-450",
    "Focused Shaped @ 9M": "focused_shaped_weights_9M.pth",
    "Minimal Shaped @ 9M": "minimal_shaped_weights_9M.pth",
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
        t      = torch.from_numpy(obs).float().unsqueeze(0)
        logits = model(t).squeeze(0)
        return np.array([torch.argmax(logits[i*3:(i+1)*3]).item() for i in range(3)])


def run_episode(env, blue_model, orange_model):
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
    if blue_r > 0.5:    return "blue"
    elif blue_r < -0.5: return "orange"
    return "draw"


def run_matchup(name, agent_model, ablation_model):
    env = soccer_twos.make(render=False)
    wins = losses = draws = 0
    agent_goals = abl_goals = 0

    for m in range(N_MATCHES):
        agent_is_blue = (m % 2 == 0)
        match_agent = match_abl = 0

        for _ in range(EPISODES_PER_MATCH):
            if agent_is_blue:
                result = run_episode(env, agent_model, ablation_model)
                if result == "blue":   match_agent += 1
                elif result == "orange": match_abl   += 1
            else:
                result = run_episode(env, ablation_model, agent_model)
                if result == "orange": match_agent += 1
                elif result == "blue":   match_abl   += 1

        won  = match_agent > match_abl
        lost = match_agent < match_abl
        tag  = "WIN " if won else ("LOSS" if lost else "DRAW")
        if won:    wins   += 1
        elif lost: losses += 1
        else:      draws  += 1
        agent_goals += match_agent
        abl_goals   += match_abl
        side = "blue  " if agent_is_blue else "orange"
        print(f"  Match {m+1:2d} [{side}]: {tag}  {name[:16]}={match_agent}  Ablation={match_abl}  "
              f"(running: {wins}W {losses}L {draws}D)")

    env.close()
    total    = agent_goals + abl_goals
    match_wr = wins / N_MATCHES * 100
    ep_wr    = agent_goals / total * 100 if total > 0 else 0
    print(f"\n  {name}: {wins}W {losses}L {draws}D  |  Match win%: {match_wr:.1f}%  "
          f"|  Goals: {agent_goals}-{abl_goals}  |  Episode win%: {ep_wr:.1f}%")
    return wins, losses, draws, agent_goals, abl_goals


if __name__ == "__main__":
    print("Loading Ablation @ 9M (opponent for all matchups)...")
    ablation = load_model(ABLATION_PATH)
    print("Loaded.\n")

    results = {}
    for name, path in AGENTS.items():
        print(f"\n{'='*60}")
        print(f"  {name}  vs  Ablation @ 9M")
        print(f"  ({N_MATCHES} matches x {EPISODES_PER_MATCH} episodes, alternating sides)")
        print(f"{'='*60}")
        model = load_model(path)
        w, l, d, ag, ablg = run_matchup(name, model, ablation)
        results[name] = (w, l, d, ag, ablg)

    print(f"\n\n{'='*70}")
    print(f"{'ALL REWARD VARIANTS @ 9M vs ABLATION @ 9M':^70}")
    print(f"{'='*70}")
    print(f"{'Agent':<24} {'W':>3} {'L':>3} {'D':>3} {'Match Win%':>11} {'Goals':>10} {'Ep Win%':>8}")
    print(f"{'-'*70}")
    for name, (w, l, d, ag, ablg) in results.items():
        total = ag + ablg
        ep_wr = ag / total * 100 if total > 0 else 0
        print(f"{name:<24} {w:>3} {l:>3} {d:>3} {w/N_MATCHES*100:>10.1f}%  {ag:>4}-{ablg:<4} {ep_wr:>7.1f}%")
    print(f"{'='*70}")
