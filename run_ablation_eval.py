"""
Evaluate ablation agent (no reward shaping) at 2M, 5M, 9M checkpoints
vs Random and CEIA Baseline.
Same format as evaluate_agents.py: 12 matches x 10 episodes each.
"""
import numpy as np
import torch
import torch.nn as nn
import pickle
import soccer_twos

N_MATCHES          = 12
EPISODES_PER_MATCH = 10

BASELINE_CHECKPOINT = (
    "ceia_baseline_agent/ray_results/PPO_selfplay_twos/"
    "PPO_Soccer_f475e_00000_0_2021-09-19_15-54-02/"
    "checkpoint_002449/checkpoint-2449"
)

ABLATION_CHECKPOINTS = {
    "Ablation @ 2M": "ablation_weights_2M.pth",
    "Ablation @ 5M": "ablation_weights_5M.pth",
    "Ablation @ 9M": "ablation_weights_9M.pth",
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


def load_ablation_model(weights_path):
    raw = torch.load(weights_path)
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


def load_baseline_model():
    with open(BASELINE_CHECKPOINT, "rb") as f:
        data = pickle.load(f)
    worker = pickle.loads(data["worker"])
    state  = worker["state"]["default"]
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
        t = torch.from_numpy(obs).float().unsqueeze(0)
        logits = model(t).squeeze(0)
        return np.array([torch.argmax(logits[i*3:(i+1)*3]).item() for i in range(3)])


def random_action(_obs):
    return np.array([np.random.randint(3) for _ in range(3)])


def run_episode(env, our_model, opponent_fn):
    obs  = env.reset()
    done = {"__all__": False}
    while not done["__all__"]:
        actions = {}
        for i in [0, 1]:
            actions[i] = get_action(our_model, obs[i]) if i in obs else np.zeros(3, dtype=int)
        for i in [2, 3]:
            actions[i] = opponent_fn(obs[i]) if i in obs else np.zeros(3, dtype=int)
        obs, reward, done, _ = env.step(actions)
    cumulative = reward.get(0, 0) + reward.get(1, 0)
    if cumulative > 0.5:   return 1, 0
    elif cumulative < -0.5: return 0, 1
    return 0, 0


def play_match(env, our_model, opponent_fn, episodes=EPISODES_PER_MATCH):
    our_total = opp_total = 0
    for _ in range(episodes):
        og, opg = run_episode(env, our_model, opponent_fn)
        our_total += og
        opp_total += opg
    won  = our_total > opp_total
    lost = our_total < opp_total
    return won, not won and not lost, lost, our_total, opp_total


def evaluate(our_model, opponent_fn, label=""):
    env = soccer_twos.make(render=False)
    mw = ml = md = 0
    total_our = total_opp = 0
    for m in range(N_MATCHES):
        won, draw, lost, og, opg = play_match(env, our_model, opponent_fn)
        if won:  mw += 1
        elif lost: ml += 1
        else: md += 1
        total_our += og
        total_opp += opg
    env.close()
    print(f"  {label}: {mw}W {ml}L {md}D | win% {mw/N_MATCHES*100:.1f}% | goals {total_our}-{total_opp}")
    return mw, ml, md, total_our, total_opp


if __name__ == "__main__":
    print("Loading baseline model...")
    baseline_model = load_baseline_model()
    baseline_fn    = lambda obs: get_action(baseline_model, obs)

    results = {}

    for agent_name, weights_path in ABLATION_CHECKPOINTS.items():
        print(f"\n{'='*55}")
        print(f"  {agent_name}")
        print(f"{'='*55}")
        model = load_ablation_model(weights_path)

        w, l, d, og, opg = evaluate(model, random_action, "vs Random  ")
        results[(agent_name, "vs Random")] = (w, l, d, og, opg)

        w, l, d, og, opg = evaluate(model, baseline_fn, "vs Baseline")
        results[(agent_name, "vs Baseline")] = (w, l, d, og, opg)

    print(f"\n\n{'='*70}")
    print(f"{'ABLATION EVALUATION SUMMARY':^70}")
    print(f"{'='*70}")
    print(f"{'Agent':<22} {'Opponent':<14} {'W':>3} {'L':>3} {'D':>3} {'Win%':>6}  {'Goals'}")
    print(f"{'-'*70}")
    for (agent, opp), (w, l, d, og, opg) in results.items():
        print(f"{agent:<22} {opp:<14} {w:>3} {l:>3} {d:>3} {w/N_MATCHES*100:>5.1f}%  {og}-{opg}")
    print(f"{'='*70}")
