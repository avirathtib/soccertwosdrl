"""
Full Shaped @ 9M vs Ablation @ 9M — extended evaluation.
20 matches x 10 episodes, alternating sides every match.
"""
import numpy as np
import torch
import torch.nn as nn
import pickle
import soccer_twos

N_MATCHES          = 20
EPISODES_PER_MATCH = 10


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


def load_model(path):
    if path.endswith(".pth"):
        raw = torch.load(path)
    else:
        with open(path, "rb") as f:
            data = pickle.load(f)
        worker = pickle.loads(data["worker"])
        state  = worker["state"]["current_team"]
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


if __name__ == "__main__":
    print("Loading models...")
    fs_model  = load_model("selfplay_team_weights.pth")
    abl_model = load_model("ablation_weights_9M.pth")
    print("Done.\n")

    env = soccer_twos.make(render=False)

    fs_wins = fs_losses = draws = 0
    fs_goals = abl_goals = 0
    blue_wins = orange_wins = 0  # track side bias

    print(f"Full Shaped FINAL (~30M)  vs  Ablation @ 9M  ({N_MATCHES} matches x {EPISODES_PER_MATCH} eps)")
    print("=" * 55)

    for m in range(N_MATCHES):
        fs_is_blue = (m % 2 == 0)
        side = "blue  " if fs_is_blue else "orange"
        match_fs = match_abl = 0

        for _ in range(EPISODES_PER_MATCH):
            if fs_is_blue:
                result = run_episode(env, fs_model, abl_model)
                if result == "blue":   match_fs  += 1
                elif result == "orange": match_abl += 1
            else:
                result = run_episode(env, abl_model, fs_model)
                if result == "orange": match_fs  += 1
                elif result == "blue":   match_abl += 1

        won  = match_fs > match_abl
        lost = match_fs < match_abl
        tag  = "WIN " if won else ("LOSS" if lost else "DRAW")
        if won:    fs_wins   += 1
        elif lost: fs_losses += 1
        else:      draws     += 1
        fs_goals  += match_fs
        abl_goals += match_abl
        print(f"  Match {m+1:2d} [{side}]: {tag}  FS={match_fs}  Abl={match_abl}  (running: {fs_wins}W {fs_losses}L {draws}D)")

    env.close()

    total_ep = fs_goals + abl_goals
    print(f"\n{'='*55}")
    print(f"FINAL — Full Shaped FINAL vs Ablation @ 9M ({N_MATCHES} matches)")
    print(f"  Full Shaped:  {fs_wins}W  {fs_losses}L  {draws}D  |  match win% {fs_wins/N_MATCHES*100:.1f}%")
    print(f"  Goals:        FS {fs_goals} — {abl_goals} Ablation")
    print(f"  Episode win%: {fs_goals/total_ep*100:.1f}% (FS) vs {abl_goals/total_ep*100:.1f}% (Ablation)")
    print(f"\n  Blue side wins (overall): {sum(1 for m in range(N_MATCHES) if m%2==0)} matches were FS-as-blue")
    print(f"  (Check for side bias if results differ strongly by side)")
