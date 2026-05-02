"""
Run 10 matches x 10 episodes each: selfplay_test_agent vs ceia_baseline_agent.
Prints match scores (e.g. 7-3, 8-2) and a final summary.

Usage:
    conda activate soccertwos
    python run_matches.py
"""

import importlib
import numpy as np
import soccer_twos
from soccer_twos.utils import get_agent_class

N_MATCHES        = 10
EPISODES_PER_MATCH = 10


def load_agent(module_name):
    env = soccer_twos.make()
    mod = importlib.import_module(module_name)
    agent = get_agent_class(mod)(env)
    env.close()
    return agent


def run_episode(env, agent1, agent2, our_team_is_blue: bool):
    """
    One episode (ends when a goal is scored or timeout).
    Returns (our_goals, opp_goals).
    """
    obs = env.reset()
    done = {"__all__": False}
    blue_reward = orange_reward = 0.0

    while not done["__all__"]:
        blue_obs    = {0: obs[0], 1: obs[1]}
        orange_obs  = {0: obs[2], 1: obs[3]}

        if our_team_is_blue:
            our_actions  = agent1.act(blue_obs)
            opp_actions  = agent2.act(orange_obs)
        else:
            opp_actions  = agent2.act(blue_obs)
            our_actions  = agent1.act(orange_obs)

        actions = {
            0: our_actions[0]  if our_team_is_blue else opp_actions[0],
            1: our_actions[1]  if our_team_is_blue else opp_actions[1],
            2: opp_actions[0]  if our_team_is_blue else our_actions[0],
            3: opp_actions[1]  if our_team_is_blue else our_actions[1],
        }

        obs, reward, done, _ = env.step(actions)
        blue_reward   = reward[0] + reward[1]
        orange_reward = reward[2] + reward[3]

    our_reward = blue_reward if our_team_is_blue else orange_reward
    if our_reward > 0.5:
        return 1, 0
    elif our_reward < -0.5:
        return 0, 1
    else:
        return 0, 0   # draw episode / timeout


def play_match(env, agent1, agent2, match_num):
    our_goals = opp_goals = 0
    for ep in range(EPISODES_PER_MATCH):
        # alternate sides each episode to remove home advantage bias
        our_blue = (ep % 2 == 0)
        og, opg = run_episode(env, agent1, agent2, our_blue)
        our_goals += og
        opp_goals += opg
        side = "blue " if our_blue else "orange"
        result = "GOAL " if og else ("concede" if opg else "draw  ")
        print(f"  ep {ep+1:2d} [{side}] {result}  |  running: {our_goals}-{opp_goals}")

    won  = our_goals > opp_goals
    lost = our_goals < opp_goals
    tag  = "WIN " if won else ("LOSS" if lost else "DRAW")
    print(f"Match {match_num:2d}: {tag}  {our_goals}-{opp_goals}\n")
    return won, lost, our_goals, opp_goals


if __name__ == "__main__":
    print("Loading agents...")
    our_agent      = load_agent("selfplay_test_agent.agent")
    baseline_agent = load_agent("ceia_baseline_agent.agent_ray")

    env = soccer_twos.make(render=False)
    print(f"\nRunning {N_MATCHES} matches x {EPISODES_PER_MATCH} episodes each")
    print(f"selfplay_test_agent  vs  ceia_baseline_agent\n")
    print("=" * 50)

    wins = losses = draws = 0
    total_our = total_opp = 0

    for m in range(N_MATCHES):
        print(f"\n--- Match {m+1}/{N_MATCHES} ---")
        won, lost, og, opg = play_match(env, our_agent, baseline_agent, m + 1)
        if won:   wins   += 1
        elif lost: losses += 1
        else:      draws  += 1
        total_our  += og
        total_opp  += opg

    env.close()

    print("=" * 50)
    print(f"FINAL:  {wins}W  {losses}L  {draws}D  out of {N_MATCHES} matches")
    print(f"Goals:  {total_our}-{total_opp}")
    print(f"Match win rate: {wins/N_MATCHES*100:.0f}%")
    ep_total = total_our + total_opp
    if ep_total > 0:
        print(f"Episode win rate: {total_our/(ep_total):.1%}  "
              f"({total_our} scored / {ep_total} decisive episodes)")
