"""
Runs full-duration matches (5000 steps, multiple goals per match) without
resetting on each goal — simulates the TA's updated evaluation format.

Usage:
    /opt/homebrew/Caskroom/miniconda/base/envs/soccertwos/bin/python3 run_fullmatch.py
"""

import importlib
import numpy as np
import soccer_twos
from soccer_twos.utils import get_agent_class

N_MATCHES = 10


def load_agent(module_name):
    env = soccer_twos.make()
    mod = importlib.import_module(module_name)
    agent = get_agent_class(mod)(env)
    env.close()
    return agent


def play_full_match(env, our_agent, baseline_agent, match_num, our_is_blue):
    """
    Run one full match until Unity episode ends naturally (timeout).
    Don't reset on goals — the env auto-resets positions internally.
    Count goals from reward spikes (team reward ≈ ±2 on goal).
    """
    obs = env.reset()
    our_goals = opp_goals = 0

    while True:
        if our_is_blue:
            our_actions  = our_agent.act({0: obs[0], 1: obs[1]})
            opp_actions  = baseline_agent.act({0: obs[2], 1: obs[3]})
        else:
            opp_actions  = baseline_agent.act({0: obs[0], 1: obs[1]})
            our_actions  = our_agent.act({0: obs[2], 1: obs[3]})

        actions = {
            0: our_actions[0]  if our_is_blue else opp_actions[0],
            1: our_actions[1]  if our_is_blue else opp_actions[1],
            2: opp_actions[0]  if our_is_blue else our_actions[0],
            3: opp_actions[1]  if our_is_blue else our_actions[1],
        }

        obs, reward, done, _ = env.step(actions)

        blue_r   = reward[0] + reward[1]
        orange_r = reward[2] + reward[3]

        # goal = team reward spike ≈ ±2
        if abs(blue_r) > 1.5 or abs(orange_r) > 1.5:
            if our_is_blue:
                if blue_r > 1.5:   our_goals += 1
                else:              opp_goals += 1
            else:
                if orange_r > 1.5: our_goals += 1
                else:              opp_goals += 1

        # episode ends on Unity timeout (not per-goal — env handles position resets)
        if done.get("__all__", False):
            break

    won  = our_goals > opp_goals
    lost = our_goals < opp_goals
    tag  = "WIN " if won else ("LOSS" if lost else "DRAW")
    side = "blue  " if our_is_blue else "orange"
    print(f"Match {match_num:2d} [{side}]: {tag}  {our_goals}-{opp_goals}")
    return won, lost, our_goals, opp_goals


if __name__ == "__main__":
    print("Loading agents...")
    our_agent      = load_agent("selfplay_test_agent.agent")
    baseline_agent = load_agent("ceia_baseline_agent.agent_ray")

    env = soccer_twos.make(render=False)
    print(f"\nRunning {N_MATCHES} full-duration matches (until Unity timeout)")
    print(f"selfplay_test_agent  vs  ceia_baseline_agent")
    print("=" * 50)

    wins = losses = draws = 0
    total_our = total_opp = 0

    for m in range(N_MATCHES):
        our_is_blue = (m % 2 == 0)  # alternate sides each match
        won, lost, og, opg = play_full_match(env, our_agent, baseline_agent, m+1, our_is_blue)
        if won:    wins   += 1
        elif lost: losses += 1
        else:      draws  += 1
        total_our  += og
        total_opp  += opg

    env.close()

    total_goals = total_our + total_opp
    print("=" * 50)
    print(f"FINAL:  {wins}W  {losses}L  {draws}D  out of {N_MATCHES} matches")
    print(f"Goals:  {total_our}-{total_opp}")
    print(f"Match win rate:    {wins/N_MATCHES*100:.0f}%")
    if total_goals > 0:
        print(f"Goal share:        {total_our/total_goals*100:.1f}%  ({total_our} scored / {total_goals} total)")
