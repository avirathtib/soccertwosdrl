# Random Access Memories — SoccerTwos Agent

## Team
Random Access Memories

**Avirath Tibrewala** — atibrewala3@gatech.edu

## Agent Description
PPO agent trained with prioritized self-play and dense reward shaping on the SoccerTwos 2v2 environment.

**Architecture:** 2-layer MLP (336 → 256 → 256 → 9 logits), MultiDiscrete [3,3,3] action space.

**Training:**
- Proximal Policy Optimization (PPO) via RLlib
- Prioritized self-play with history pool of 20 snapshots, triangular sampling biased toward recent opponents
- Dense reward shaping: ball proximity, ball-to-goal angle, approach deltas, defensive cover, anti-clustering, pressure escape, uncontested clear, assist/own-goal credit assignment
- League training phase: 40% CEIA baseline mixing after ~20M steps
- ~30M total environment steps

**Inference:** Deterministic argmax action selection.

## Files
- `agent.py` — Agent implementation (`RandomAccessMemoriesAgent`)
- `__init__.py` — Module export
- `selfplay_team_weights.pth` — Trained policy weights

## Usage
```bash
# Watch vs baseline
python -m soccer_twos.watch -m1 RANDOMACCESSMEMORIES_AGENT -m2 ceia_baseline_agent

# Self-play
python -m soccer_twos.watch -m RANDOMACCESSMEMORIES_AGENT
```

