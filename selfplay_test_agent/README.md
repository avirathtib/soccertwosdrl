# Self-Play Test Agent

## Team
Random Access Memories

**Avirath Tibrewala** — atibrewala3@gatech.edu

## Agent Description
PPO agent trained with prioritized self-play and full shaped rewards, used for intermediate testing and evaluation. Uses stochastic action selection during inference (softmax sampling).

**Architecture:** 2-layer MLP (336 → 256 → 256 → 9 logits), MultiDiscrete [3,3,3] action space.

**Training:**
- Proximal Policy Optimization (PPO) via RLlib
- Prioritized self-play with history pool of 20 snapshots
- Dense reward shaping (full 14-component set)

**Inference:** Stochastic softmax sampling.

## Files
- `agent.py` — Agent implementation (`SelfPlayTestAgent`)
- `__init__.py` — Module export
- Weights loaded from `selfplay_team_weights.pth` (repo root)
