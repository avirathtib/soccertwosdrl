# Ablation Agent — Sparse Reward Baseline

## Team
Random Access Memories

**Avirath Tibrewala** — atibrewala3@gatech.edu

## Agent Description
PPO agent trained with **no reward shaping** — only the environment's default sparse ±1 terminal signal (goal scored / conceded). Used as the primary baseline in the reward shaping ablation study.

**Architecture:** 2-layer MLP (336 → 256 → 256 → 9 logits), MultiDiscrete [3,3,3] action space.

**Training:**
- Proximal Policy Optimization (PPO) via RLlib
- Prioritized self-play with history pool of 20 snapshots, triangular sampling biased toward recent opponents
- No dense reward shaping — sparse ±1 terminal only
- 9M environment steps

**Role in study:** Reward modification baseline — establishes the sample-efficiency floor and serves as the fixed opponent in all head-to-head evaluations.

**Inference:** Deterministic argmax action selection.

## Files
- `agent.py` — Agent implementation (`AblationAgent`)
- `__init__.py` — Module export
- Weights loaded from `ablation_weights_9M.pth` (repo root)
