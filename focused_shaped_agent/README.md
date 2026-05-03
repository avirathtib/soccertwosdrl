# Focused Shaped Agent — 5-Component Dense Reward @ 9M

## Team
Random Access Memories

**Avirath Tibrewala** — atibrewala3@gatech.edu

## Agent Description
PPO agent trained with a focused 5-component dense reward function at 9M steps. Used in the ablation study to demonstrate proxy exploitation (Goodhart's Law) — ball_angle dominates and crowds out the goal-scoring signal.

**Architecture:** 2-layer MLP (336 → 256 → 256 → 9 logits), MultiDiscrete [3,3,3] action space.

**Training:**
- Proximal Policy Optimization (PPO) via RLlib
- Prioritized self-play with history pool of 20 snapshots, triangular sampling biased toward recent opponents
- Reward shaping: proximity, contact, approach delta, ball angle, ball-to-goal delta
- 9M environment steps

**Role in study:** Reward modification comparison — illustrates Goodhart's Law in RL reward design; achieves 25% match win rate against sparse ablation at equal compute.

**Inference:** Deterministic argmax action selection.

## Files
- `agent.py` — Agent implementation (`FocusedShapedAgent`)
- `__init__.py` — Module export
- `weights.pth` — Trained policy weights (9M steps)
