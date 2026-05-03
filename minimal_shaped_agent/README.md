# Minimal Shaped Agent — 2-Component Dense Reward @ 13M

## Team
Random Access Memories

**Avirath Tibrewala** — atibrewala3@gatech.edu

## Agent Description
PPO agent trained with a minimal 2-component dense reward function, extended to 13M steps. Best-performing reward variant in the ablation study — achieves 83% match win rate against the sparse ablation baseline.

**Architecture:** 2-layer MLP (336 → 256 → 256 → 9 logits), MultiDiscrete [3,3,3] action space.

**Training:**
- Proximal Policy Optimization (PPO) via RLlib
- Prioritized self-play with history pool of 20 snapshots, triangular sampling biased toward recent opponents
- Reward shaping: ball contact (0.001/step within 1.5 units) + ball-to-goal delta (0.025 × displacement) + terminal ±1
- 13M environment steps

**Role in study:** Primary reward modification agent — demonstrates that minimal causally-aligned dense shaping outperforms both sparse and richly-shaped alternatives at extended compute.

**Inference:** Deterministic argmax action selection.

## Files
- `agent.py` — Agent implementation (`MinimalShapedAgent`)
- `__init__.py` — Module export
- `weights.pth` — Trained policy weights (13M steps)
