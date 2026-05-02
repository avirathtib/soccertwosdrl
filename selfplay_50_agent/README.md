# Self-Play Agent — Early Checkpoint (50 iterations)

## Team
Random Access Memories

**Avirath Tibrewala** — atibrewala3@gatech.edu

## Agent Description
PPO agent trained with prioritized self-play, captured at an early checkpoint (50 training iterations, ~2M steps). Used for early-stage evaluation and debugging.

**Architecture:** 2-layer MLP (336 → 256 → 256 → 9 logits), MultiDiscrete [3,3,3] action space.

**Training:**
- Proximal Policy Optimization (PPO) via RLlib
- Prioritized self-play with history pool of 20 snapshots
- Dense reward shaping (full 14-component set)
- ~2M environment steps

**Inference:** Deterministic argmax action selection.

## Files
- `agent.py` — Agent implementation (`SelfPlay50Agent`)
- `__init__.py` — Module export
- Weights loaded from `selfplay_50_weights.pth` (repo root)
