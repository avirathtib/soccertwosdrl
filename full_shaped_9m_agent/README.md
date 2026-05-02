# Full Shaped Agent — 14-Component Dense Reward @ 9M

## Team
Random Access Memories

**Avirath Tibrewala** — atibrewala3@gatech.edu

## Agent Description
PPO agent trained with a **14-component dense reward function** covering offense, defense, and teamwork. Included in the ablation study as the richest reward variant.

**Architecture:** 2-layer MLP (336 → 256 → 256 → 9 logits), MultiDiscrete [3,3,3] action space.

**Training:**
- Proximal Policy Optimization (PPO) via RLlib
- Prioritized self-play with history pool of 20 snapshots, triangular sampling biased toward recent opponents
- Dense reward shaping: time penalty, proximity, contact, approach delta, ball angle, ball-to-goal delta, ball behind penalty, ball deep penalty, defensive cover, pressure escape, uncontested clear, anti-clustering, assist, own-goal
- 9M environment steps

**Role in study:** Reward modification comparison — demonstrates that comprehensive reward shaping hurts at equal compute due to proxy exploitation (0% match win rate vs sparse ablation).

**Inference:** Deterministic argmax action selection.

## Files
- `agent.py` — Agent implementation (`FullShaped9MAgent`)
- `__init__.py` — Module export
- Weights loaded from `full_shaped_weights_9M.pth` (repo root)
