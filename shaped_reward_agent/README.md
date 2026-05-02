# Shaped Reward Agent — Early Prototype

## Team
Random Access Memories

**Avirath Tibrewala** — atibrewala3@gatech.edu

## Agent Description
Early prototype PPO agent trained with basic shaped rewards in team-vs-policy mode with a single-player variation. Uses a different architecture (1-hidden-layer, 512 units, flattened action space) from later agents and is retained for historical comparison.

**Architecture:** 1-layer MLP (336 → 512 → 27 logits), flattened action space.

**Training:**
- Proximal Policy Optimization (PPO) via RLlib
- Team-vs-policy variation, single_player=True, flatten_branched=True
- Basic shaped rewards: ball proximity + ball-to-goal delta

**Inference:** Deterministic argmax over flattened action space.

## Files
- `agent.py` — Agent implementation (`ShapedRewardAgent`)
- `__init__.py` — Module export
- `policy_weights.pth` — Trained policy weights (self-contained)
