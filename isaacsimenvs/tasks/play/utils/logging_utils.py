"""Task metric publishing for Play."""

from __future__ import annotations

import torch


def log_step_metrics(env) -> None:
    """Publish step-level extras consumed by RL-Games observers."""
    term_cfg = env.cfg.termination
    if term_cfg.max_consecutive_successes > 0:
        all_goals_hit = env._successes >= term_cfg.max_consecutive_successes
    else:
        all_goals_hit = torch.zeros_like(env._successes, dtype=torch.bool)

    episode_final = {
        "successes": env._successes.float(),
        "all_goals_hit": all_goals_hit.float(),
    }
    episode_final.update(
        {
            f"done_{name}": value.float()
            for name, value in env._termination_reasons.items()
        }
    )

    env.extras["episode_cumulative"] = env._reward_terms
    env.extras["episode_final"] = episode_final
    env.extras["successes"] = env._prev_episode_successes.float()
    env.extras["current_success_tolerance"] = float(env._current_success_tolerance)

    # --- Tracking-error (pos/angle) + success-rate scalars for wandb ---
    if hasattr(env, "object") and hasattr(env, "goal_viz"):
        obj_pos = env.object.data.root_pos_w
        goal_pos = env.goal_viz.data.root_pos_w
        pos_err = torch.norm(obj_pos - goal_pos, dim=-1)  # (N,) metres
        obj_q = env.object.data.root_quat_w
        goal_q = env.goal_viz.data.root_quat_w
        # geodesic quaternion angle = 2*acos(|<q1,q2>|)
        dot = (obj_q * goal_q).sum(dim=-1).abs().clamp(max=1.0)
        ang_err = 2.0 * torch.acos(dot)  # (N,) radians
        env.extras["track_pos_error_cm"] = pos_err.mean().item() * 100.0
        env.extras["track_pos_error_median_cm"] = pos_err.median().item() * 100.0
        env.extras["track_angle_error_deg"] = ang_err.mean().item() * 57.2957795
        env.extras["track_angle_error_median_deg"] = ang_err.median().item() * 57.2957795
    if hasattr(env, "_is_success"):
        env.extras["success_rate"] = env._is_success.float().mean().item()
    if getattr(env, "_is_track_complete", None) is not None:
        env.extras["track_follow_complete_rate"] = (
            env._is_track_complete.float().mean().item()
        )


__all__ = ["log_step_metrics"]
