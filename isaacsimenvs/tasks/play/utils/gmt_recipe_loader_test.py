"""Offline unit test for gmt_recipe_loader (pure numpy, no isaac/torch).

Run directly::

    python gmt_recipe_loader_test.py

Numeric tests (quat / stride / delta / safety) always run on synthetic data.
Data-dependent tests (resolve / load / mesh) need the GMT forge release; if it
is absent they SKIP (not fail). Override data locations with env vars
GMT_FORGE_ROOT / GMT_RECIPE so the test carries no hardcoded assumption when
run elsewhere.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gmt_recipe_loader import (  # noqa: E402
    _assert_safe_component,
    build_tracking_pool,
    load_object_pose_trajectory,
    quat_inv,
    quat_mul,
    resolve_gmt_recipe,
    stride_sample,
    to_delta,
    unique_mesh_paths,
)

FORGE_ROOT = os.environ.get("GMT_FORGE_ROOT", "/home/l/ws/RL/GMT/data/forge_releases")
RECIPE = os.environ.get(
    "GMT_RECIPE", "/home/l/ws/RL/GMT/data/recipes/v16_small6__train.recipe.json"
)
# v16_small6 has 16 unique (supplier, asset) geoms.
EXPECTED_ASSETS = 16


class _Skip(Exception):
    pass


def _require_data() -> None:
    if not (Path(RECIPE).exists() and Path(FORGE_ROOT).exists()):
        raise _Skip(f"GMT forge data absent (RECIPE={RECIPE}, FORGE_ROOT={FORGE_ROOT})")


# --- numeric tests (always run) ---------------------------------------------
def test_quat_inverse_roundtrip() -> None:
    rng = np.random.default_rng(0)
    q = rng.standard_normal((5, 4))
    q /= np.linalg.norm(q, axis=-1, keepdims=True)
    prod = quat_mul(q, quat_inv(q))
    identity = np.zeros((5, 4))
    identity[:, 0] = 1.0
    assert np.allclose(prod, identity, atol=1e-6), prod


def test_stride_includes_first_and_last() -> None:
    op = np.zeros((10, 7), dtype=np.float32)
    op[:, 3] = 1.0
    op[:, 0] = np.arange(10)  # x = frame index, easy to eyeball
    s = stride_sample(op, 3)  # range(0,10,3)=[0,3,6,9]; 9==last
    assert s.shape == (4, 7)
    assert s[0, 0] == 0 and s[-1, 0] == 9
    s2 = stride_sample(op, 4)  # [0,4,8] + append 9
    assert s2.shape == (4, 7)
    assert s2[-1, 0] == 9


def test_delta_first_is_zero_identity() -> None:
    wp = np.zeros((3, 7), dtype=np.float32)
    wp[:, 3] = 1.0
    wp[0, :3] = [0.0, 0.0, 0.02]
    wp[1, :3] = [0.1, -0.2, 0.30]
    wp[2, :3] = [0.0, 0.0, 0.05]
    # frame 1 rotated 90deg about +z
    c, s = np.cos(np.pi / 4), np.sin(np.pi / 4)
    wp[1, 3:7] = [c, 0.0, 0.0, s]
    dp, dq = to_delta(wp)
    assert np.allclose(dp[0], 0.0)
    assert np.allclose(dq[0], [1.0, 0.0, 0.0, 0.0], atol=1e-6)
    # dp preserves the pick-place lift (0.02 -> 0.30 = +0.28)
    assert abs(dp[1, 2] - 0.28) < 1e-5


def test_safe_component_rejects_traversal() -> None:
    for bad in ["", "..", ".", "a/b", "/abs", "a\\b"]:
        raised = False
        try:
            _assert_safe_component(bad)
        except ValueError:
            raised = True
        assert raised, f"should reject {bad!r}"
    _assert_safe_component("cube42_local_mocap")  # a valid one must pass


# --- data-dependent tests (skip if forge absent) ----------------------------
def test_resolve_asset_count_and_mesh_exists() -> None:
    _require_data()
    r = resolve_gmt_recipe(RECIPE, FORGE_ROOT)
    assert len(r.asset_groups) == EXPECTED_ASSETS, len(r.asset_groups)
    for g in r.asset_groups:
        assert g.mesh_path.exists(), g.mesh_path
        assert g.asset_key == f"{g.supplier_id}/{g.asset_id}"
        assert len(g.clips) > 0


def test_unique_mesh_paths_dedup_ordered() -> None:
    _require_data()
    r = resolve_gmt_recipe(RECIPE, FORGE_ROOT)
    mp = unique_mesh_paths(r)
    assert len(mp) == EXPECTED_ASSETS
    assert len(set(mp)) == len(mp), "mesh paths must be unique"


def test_load_object_pose_shape() -> None:
    _require_data()
    r = resolve_gmt_recipe(RECIPE, FORGE_ROOT)
    clip = r.asset_groups[0].clips[0]
    op = load_object_pose_trajectory(clip)
    assert op.ndim == 2 and op.shape[1] == 7, op.shape
    assert op.dtype == np.float32
    q0 = op[0, 3:7]
    assert abs(np.linalg.norm(q0) - 1.0) < 1e-3, "quat should be unit"


def test_stride_then_delta_on_real_clip() -> None:
    _require_data()
    r = resolve_gmt_recipe(RECIPE, FORGE_ROOT)
    clip = r.asset_groups[0].clips[0]
    op = load_object_pose_trajectory(clip)
    wp = stride_sample(op, 25)
    assert wp.shape[0] >= 2
    dp, dq = to_delta(wp)
    assert np.allclose(dp[0], 0.0)
    assert np.allclose(dq[0], [1.0, 0.0, 0.0, 0.0], atol=1e-5)
    assert np.allclose(np.linalg.norm(dq, axis=-1), 1.0, atol=1e-5)


def test_build_tracking_pool_shapes() -> None:
    _require_data()
    from dataclasses import replace

    r = resolve_gmt_recipe(RECIPE, FORGE_ROOT)
    # mini pool: first 2 assets x 1 clip each -> only 2 npz loaded (fast).
    mini = replace(
        r, asset_groups=tuple(replace(g, clips=g.clips[:1]) for g in r.asset_groups[:2])
    )
    pool = build_tracking_pool(mini, goal_stride=25)
    assert pool["traj_dp"].shape[0] == 2 and pool["traj_dq"].shape[0] == 2
    assert pool["traj_len"].shape == (2,)
    # asset_index must index into the mini mesh list (0 or 1), one per source asset
    assert sorted(pool["traj_asset_index"].tolist()) == [0, 1]
    max_k = pool["traj_dp"].shape[1]
    for i in range(2):
        length = int(pool["traj_len"][i])
        assert np.allclose(pool["traj_dp"][i, 0], 0.0)
        assert np.allclose(pool["traj_dq"][i, 0], [1, 0, 0, 0], atol=1e-5)
        if length < max_k:  # padding holds the last real waypoint
            assert np.allclose(pool["traj_dp"][i, -1], pool["traj_dp"][i, length - 1])
            assert np.allclose(pool["traj_dq"][i, length:], pool["traj_dq"][i, length - 1], atol=1e-6)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = skipped = failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
            passed += 1
        except _Skip as e:
            print(f"SKIP {fn.__name__}: {e}")
            skipped += 1
        except AssertionError as e:
            print(f"FAIL {fn.__name__}: {e}")
            failed += 1
    print(f"\n{passed} passed, {skipped} skipped, {failed} failed")
    sys.exit(1 if failed else 0)
