"""Minimal loader for GMT-style dataset recipes (schema ``wuji_recipe:4``).

Parses a recipe JSON into (a) the unique geom mesh list used as Play training
geometry and (b) per-clip object 6D-pose trajectories used as stage-2 tracking
goals. Only ``object_pose_w`` is read from each clip; hand/wrist/landmark arrays
are ignored (the P2P embodiment is unchanged, so only the object is tracked).

Pure stdlib + numpy: no torch / isaac / wuji_tracking import, so it is unit
testable offline. Paths are anchored on the caller-supplied ``releases_root``
(never a hardcoded machine path) and resolved to absolute before use.

Recipe schema (wuji_recipe:4)::

    {"schema": "wuji_recipe:4", "release": "v16",
     "selections": [{"supplier": "local_mocap",
                     "clips": ["group_id/asset_id/clip_name", ...]}, ...]}

Layout under ``{releases_root}/{release}/``::

    {supplier}/assets/{asset_id}/mesh.stl
    {supplier}/trajectories/{group_id}/{asset_id}/{clip_name}.npz
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

RECIPE_SCHEMA = "wuji_recipe:4"
# GMT forge releases are fixed 50 fps (frame_dt = 0.02 s); recorded in each
# group.json but constant across the release, so kept as a default here.
DEFAULT_FRAME_DT = 0.02


@dataclass(frozen=True)
class GmtClip:
    """One motion clip (object-pose trajectory) bound to one asset."""

    clip_id: str  # "supplier/group/asset/clip"
    asset_key: str  # "supplier/asset_id" -- the mesh-binding key
    npz_path: Path


@dataclass(frozen=True)
class GmtAssetGroup:
    """One geom (mesh) plus every clip recorded on it."""

    supplier_id: str
    asset_id: str
    asset_key: str  # "supplier/asset_id"
    mesh_path: Path  # .../assets/{asset_id}/mesh.stl
    frame_dt: float
    clips: tuple[GmtClip, ...]


@dataclass(frozen=True)
class ResolvedGmtRecipe:
    """Recipe resolved to concrete mesh paths + clip trajectories."""

    release: str
    releases_root: Path
    asset_groups: tuple[GmtAssetGroup, ...]


def _assert_safe_component(name: str) -> None:
    """Reject path-traversal in an id component (must be a bare name stem)."""
    if not name or "/" in name or "\\" in name or name in (".", "..") or name.startswith("/"):
        raise ValueError(f"unsafe recipe id component: {name!r}")


def resolve_gmt_recipe(
    recipe_path: str | Path,
    releases_root: str | Path,
    *,
    frame_dt: float = DEFAULT_FRAME_DT,
    verify_mesh: bool = True,
) -> ResolvedGmtRecipe:
    """Resolve a ``wuji_recipe:4`` JSON into mesh paths + clip trajectories.

    Aggregates clips by ``asset_key = "supplier/asset_id"`` so every unique geom
    appears once with all its clips. ``verify_mesh`` checks each mesh.stl exists
    (cheap: <=~16 assets); clip npz existence is checked lazily at load time.
    """
    recipe_path = Path(recipe_path).resolve()
    releases_root = Path(releases_root).resolve()
    data = json.loads(recipe_path.read_text())

    schema = data.get("schema")
    if schema != RECIPE_SCHEMA:
        raise ValueError(f"unsupported recipe schema {schema!r}, expected {RECIPE_SCHEMA!r}")
    release = data["release"]
    _assert_safe_component(release)
    release_root = releases_root / release

    # asset_key -> mutable group accumulator (insertion order preserved).
    groups: dict[str, dict] = {}
    for sel in data["selections"]:
        supplier = sel["supplier"]
        _assert_safe_component(supplier)
        for rel in sel.get("clips", []):
            parts = rel.split("/")
            if len(parts) != 3:
                raise ValueError(f"clip id must be group/asset/clip, got {rel!r}")
            group_id, asset_id, clip_name = parts
            for comp in (group_id, asset_id, clip_name):
                _assert_safe_component(comp)
            asset_key = f"{supplier}/{asset_id}"
            npz_path = (
                release_root / supplier / "trajectories" / group_id / asset_id / f"{clip_name}.npz"
            )
            g = groups.setdefault(
                asset_key,
                {
                    "supplier": supplier,
                    "asset_id": asset_id,
                    "mesh_path": release_root / supplier / "assets" / asset_id / "mesh.stl",
                    "clips": [],
                },
            )
            g["clips"].append(
                GmtClip(clip_id=f"{supplier}/{rel}", asset_key=asset_key, npz_path=npz_path)
            )

    asset_groups: list[GmtAssetGroup] = []
    for asset_key, g in groups.items():
        if verify_mesh and not g["mesh_path"].exists():
            raise FileNotFoundError(f"mesh not found for {asset_key}: {g['mesh_path']}")
        asset_groups.append(
            GmtAssetGroup(
                supplier_id=g["supplier"],
                asset_id=g["asset_id"],
                asset_key=asset_key,
                mesh_path=g["mesh_path"],
                frame_dt=frame_dt,
                clips=tuple(g["clips"]),
            )
        )

    return ResolvedGmtRecipe(
        release=release, releases_root=releases_root, asset_groups=tuple(asset_groups)
    )


def unique_mesh_paths(recipe: ResolvedGmtRecipe) -> list[str]:
    """Ordered, de-duplicated mesh.stl paths (one per unique asset_key).

    This ordering is the single source of truth for the stage-1 object pool and
    the stage-2 traj->mesh index binding; both must derive mesh_index from the
    same list to avoid trajectory/geometry mismatch.
    """
    return [str(g.mesh_path) for g in recipe.asset_groups]


def load_object_pose_trajectory(clip: GmtClip) -> np.ndarray:
    """Load ``object_pose_w`` for one clip: (N, 7) float32 [x,y,z, qw,qx,qy,qz].

    World-frame, wxyz quaternion, metres, 50 fps. Only this array is read; the
    npz's hand/wrist/landmark arrays are left untouched (numpy lazily unpacks
    just the requested key).
    """
    with np.load(clip.npz_path, allow_pickle=False) as d:
        op = np.asarray(d["object_pose_w"], dtype=np.float32)
    if op.ndim != 2 or op.shape[1] != 7:
        raise ValueError(f"object_pose_w must be (N,7), got {op.shape} in {clip.npz_path}")
    return op


def stride_sample(object_pose: np.ndarray, goal_stride: int) -> np.ndarray:
    """Sub-sample a (N,7) trajectory every ``goal_stride`` frames.

    The original clip is a sequential 50 fps motion; stage-2 wants sparse
    tracking goals. Always includes frame 0 and the final frame so the discrete
    waypoint chain spans the whole clip. Returns (K, 7).
    """
    if goal_stride < 1:
        raise ValueError(f"goal_stride must be >= 1, got {goal_stride}")
    n = object_pose.shape[0]
    idx = list(range(0, n, goal_stride))
    if idx[-1] != n - 1:
        idx.append(n - 1)
    return object_pose[idx]


# --- Quaternion helpers (wxyz, pure numpy) -----------------------------------
def quat_inv(q: np.ndarray) -> np.ndarray:
    """Inverse of a (near-)unit quaternion, wxyz, shape (...,4)."""
    q = np.asarray(q, dtype=np.float64)
    conj = q.copy()
    conj[..., 1:] *= -1.0
    norm2 = np.sum(q * q, axis=-1, keepdims=True)
    return conj / norm2


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product a (x) b, wxyz, broadcasting over leading dims."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        axis=-1,
    )


def to_delta(waypoints: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert absolute (K,7) waypoints to first-frame-relative delta.

    Returns (dp (K,3) float32, dq (K,4) float32 wxyz) where::

        dp_k = p_k - p_0
        dq_k = q_k (x) q_0^{-1}

    So dp_0 = 0 and dq_0 = identity. At runtime a P2P env-local goal is rebuilt
    as ``goal_pos = anchor_pos + dp_k`` and ``goal_quat = dq_k (x) anchor_quat``,
    anchoring the GMT world trajectory onto the object's reset pose (supplier
    world-frame origins are inconsistent, so only relative motion is portable).
    """
    pos = np.asarray(waypoints[:, :3], dtype=np.float64)
    quat = np.asarray(waypoints[:, 3:7], dtype=np.float64)
    dp = pos - pos[0:1]
    q0_inv = quat_inv(quat[0:1])  # (1,4)
    dq = quat_mul(quat, np.broadcast_to(q0_inv, quat.shape))
    dq = dq / np.linalg.norm(dq, axis=-1, keepdims=True)
    return dp.astype(np.float32), dq.astype(np.float32)


def build_tracking_pool(
    recipe: ResolvedGmtRecipe,
    goal_stride: int,
    *,
    table_z: float,
    z_upper: float,
    xy_half: tuple[float, float],
    max_goals: int = 0,
) -> dict[str, np.ndarray]:
    """Discretize clips into P2P world-frame tracking goals + workspace filter.

    z is **table-aligned**: GMT world z=0 is the table, so the P2P goal height is
    ``table_z + clip_z`` (preserves the object's real height above the table). xy
    is **first-frame-relative** (absolute GMT xy has no physical meaning; anchored
    to the object reset pose at runtime, keeping motion centered in the workspace).

    A clip is **dropped** if its table-aligned goal_z ever exceeds ``z_upper``
    (arm can't reach that high) or its xy travel exceeds ``xy_half`` (leaves the
    workspace). Returns padded numpy arrays::

        traj_dxy         (N, max_K, 2) float32  xy delta vs first frame
        traj_z           (N, max_K)    float32  clip z (height above table)
        traj_dq          (N, max_K, 4) float32  wxyz rotation delta
        traj_len         (N,)          int64    real waypoint count
        traj_asset_index (N,)          int64    mesh index (unique_mesh_paths order)
        n_total / n_kept                        filter stats (ints)
    """
    key_to_idx = {g.asset_key: i for i, g in enumerate(recipe.asset_groups)}
    dxys: list[np.ndarray] = []
    zs: list[np.ndarray] = []
    dqs: list[np.ndarray] = []
    lens: list[int] = []
    aidx: list[int] = []
    n_total = 0
    for g in recipe.asset_groups:
        idx = key_to_idx[g.asset_key]
        for clip in g.clips:
            n_total += 1
            op = load_object_pose_trajectory(clip)
            wp = stride_sample(op, goal_stride)
            z = wp[:, 2].astype(np.float32)  # height above GMT table (z=0)
            dxy = (wp[:, :2] - wp[0:1, :2]).astype(np.float32)
            _, dq = to_delta(wp)
            k = z.shape[0]
            if max_goals > 0 and k > max_goals:
                z, dxy, dq, k = z[:max_goals], dxy[:max_goals], dq[:max_goals], max_goals
            # Workspace filter: table-aligned reach + xy travel.
            if float((table_z + z).max()) > z_upper:
                continue
            if (
                float(np.abs(dxy[:, 0]).max()) > xy_half[0]
                or float(np.abs(dxy[:, 1]).max()) > xy_half[1]
            ):
                continue
            dxys.append(dxy)
            zs.append(z)
            dqs.append(dq)
            lens.append(k)
            aidx.append(idx)

    n_kept = len(lens)
    if n_kept == 0:
        raise ValueError(
            f"no clips survived the workspace filter (z_upper={z_upper}, "
            f"xy_half={xy_half}); loosen the bounds or check table_z/recipe."
        )
    max_k = max(lens)
    traj_dxy = np.zeros((n_kept, max_k, 2), dtype=np.float32)
    traj_z = np.zeros((n_kept, max_k), dtype=np.float32)
    traj_dq = np.zeros((n_kept, max_k, 4), dtype=np.float32)
    traj_dq[..., 0] = 1.0  # identity-quat pad
    for i, (dxy, z, dq, k) in enumerate(zip(dxys, zs, dqs, lens)):
        traj_dxy[i, :k] = dxy
        traj_z[i, :k] = z
        traj_dq[i, :k] = dq
        if k < max_k:
            traj_dxy[i, k:] = dxy[-1]
            traj_z[i, k:] = z[-1]
            traj_dq[i, k:] = dq[-1]
    return {
        "traj_dxy": traj_dxy,
        "traj_z": traj_z,
        "traj_dq": traj_dq,
        "traj_len": np.asarray(lens, dtype=np.int64),
        "traj_asset_index": np.asarray(aidx, dtype=np.int64),
        "n_total": n_total,
        "n_kept": n_kept,
    }


__all__ = [
    "GmtClip",
    "GmtAssetGroup",
    "ResolvedGmtRecipe",
    "resolve_gmt_recipe",
    "unique_mesh_paths",
    "load_object_pose_trajectory",
    "stride_sample",
    "to_delta",
    "build_tracking_pool",
    "quat_inv",
    "quat_mul",
]
