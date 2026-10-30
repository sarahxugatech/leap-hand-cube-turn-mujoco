"""Import one official LEAP cube-cache pose into the current Menagerie scene."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MAIN_SCENE = ROOT / "scenes" / "table_cube.xml"
EXPECTED_SCENE_SHA256 = "1e1e7666992f8e2c82ed11c05723489df0f9614f5a345021be16360eadf7f2a1"
CUBE_CENTRE = np.array([0.0, 0.0, 0.075])
SOURCE_HAND_POSITION = np.array([0.0, 0.0, 0.5])
SOURCE_HAND_ROTATION = np.diag([1.0, -1.0, -1.0])
MENAGERIE_PALM_POSITION = np.array([0.0, 0.0, 0.1])
MENAGERIE_PALM_ROTATION = np.diag([1.0, -1.0, -1.0])
CACHE_ORDER = (
    "if_mcp", "if_rot", "if_pip", "if_dip",
    "th_cmc", "th_axl", "th_mcp", "th_ipl",
    "mf_mcp", "mf_rot", "mf_pip", "mf_dip",
    "rf_mcp", "rf_rot", "rf_pip", "rf_dip",
)
MENAGERIE_ORDER = (
    "if_mcp", "if_rot", "if_pip", "if_dip",
    "mf_mcp", "mf_rot", "mf_pip", "mf_dip",
    "rf_mcp", "rf_rot", "rf_pip", "rf_dip",
    "th_cmc", "th_axl", "th_mcp", "th_ipl",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rotation_xyzw(quaternion: np.ndarray) -> np.ndarray:
    x, y, z, w = quaternion
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def quaternion_wxyz(rotation: np.ndarray) -> np.ndarray:
    quaternion = np.empty(4)
    trace = np.trace(rotation)
    if trace > 0:
        scale = 2 * np.sqrt(trace + 1)
        quaternion[:] = (
            0.25 * scale,
            (rotation[2, 1] - rotation[1, 2]) / scale,
            (rotation[0, 2] - rotation[2, 0]) / scale,
            (rotation[1, 0] - rotation[0, 1]) / scale,
        )
    else:
        axis = int(np.argmax(np.diag(rotation)))
        first, second = ((1, 2), (2, 0), (0, 1))[axis]
        scale = 2 * np.sqrt(1 + rotation[axis, axis] - rotation[first, first] - rotation[second, second])
        quaternion[axis + 1] = 0.25 * scale
        quaternion[0] = (rotation[second, first] - rotation[first, second]) / scale
        quaternion[first + 1] = (rotation[axis, first] + rotation[first, axis]) / scale
        quaternion[second + 1] = (rotation[axis, second] + rotation[second, axis]) / scale
    return quaternion / np.linalg.norm(quaternion)


def cube_symmetries() -> list[np.ndarray]:
    rotations = []
    for permutation in itertools.permutations(range(3)):
        for signs in itertools.product((-1.0, 1.0), repeat=3):
            rotation = np.zeros((3, 3))
            rotation[np.arange(3), permutation] = signs
            if np.isclose(np.linalg.det(rotation), 1.0):
                rotations.append(rotation)
    return rotations


def free_mount_scene(source: str, directory: Path) -> Path:
    relative = 'file="../menagerie/leap_hand/right_hand.xml"'
    absolute = f'file="{(ROOT / "menagerie/leap_hand/right_hand.xml").resolve()}"'
    source = source.replace(relative, absolute)
    source = source.replace('    <body name="mount" pos="0 0 0.25">', '    <body name="mount">')
    joint_lines = [
        '      <joint name="mount_x" type="slide" axis="1 0 0" range="-0.15 0.15" damping="1"/>',
        '      <joint name="mount_y" type="slide" axis="0 1 0" range="-0.15 0.15" damping="1"/>',
        '      <joint name="mount_z" type="slide" axis="0 0 1" range="-0.30 0.10" damping="1"/>',
        '      <joint name="mount_pitch" type="hinge" axis="0 1 0" range="-0.35 0.35" damping="1"/>',
        '      <joint name="mount_roll" type="hinge" axis="1 0 0" range="-0.35 0.35" damping="1"/>',
    ]
    if any(source.count(line) != 1 for line in joint_lines):
        raise ValueError("main scene mount joints changed")
    source = source.replace(joint_lines[0], '      <freejoint name="prompt06_mount_free"/>')
    for line in joint_lines[1:]:
        source = source.replace(line + "\n", "")
    for name in ("x", "y", "z", "pitch", "roll"):
        source = source.replace(
            next(line for line in source.splitlines() if f'name="mount_{name}_act"' in line) + "\n",
            "",
        )
    key_start = source.index("  <keyframe>")
    key_end = source.index("  </keyframe>", key_start) + len("  </keyframe>")
    source = source[:key_start] + source[key_end:]
    path = directory / "prompt06_free_mount.xml"
    path.write_text(source, encoding="utf-8")
    return path


def object_id(model: mujoco.MjModel, kind: mujoco.mjtObj, name: str) -> int:
    result = mujoco.mj_name2id(model, kind, name)
    if result < 0:
        raise ValueError(f"missing model object: {name}")
    return result


def is_descendant(model: mujoco.MjModel, body: int, ancestor: int) -> bool:
    while body > 0:
        if body == ancestor:
            return True
        body = int(model.body_parentid[body])
    return False


def scan_orientation(
    model: mujoco.MjModel,
    joints: dict[str, float],
    position: np.ndarray,
    rotation: np.ndarray,
) -> dict[str, float]:
    data = mujoco.MjData(model)
    cube_joint = object_id(model, mujoco.mjtObj.mjOBJ_JOINT, "cube_free")
    mount_joint = object_id(model, mujoco.mjtObj.mjOBJ_JOINT, "prompt06_mount_free")
    data.qpos[model.jnt_qposadr[cube_joint]:model.jnt_qposadr[cube_joint] + 7] = (*CUBE_CENTRE, 1, 0, 0, 0)
    mount_qpos = model.jnt_qposadr[mount_joint]
    data.qpos[mount_qpos:mount_qpos + 7] = (*position, *quaternion_wxyz(rotation))
    for name, value in joints.items():
        joint = object_id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        data.qpos[model.jnt_qposadr[joint]] = value
    mujoco.mj_forward(model, data)

    palm = object_id(model, mujoco.mjtObj.mjOBJ_BODY, "palm")
    hand = [geom for geom in range(model.ngeom) if is_descendant(model, int(model.geom_bodyid[geom]), palm)]
    table = object_id(model, mujoco.mjtObj.mjOBJ_GEOM, "table")
    cube = object_id(model, mujoco.mjtObj.mjOBJ_GEOM, "cube_geom")
    table_distance = min(float(mujoco.mj_geomDistance(model, data, table, geom, 1.0, None)) for geom in hand)
    cube_distance = min(float(mujoco.mj_geomDistance(model, data, cube, geom, 1.0, None)) for geom in hand)
    return {"hand_table_m": table_distance, "hand_cube_m": cube_distance}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--row", default=0, type=int)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--source-cube-side", required=True, type=float)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    before_hash = sha256(MAIN_SCENE)
    if before_hash != EXPECTED_SCENE_SHA256:
        raise RuntimeError(f"frozen scene hash mismatch: {before_hash}")
    cache = np.load(args.cache, allow_pickle=False)
    if cache.ndim != 2 or cache.shape[1] != 23:
        raise ValueError(f"expected cache shape (N, 23), got {cache.shape}")
    row = cache[args.row].astype(float)
    joints_by_cache = dict(zip(CACHE_ORDER, row[:16], strict=True))
    joints = {name: float(joints_by_cache[name]) for name in MENAGERIE_ORDER}

    object_rotation = rotation_xyzw(row[19:23])
    relative_position = object_rotation.T @ (SOURCE_HAND_POSITION - row[16:19])
    relative_rotation = object_rotation.T @ SOURCE_HAND_ROTATION

    source = MAIN_SCENE.read_text(encoding="utf-8")
    scan_rows = []
    with TemporaryDirectory(prefix="prompt06_import_", dir=ROOT / "logs") as temporary:
        scene = free_mount_scene(source, Path(temporary))
        model = mujoco.MjModel.from_xml_path(str(scene))
        for index, symmetry in enumerate(cube_symmetries()):
            palm_position = CUBE_CENTRE + symmetry @ relative_position
            palm_rotation = symmetry @ relative_rotation
            rotation = palm_rotation @ MENAGERIE_PALM_ROTATION.T
            position = palm_position - rotation @ MENAGERIE_PALM_POSITION
            distances = scan_orientation(model, joints, position, rotation)
            scan_rows.append({
                "index": index,
                "symmetry": symmetry,
                "position": position,
                "rotation": rotation,
                "palm_position": palm_position,
                **distances,
            })
    safe_rows = [row for row in scan_rows if row["hand_table_m"] >= -0.002 and row["hand_cube_m"] >= -0.002]
    if not safe_rows:
        raise RuntimeError("all 24 cube-equivalent orientations exceed the 2 mm penetration limit")
    selected = max(safe_rows, key=lambda row: row["hand_table_m"])
    position = selected["position"]
    quaternion = quaternion_wxyz(selected["rotation"])

    payload = {
        "mount": {
            "mount_x": float(position[0]),
            "mount_y": float(position[1]),
            "mount_z": float(position[2] - 0.25),
            "mount_pitch": 0.0,
            "mount_roll": 0.0,
        },
        "mount_quat_wxyz": quaternion.tolist(),
        "joints": joints,
        "provenance": {
            "source_repository": "https://github.com/leap-hand/LEAP_Hand_Sim",
            "source_commit": args.source_commit,
            "source_cache_file": args.cache.name,
            "source_cache_sha256": sha256(args.cache),
            "source_cache_row": args.row,
            "source_cube_side_m": args.source_cube_side,
            "current_cube_side_m": 0.05,
            "cache_dof_order": list(CACHE_ORDER),
            "selected_cube_symmetry_index": selected["index"],
            "selected_cube_symmetry_matrix": selected["symmetry"].tolist(),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    lines = [
        f"source_cache={args.cache}",
        f"source_cache_sha256={sha256(args.cache)}",
        f"source_commit={args.source_commit}",
        f"source_row={args.row}",
        f"source_cube_side_m={args.source_cube_side:.6f}",
        "current_cube_side_m=0.050000",
        f"cache_shape={cache.shape}",
        f"cache_dof_order={CACHE_ORDER}",
        f"menagerie_dof_order={MENAGERIE_ORDER}",
        "orientation_index | hand_table_m | hand_cube_m | mount_xyz | palm_xyz",
    ]
    lines.extend(
        f"{row['index']:17d} | {row['hand_table_m']:+.9f} | {row['hand_cube_m']:+.9f} | "
        f"{np.array2string(row['position'], precision=9, separator=', ')} | "
        f"{np.array2string(row['palm_position'], precision=9, separator=', ')}"
        for row in scan_rows
    )
    lines.extend((
        f"selected_orientation_index={selected['index']}",
        f"selected_hand_table_m={selected['hand_table_m']:+.9f}",
        f"selected_hand_cube_m={selected['hand_cube_m']:+.9f}",
        f"output_pose={args.output}",
        f"main_scene_sha256_before={before_hash}",
        f"main_scene_sha256_after={sha256(MAIN_SCENE)}",
    ))
    (ROOT / "logs" / "prompt06_import_scan.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if sha256(MAIN_SCENE) != before_hash:
        raise RuntimeError("main scene changed during import")
    print(f"wrote {args.output}; selected cube symmetry {selected['index']}")


if __name__ == "__main__":
    main()
