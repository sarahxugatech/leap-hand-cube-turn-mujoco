"""Mirror one public 5 cm left-LEAP cache pose into the right-hand test scene."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import mujoco
import numpy as np

from import_leap_cache_pose import (
    CUBE_CENTRE,
    EXPECTED_SCENE_SHA256,
    MAIN_SCENE,
    MENAGERIE_ORDER,
    ROOT,
    cube_symmetries,
    free_mount_scene,
    quaternion_wxyz,
    scan_orientation,
    sha256,
)


MIRRORED_JOINTS = {"if_rot", "mf_rot", "rf_rot", "th_axl"}
LEFT_FROM_RIGHT = np.diag([1.0, -1.0, 1.0])
# Derived from the three corresponding finger-base origins in the source left/right MJCFs.
LEFT_FROM_RIGHT_TRANSLATION = np.array([-0.0755, 0.0627, -0.0265])
SOURCE_ENTITY_POSITION = np.array([0.0, 0.0, 0.1])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    before_hash = sha256(MAIN_SCENE)
    if before_hash != EXPECTED_SCENE_SHA256:
        raise RuntimeError(f"frozen scene hash mismatch: {before_hash}")
    with np.load(args.cache, allow_pickle=False) as cache:
        required = {"cube_size", "joint_pos", "cube_pose_rel", "joint_names"}
        if set(cache.files) != required:
            raise ValueError(f"unexpected cache fields: {cache.files}")
        names = [str(name) for name in cache["joint_names"]]
        if names != list(MENAGERIE_ORDER):
            raise ValueError(f"unexpected joint order: {names}")
        row_index = int(np.argmin(np.abs(cache["cube_size"] - 0.025)))
        source_size = float(cache["cube_size"][row_index])
        source_joints = cache["joint_pos"][row_index].astype(float)
        source_cube = cache["cube_pose_rel"][row_index].astype(float)
        cache_count = int(cache["cube_size"].shape[0])

    joints = {
        name: float(-value if name in MIRRORED_JOINTS else value)
        for name, value in zip(names, source_joints, strict=True)
    }
    right_cube_position = LEFT_FROM_RIGHT @ (
        source_cube[:3] - SOURCE_ENTITY_POSITION - LEFT_FROM_RIGHT_TRANSLATION
    )

    source = MAIN_SCENE.read_text(encoding="utf-8")
    rows = []
    with TemporaryDirectory(prefix="prompt07_import_", dir=ROOT / "logs") as temporary:
        scene = free_mount_scene(source, Path(temporary))
        model = mujoco.MjModel.from_xml_path(str(scene))
        for index, symmetry in enumerate(cube_symmetries()):
            position = CUBE_CENTRE - symmetry @ right_cube_position
            mount = np.array([position[0], position[1], position[2] - 0.25])
            within_mount_range = (
                abs(mount[0]) <= 0.15
                and abs(mount[1]) <= 0.15
                and -0.30 <= mount[2] <= 0.10
            )
            distances = scan_orientation(model, joints, position, symmetry)
            rows.append({
                "index": index,
                "symmetry": symmetry,
                "position": position,
                "mount": mount,
                "within_mount_range": within_mount_range,
                **distances,
            })
    safe = [
        row for row in rows
        if row["within_mount_range"]
        and row["hand_table_m"] >= -0.002
        and row["hand_cube_m"] >= -0.002
    ]
    lines = [
        f"source_cache={args.cache}",
        f"source_cache_sha256={sha256(args.cache)}",
        f"source_commit={args.source_commit}",
        f"cache_count={cache_count}",
        f"selected_source_row={row_index}",
        f"selected_source_cube_half_size_m={source_size:.9f}",
        f"joint_order={names}",
        f"joint_sign_mirror={sorted(MIRRORED_JOINTS)}",
        f"right_frame_cube_position_m={np.array2string(right_cube_position, precision=9, separator=', ')}",
        f"source_entity_position_m={np.array2string(SOURCE_ENTITY_POSITION, precision=9, separator=', ')}",
        "custom_thumb_geometry_offset_risk_m=0.028500000",
        "orientation_index | in_range | hand_table_m | hand_cube_m | mount_qpos_xyz",
    ]
    lines.extend(
        f"{row['index']:17d} | {'YES' if row['within_mount_range'] else 'NO ':8s} | "
        f"{row['hand_table_m']:+.9f} | {row['hand_cube_m']:+.9f} | "
        f"{np.array2string(row['mount'], precision=9, separator=', ')}"
        for row in rows
    )
    if not safe:
        lines.extend((
            "selected_orientation_index=none",
            "stop_reason=no orientation satisfies mount and 0.002 m penetration limits",
            f"main_scene_sha256_before={before_hash}",
            f"main_scene_sha256_after={sha256(MAIN_SCENE)}",
        ))
        (ROOT / "logs" / "prompt07_mirrored_import_scan.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        raise RuntimeError("no cube-equivalent orientation satisfies mount and penetration limits")
    selected = max(safe, key=lambda row: row["hand_table_m"])

    payload = {
        "mount": {
            "mount_x": float(selected["mount"][0]),
            "mount_y": float(selected["mount"][1]),
            "mount_z": float(selected["mount"][2]),
            "mount_pitch": 0.0,
            "mount_roll": 0.0,
        },
        "mount_quat_wxyz": quaternion_wxyz(selected["symmetry"]).tolist(),
        "joints": joints,
        "provenance": {
            "source_repository": "https://github.com/Msornerrrr/in-hand-rotation-mjlab",
            "source_commit": args.source_commit,
            "source_cache_file": args.cache.name,
            "source_cache_sha256": sha256(args.cache),
            "source_cache_row": row_index,
            "source_cube_half_size_m": source_size,
            "joint_sign_mirror": sorted(MIRRORED_JOINTS),
            "left_from_right_translation_m": LEFT_FROM_RIGHT_TRANSLATION.tolist(),
            "source_entity_position_m": SOURCE_ENTITY_POSITION.tolist(),
            "custom_thumb_geometry_offset_risk_m": 0.0285,
            "selected_cube_symmetry_index": selected["index"],
            "selected_cube_symmetry_matrix": selected["symmetry"].tolist(),
        },
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    lines.extend((
        f"selected_orientation_index={selected['index']}",
        f"selected_hand_table_m={selected['hand_table_m']:+.9f}",
        f"selected_hand_cube_m={selected['hand_cube_m']:+.9f}",
        f"output_pose={args.output}",
        f"main_scene_sha256_before={before_hash}",
        f"main_scene_sha256_after={sha256(MAIN_SCENE)}",
    ))
    (ROOT / "logs" / "prompt07_mirrored_import_scan.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if sha256(MAIN_SCENE) != before_hash:
        raise RuntimeError("main scene changed during import")
    print(f"wrote {args.output}; source row {row_index}; cube symmetry {selected['index']}")


if __name__ == "__main__":
    main()
