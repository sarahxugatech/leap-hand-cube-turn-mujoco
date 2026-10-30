"""Statically scan every 5 cm MjLab cache pose against the fixed source reset hand pose."""

from __future__ import annotations

import time
from pathlib import Path
from tempfile import TemporaryDirectory

import mujoco
import numpy as np

from validate_mjlab_left_source_hold import (
    CACHE,
    EXPECTED_MAIN_SHA,
    JOINT_NAMES,
    MAIN_SCENE,
    ROOT,
    SOURCE_RESET_JOINTS,
    descendant,
    finger,
    name,
    sha256,
    test_scene,
)


OUTPUT = ROOT / "logs" / "prompt11_cache_static_scan.txt"
PENETRATION_LIMIT = 0.002


def main() -> None:
    started = time.monotonic()
    before_hash = sha256(MAIN_SCENE)
    if before_hash != EXPECTED_MAIN_SHA:
        raise RuntimeError(f"frozen scene hash mismatch: {before_hash}")
    with np.load(CACHE, allow_pickle=False) as cache:
        sizes = cache["cube_size"].astype(float).copy()
        poses = cache["cube_pose_rel"].astype(float).copy()
        cache_names = tuple(str(value) for value in cache["joint_names"])
    if cache_names != JOINT_NAMES or poses.shape != (len(sizes), 7):
        raise ValueError("unexpected cache schema")

    with TemporaryDirectory(prefix="prompt11_scan_", dir=ROOT / "logs") as temporary:
        scene = Path(temporary) / "source_left_scan.xml"
        # Compile with the largest box so the static broadphase bound encloses every scanned size.
        test_scene(scene, float(np.max(sizes)), poses[0])
        model = mujoco.MjModel.from_xml_path(str(scene))
        data = mujoco.MjData(model)
        cube_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "cube_free")
        cube_qpos = int(model.jnt_qposadr[cube_joint])
        cube_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "cube_geom")
        palm = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "palm")
        hand = [
            geom for geom in range(model.ngeom)
            if descendant(model, int(model.geom_bodyid[geom]), palm)
            and (model.geom_contype[geom] or model.geom_conaffinity[geom])
        ]
        hand_set = set(hand)
        for joint_name in JOINT_NAMES:
            value = SOURCE_RESET_JOINTS[joint_name]
            joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            actuator = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{joint_name}_act")
            data.qpos[int(model.jnt_qposadr[joint])] = value
            data.ctrl[actuator] = value
        model.geom_friction[:, 0] = 1.0
        mujoco.mj_forward(model, data)

        finger_pairs = [
            (first, second)
            for index, first in enumerate(hand)
            for second in hand[index + 1 :]
            if finger(name(model, mujoco.mjtObj.mjOBJ_GEOM, first)) is not None
            and finger(name(model, mujoco.mjtObj.mjOBJ_GEOM, second)) is not None
            and finger(name(model, mujoco.mjtObj.mjOBJ_GEOM, first))
            != finger(name(model, mujoco.mjtObj.mjOBJ_GEOM, second))
        ]
        minimum_finger_finger = min(
            float(mujoco.mj_geomDistance(model, data, first, second, 0.01, None))
            for first, second in finger_pairs
        )
        self_safe = minimum_finger_finger >= -PENETRATION_LIMIT

        rows = []
        for row, (size, pose) in enumerate(zip(sizes, poses, strict=True)):
            if not np.isclose(np.linalg.norm(pose[3:]), 1.0, atol=1e-5):
                raise ValueError(f"row {row} has non-unit cube quaternion")
            model.geom_size[cube_geom] = size
            data.qpos[cube_qpos : cube_qpos + 7] = pose
            data.qvel[:] = 0.0
            mujoco.mj_forward(model, data)

            contact_names = []
            contact_distances = []
            for contact in data.contact[: data.ncon]:
                geoms = {int(contact.geom1), int(contact.geom2)}
                if cube_geom not in geoms:
                    continue
                other = int(contact.geom2 if contact.geom1 == cube_geom else contact.geom1)
                if other in hand_set:
                    contact_names.append(name(model, mujoco.mjtObj.mjOBJ_GEOM, other))
                    contact_distances.append(float(contact.dist))
            unique_contacts = sorted(set(contact_names))
            thumb = any(finger(value) == "th" for value in unique_contacts)
            fingers = sorted({
                prefix for value in unique_contacts
                if (prefix := finger(value)) in {"if", "mf", "rf"}
            })
            tips = sorted({
                prefix for value in unique_contacts
                if value.endswith("_tip") and (prefix := finger(value)) in {"if", "mf", "rf"}
            })
            minimum_distance = min(contact_distances, default=float("inf"))
            contact_gate = thumb and len(fingers) >= 2
            penetration_safe = minimum_distance >= -PENETRATION_LIMIT and self_safe
            candidate = contact_gate and penetration_safe
            rows.append({
                "row": row,
                "size": size,
                "thumb": thumb,
                "fingers": fingers,
                "tips": tips,
                "contacts": unique_contacts,
                "minimum_distance": minimum_distance,
                "contact_gate": contact_gate,
                "penetration_safe": penetration_safe,
                "candidate": candidate,
            })

    candidates = [row for row in rows if row["candidate"]]
    candidates.sort(
        key=lambda row: (
            len(row["tips"]), len(row["fingers"]), len(row["contacts"]), row["minimum_distance"]
        ),
        reverse=True,
    )
    after_hash = sha256(MAIN_SCENE)
    if after_hash != before_hash:
        raise RuntimeError("main scene changed during cache scan")
    summary = [
        "test=Prompt 11 full 5 cm cache static scan",
        f"cache={CACHE}",
        f"cache_sha256={sha256(CACHE)}",
        f"rows_scanned={len(rows)}",
        "joint_source=source-reset",
        f"finger_finger_minimum_distance_m={minimum_finger_finger:+.9f}",
        f"finger_finger_safe={self_safe}",
        f"rows_with_any_hand_contact={sum(bool(row['contacts']) for row in rows)}",
        f"rows_with_thumb_contact={sum(row['thumb'] for row in rows)}",
        f"rows_with_two_or_more_non_thumb_fingers={sum(len(row['fingers']) >= 2 for row in rows)}",
        f"rows_passing_raw_contact_gate={sum(row['contact_gate'] for row in rows)}",
        f"rows_penetration_safe={sum(row['penetration_safe'] for row in rows)}",
        f"static_candidates={len(candidates)}",
        f"elapsed_seconds={time.monotonic() - started:.3f}",
        "dynamic_steps_run=0",
        "main_scene_updated=NO",
        f"main_scene_sha256_before={before_hash}",
        f"main_scene_sha256_after={after_hash}",
        "",
        "top_static_candidates:",
        "row | size_m | thumb | fingers | tips | minimum_hand_cube_m | contacts",
    ]
    summary.extend(
        f"{row['row']} | {row['size']:.9f} | YES | {','.join(row['fingers'])} | "
        f"{','.join(row['tips']) or '-'} | {row['minimum_distance']:+.9f} | {','.join(row['contacts'])}"
        for row in candidates[:20]
    )
    if not candidates:
        summary.append("none")
    summary.extend((
        "",
        "all_rows:",
        "row | size_m | thumb | fingers | tips | contact_gate | penetration_safe | candidate | minimum_hand_cube_m | contacts",
    ))
    summary.extend(
        f"{row['row']} | {row['size']:.9f} | {'YES' if row['thumb'] else 'NO'} | "
        f"{','.join(row['fingers']) or '-'} | {','.join(row['tips']) or '-'} | "
        f"{'PASS' if row['contact_gate'] else 'FAIL'} | {'PASS' if row['penetration_safe'] else 'FAIL'} | "
        f"{'YES' if row['candidate'] else 'NO'} | "
        f"{row['minimum_distance']:+.9f} | {','.join(row['contacts']) or 'none'}"
        for row in rows
    )
    OUTPUT.write_text("\n".join(summary) + "\n", encoding="utf-8")
    print("\n".join(summary[: summary.index("all_rows:")]))


if __name__ == "__main__":
    main()
