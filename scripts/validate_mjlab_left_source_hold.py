"""Re-prove one MjLab cache row in its original modified-left LEAP geometry."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MAIN_SCENE = ROOT / "scenes" / "table_cube.xml"
EXPECTED_MAIN_SHA = "1e1e7666992f8e2c82ed11c05723489df0f9614f5a345021be16360eadf7f2a1"
SOURCE_ROOT = Path("/private/tmp/in_hand_rotation_mjlab_prompt06")
SOURCE_XML = SOURCE_ROOT / "src/in_hand_rotation_mjlab/robots/leap_hand/xmls/left_hand_custom.xml"
CACHE = SOURCE_ROOT / "src/in_hand_rotation_mjlab/tasks/hand_cube/cache/leap_left_custom_grasp_cache_cube5cm.npz"
ROW = 3388
STEPS = 500
SUSTAINED_STEPS = 100
PENETRATION_LIMIT = 0.002

JOINT_NAMES = (
    "if_mcp", "if_rot", "if_pip", "if_dip",
    "mf_mcp", "mf_rot", "mf_pip", "mf_dip",
    "rf_mcp", "rf_rot", "rf_pip", "rf_dip",
    "th_cmc", "th_axl", "th_mcp", "th_ipl",
)
SOURCE_RESET_JOINTS = {
    "if_mcp": 0.1, "if_rot": 0.4, "if_pip": 1.3, "if_dip": 0.0,
    "mf_mcp": 0.1, "mf_rot": 0.0, "mf_pip": 1.3, "mf_dip": 0.0,
    "rf_mcp": 0.1, "rf_rot": -0.4, "rf_pip": 1.3, "rf_dip": 0.0,
    "th_cmc": 1.45, "th_axl": -1.5, "th_mcp": 0.579, "th_ipl": 1.37,
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def name(model: mujoco.MjModel, kind: mujoco.mjtObj, index: int) -> str:
    return mujoco.mj_id2name(model, kind, index) or f"unnamed_{index}"


def finger(geom_name: str) -> str | None:
    prefix = geom_name.split("_", 1)[0]
    return prefix if prefix in {"if", "mf", "rf", "th"} else None


def descendant(model: mujoco.MjModel, body: int, ancestor: int) -> bool:
    while body > 0:
        if body == ancestor:
            return True
        body = int(model.body_parentid[body])
    return False


def test_scene(path: Path, cube_size: float, cube_pose: np.ndarray) -> None:
    quat = " ".join(f"{value:.12g}" for value in cube_pose[3:])
    source = f"""<mujoco model="prompt09 source left hold">
  <compiler angle="radian"/>
  <option timestep="0.005" integrator="implicitfast" cone="elliptic" impratio="10" iterations="10" ls_iterations="20"/>
  <asset>
    <model name="source_left" file="{SOURCE_XML}"/>
  </asset>
  <worldbody>
    <body name="robot_root" pos="0 0 0.1">
      <attach model="source_left" body="palm" prefix=""/>
    </body>
    <body name="cube" pos="{cube_pose[0]:.12g} {cube_pose[1]:.12g} {cube_pose[2]:.12g}" quat="{quat}">
      <freejoint name="cube_free"/>
      <geom name="cube_geom" type="box" size="{cube_size:.12g} {cube_size:.12g} {cube_size:.12g}" mass="0.1" friction="1.0" rgba="0.85 0.3 0.15 1"/>
    </body>
  </worldbody>
</mujoco>
"""
    path.write_text(source, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--joint-source", choices=("cache", "source-reset"), default="cache")
    args = parser.parse_args()
    prompt = "09" if args.joint_source == "cache" else "10"
    log = ROOT / "logs" / f"prompt{prompt}_source_left_passive_hold.txt"
    main_before = sha256(MAIN_SCENE)
    if main_before != EXPECTED_MAIN_SHA:
        raise RuntimeError(f"frozen main scene hash mismatch: {main_before}")
    if not SOURCE_XML.exists() or not CACHE.exists():
        raise FileNotFoundError("Prompt 09 source model or cache is missing")

    with np.load(CACHE, allow_pickle=False) as cache:
        cache_names = tuple(str(value) for value in cache["joint_names"])
        if cache_names != JOINT_NAMES:
            raise ValueError(f"unexpected cache joint order: {cache_names}")
        cube_size = float(cache["cube_size"][ROW])
        cache_joint_values = cache["joint_pos"][ROW].astype(float)
        cube_pose = cache["cube_pose_rel"][ROW].astype(float)
    joint_values = (
        cache_joint_values
        if args.joint_source == "cache"
        else np.array([SOURCE_RESET_JOINTS[name] for name in JOINT_NAMES], dtype=float)
    )

    with TemporaryDirectory(prefix="prompt09_source_left_", dir=ROOT / "logs") as temporary:
        scene = Path(temporary) / "source_left_hold.xml"
        test_scene(scene, cube_size, cube_pose)
        model = mujoco.MjModel.from_xml_path(str(scene))
        data = mujoco.MjData(model)
        if model.neq:
            raise RuntimeError("TEST scene contains equality constraints")
        cube_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "cube_free")
        if model.jnt_type[cube_joint] != mujoco.mjtJoint.mjJNT_FREE:
            raise RuntimeError("cube is not free")
        for joint_name, value in zip(JOINT_NAMES, joint_values, strict=True):
            joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            actuator = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{joint_name}_act")
            if joint < 0 or actuator < 0:
                raise ValueError(f"source model is missing {joint_name}")
            data.qpos[int(model.jnt_qposadr[joint])] = value
            data.ctrl[actuator] = value
        # The source task randomizes one shared slide-friction value in [0.6, 1.4]; 1.0 is nominal.
        model.geom_friction[:, 0] = 1.0
        mujoco.mj_forward(model, data)

        palm = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "palm")
        cube_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")
        cube_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "cube_geom")
        hand = [
            geom for geom in range(model.ngeom)
            if descendant(model, int(model.geom_bodyid[geom]), palm)
            and (model.geom_contype[geom] or model.geom_conaffinity[geom])
        ]
        hand_set = set(hand)

        def contacts() -> list[str]:
            found = []
            for contact in data.contact[: data.ncon]:
                geoms = {int(contact.geom1), int(contact.geom2)}
                if cube_geom not in geoms:
                    continue
                other = int(contact.geom2 if contact.geom1 == cube_geom else contact.geom1)
                if other in hand_set:
                    found.append(name(model, mujoco.mjtObj.mjOBJ_GEOM, other))
            return found

        initial_contacts = contacts()
        initial_thumb = any(finger(value) == "th" for value in initial_contacts)
        initial_fingers = sorted({
            prefix for value in initial_contacts
            if (prefix := finger(value)) in {"if", "mf", "rf"}
        })
        initial_gate = initial_thumb and len(initial_fingers) >= 2

        cube_qpos = int(model.jnt_qposadr[cube_joint])
        start_position = data.xpos[cube_body].copy()
        start_quat = data.qpos[cube_qpos + 3 : cube_qpos + 7].copy()
        maximum_displacement = 0.0
        minimum_hand_cube = min(
            float(mujoco.mj_geomDistance(model, data, cube_geom, geom, 1.0, None))
            for geom in hand
        )
        final_contacts = initial_contacts
        thumb_history = [initial_thumb]
        last_contact_step = 0 if initial_contacts else None
        warnings: list[str] = []

        if initial_gate:
            old_handler = mujoco.set_mju_user_warning(warnings.append)
            try:
                for step in range(1, STEPS + 1):
                    mujoco.mj_step(model, data)
                    maximum_displacement = max(
                        maximum_displacement,
                        float(np.linalg.norm(data.xpos[cube_body] - start_position)),
                    )
                    minimum_hand_cube = min(
                        minimum_hand_cube,
                        *(float(mujoco.mj_geomDistance(model, data, cube_geom, geom, 1.0, None)) for geom in hand),
                    )
                    final_contacts = contacts()
                    thumb_history.append(any(finger(value) == "th" for value in final_contacts))
                    if final_contacts:
                        last_contact_step = step
            finally:
                mujoco.set_mju_user_warning(old_handler)
            warnings.extend(
                mujoco.mjtWarning(index).name
                for index in range(mujoco.mjtWarning.mjNWARNING)
                if data.warning[index].number
            )

        end_position = data.xpos[cube_body].copy()
        end_displacement = float(np.linalg.norm(end_position - start_position))
        end_quat = data.qpos[cube_qpos + 3 : cube_qpos + 7]
        rotation_change = float(2 * np.arccos(np.clip(abs(np.dot(start_quat, end_quat)), 0.0, 1.0)))
        final_thumb = any(finger(value) == "th" for value in final_contacts)
        final_fingers = sorted({
            prefix for value in final_contacts
            if (prefix := finger(value)) in {"if", "mf", "rf"}
        })
        final_tip_fingers = sorted({
            prefix for value in final_contacts
            if value.endswith("_tip") and (prefix := finger(value)) in {"if", "mf", "rf"}
        })
        thumb_sustained = len(thumb_history) >= SUSTAINED_STEPS and all(thumb_history[-SUSTAINED_STEPS:])
        penetration = minimum_hand_cube < -PENETRATION_LIMIT
        safe = not penetration and not warnings
        gate_a = initial_gate and thumb_sustained and len(final_tip_fingers) >= 2 and maximum_displacement < 0.005 and end_displacement < 0.005 and safe
        relaxed = initial_gate and thumb_sustained and final_thumb and len(final_fingers) >= 2 and maximum_displacement < 0.015 and end_displacement < 0.015 and safe
        passive_hold = relaxed

    main_after = sha256(MAIN_SCENE)
    if main_after != main_before:
        raise RuntimeError("main scene changed during Prompt 09")
    lines = [
        f"test=Prompt {prompt} original modified-left source geometry passive hold",
        f"source_commit=0ab03eb2244b4a1a020d94ef38a4823bb9e15ba0",
        f"source_xml={SOURCE_XML}",
        f"source_xml_sha256={sha256(SOURCE_XML)}",
        f"source_cache={CACHE}",
        f"source_cache_sha256={sha256(CACHE)}",
        f"source_row={ROW}",
        f"joint_source={args.joint_source}",
        f"joint_values={dict(zip(JOINT_NAMES, joint_values.tolist(), strict=True))}",
        f"cube_half_size_m={cube_size:.9f}",
        "cube_mass_kg=0.100000000",
        "nominal_shared_slide_friction=1.000000000",
        "timestep_s=0.005000000",
        f"cube_start_pose_wxyz={np.array2string(cube_pose, precision=9, separator=', ')}",
        f"initial_contacts={initial_contacts or 'none'}",
        f"initial_thumb={initial_thumb}",
        f"initial_non_thumb_fingers={initial_fingers}",
        f"INITIAL_CONTACT_GATE={'PASS' if initial_gate else 'FAIL'}",
        f"dynamic_steps_run={STEPS if initial_gate else 0}",
        "cube_hand_attachment=NO",
        "equality_constraints=0",
        "cube_qpos_updates_during_run=0",
        f"cube_start_m={np.array2string(start_position, precision=9, separator=', ')}",
        f"cube_end_m={np.array2string(end_position, precision=9, separator=', ')}",
        f"maximum_3d_displacement_m={maximum_displacement:.9f}",
        f"end_3d_displacement_m={end_displacement:.9f}",
        f"rotation_change_rad={rotation_change:.9f}",
        f"last_hand_contact_step={last_contact_step if last_contact_step is not None else 'none'}",
        f"thumb_sustained={thumb_sustained}",
        f"final_contacts={final_contacts or 'none'}",
        f"final_non_thumb_fingers={final_fingers}",
        f"minimum_hand_cube_distance_m={minimum_hand_cube:+.9f}",
        f"penetration_over_0.002_m={penetration}",
        f"warnings={warnings or 'none'}",
        f"GATE_A={'PASS' if gate_a else 'FAIL'}",
        f"RELAXED={'PASS' if relaxed else 'FAIL'}",
        f"PASSIVE_HOLD={'PASS' if passive_hold else 'FAIL'}",
        "main_scene_updated=NO",
        f"main_scene_sha256_before={main_before}",
        f"main_scene_sha256_after={main_after}",
    ]
    log.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
