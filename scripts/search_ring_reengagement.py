"""Scan partial positive-DIP unload and early MCP/PIP re-engagement."""

from __future__ import annotations

import csv
import json

import mujoco
import numpy as np

import search_ring_phased_delay as phased
from search_articulated_face_kf03 import replay_to_kf02
from validate_poses import ROOT, object_id, qpos_address, sha256


OUTPUT_LOG = ROOT / "logs" / "prompt32_ring_reengagement_search.txt"
OUTPUT_CSV = ROOT / "logs" / "prompt32_ring_reengagement_traces.csv"
DIP_DELTAS = (0.04, 0.06, 0.08, 0.10)
DELAYS = (15, 25, 35, 45)


def classify(result: dict[str, object]) -> None:
    result["redlines_pass"] = (
        float(result["transition_contact_rate"]) >= 0.95
        and float(result["max_axis_drift_mm"]) < 0.3
        and float(result["transition_peak_normal_force_n"]) < 0.4
        and float(result["final_angle_deg"]) > 15.0
    )
    result["project_pass"] = (
        bool(result["redlines_pass"])
        and float(result["final_angle_deg"]) <= 20.0
        and float(result["hold_contact_rate"]) >= 0.95
        and float(result["support_rate"]) == 1.0
        and float(result["ring_core_rate"]) == 0.0
        and float(result["last_100_range_deg"]) <= 0.25
        and not bool(result["penetration_over_0.002_m"])
        and not result["warnings"]
    )


def clean_replay(
    model: mujoco.MjModel, dip_delta: float, delay: int
) -> dict[str, object]:
    data = mujoco.MjData(model)
    replay_to_kf02(model, data)
    cube_body = object_id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")
    face_joint = object_id(model, mujoco.mjtObj.mjOBJ_JOINT, "face_turn")
    face_qpos = int(model.jnt_qposadr[face_joint])
    starts = {
        name: float(data.qpos[qpos_address(model, name)]) for name in phased.RING_JOINTS
    }
    phased.DELTAS["rf_dip"] = dip_delta
    result, _ = phased.evaluate(
        model,
        data,
        delay,
        starts,
        float(data.qpos[face_qpos]),
        data.xpos[cube_body].copy(),
    )
    result["dip_delta_rad"] = dip_delta
    classify(result)
    return result


def main() -> None:
    main_before = sha256(phased.MAIN_SCENE)
    scene_before = sha256(phased.SCENE)
    if main_before != phased.EXPECTED_MAIN_SHA256 or scene_before != phased.EXPECTED_SCENE_SHA256:
        raise RuntimeError("scene hash mismatch before re-engagement search")
    model = mujoco.MjModel.from_xml_path(str(phased.SCENE))
    data = mujoco.MjData(model)
    replay_to_kf02(model, data)
    cube_body = object_id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")
    face_joint = object_id(model, mujoco.mjtObj.mjOBJ_JOINT, "face_turn")
    face_qpos = int(model.jnt_qposadr[face_joint])
    starts = {
        name: float(data.qpos[qpos_address(model, name)]) for name in phased.RING_JOINTS
    }
    baseline_angle = float(data.qpos[face_qpos])
    baseline_cube_xyz = data.xpos[cube_body].copy()
    state_spec = mujoco.mjtState.mjSTATE_INTEGRATION
    state = np.empty(mujoco.mj_stateSize(model, state_spec))
    mujoco.mj_getState(model, data, state, state_spec)

    original_dip_delta = phased.DELTAS["rf_dip"]
    results: list[dict[str, object]] = []
    all_rows: list[dict[str, float | int | str]] = []
    try:
        for dip_delta in DIP_DELTAS:
            phased.DELTAS["rf_dip"] = dip_delta
            for delay in DELAYS:
                mujoco.mj_setState(model, data, state, state_spec)
                mujoco.mj_forward(model, data)
                result, rows = phased.evaluate(
                    model,
                    data,
                    delay,
                    starts,
                    baseline_angle,
                    baseline_cube_xyz,
                )
                result["dip_delta_rad"] = dip_delta
                classify(result)
                results.append(result)
                for row in rows:
                    row["dip_delta_rad"] = dip_delta
                all_rows.extend(rows)
    finally:
        phased.DELTAS["rf_dip"] = original_dip_delta

    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)

    redline_passing = sorted(
        (result for result in results if result["redlines_pass"]),
        key=lambda result: (
            float(result["max_axis_drift_mm"]),
            float(result["transition_peak_normal_force_n"]),
        ),
    )
    project_passing = [result for result in redline_passing if result["project_pass"]]
    repeat_runs: list[dict[str, object]] = []
    if project_passing:
        best = project_passing[0]
        repeat_runs = [
            clean_replay(model, float(best["dip_delta_rad"]), int(best["delay_steps"])),
            clean_replay(model, float(best["dip_delta_rad"]), int(best["delay_steps"])),
        ]
    main_after = sha256(phased.MAIN_SCENE)
    scene_after = sha256(phased.SCENE)
    if main_after != main_before or scene_after != scene_before:
        raise RuntimeError("scene changed during re-engagement search")
    lines = [
        "test=Prompt 32 partial-unload early-reengagement 2D scan",
        f"dt_s={float(model.opt.timestep)}",
        f"fixed_mcp_pip_deltas_rad={json.dumps({'rf_mcp': 0.08, 'rf_pip': -0.06}, sort_keys=True)}",
        *(f"result={json.dumps(result, sort_keys=True)}" for result in results),
        f"best_redlines={json.dumps(redline_passing[0], sort_keys=True) if redline_passing else 'NONE'}",
        f"best_project={json.dumps(project_passing[0], sort_keys=True) if project_passing else 'NONE'}",
        f"repeat_runs={json.dumps(repeat_runs, sort_keys=True)}",
        f"trace={OUTPUT_CSV}",
        f"main_scene_sha256_before={main_before}",
        f"main_scene_sha256_after={main_after}",
        f"test_scene_sha256_before={scene_before}",
        f"test_scene_sha256_after={scene_after}",
    ]
    OUTPUT_LOG.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
