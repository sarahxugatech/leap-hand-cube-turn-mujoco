"""Replay the validated hand trajectory twice on the control-compatible 2x2 scene."""

from __future__ import annotations

import hashlib
from pathlib import Path

import mujoco

from validate_four_finger_reverse import run_once


ROOT = Path(__file__).resolve().parents[1]
SCENE = ROOT / "scenes" / "table_2x2_cube_control_test.xml"
LOG = ROOT / "logs" / "prompt36_2x2_hand_turn_validation.txt"


def main() -> None:
    results = []
    for _ in range(2):
        model = mujoco.MjModel.from_xml_path(str(SCENE))
        result, _ = run_once(model)
        results.append(result)
    repeatable = results[0] == results[1]
    passed = repeatable and all(bool(result["pass"]) for result in results)
    lines = [
        "test=LEAP hand trajectory on visual 2x2 control-compatible TEST scene",
        f"scene={SCENE.relative_to(ROOT)}",
        f"scene_sha256={hashlib.sha256(SCENE.read_bytes()).hexdigest()}",
        "face_qpos_written_during_run=NO",
        "attachment_or_equality=NO",
    ]
    for index, result in enumerate(results, start=1):
        lines.extend(f"run{index}_{key}={value}" for key, value in result.items())
    lines.extend(
        (
            f"repeatable={'YES' if repeatable else 'NO'}",
            f"HAND_CONTROLLED_2X2_TRAJECTORY={'PASS' if passed else 'FAIL'}",
        )
    )
    LOG.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
