"""Render the real LEAP hand trajectory on the visual 2x2 TEST scene."""

from __future__ import annotations

import hashlib

import render_trajectory as renderer


def main() -> None:
    renderer.SCENE = renderer.ROOT / "scenes" / "table_2x2_cube_control_test.xml"
    renderer.OUTPUT = renderer.ROOT / "videos" / "leap_hand_2x2_turn.mp4"
    renderer.TITLE = "LEAP hand turns a 2x2 cube"
    renderer.EXPECTED_SCENE_SHA256 = hashlib.sha256(renderer.SCENE.read_bytes()).hexdigest()
    renderer.main()


if __name__ == "__main__":
    main()
