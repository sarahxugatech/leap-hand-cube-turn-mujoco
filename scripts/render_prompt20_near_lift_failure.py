"""Render the independently verified Prompt 20 near-miss lift."""

import render_prompt19_lift_failure as renderer


renderer.POSE = renderer.ROOT / "poses" / "axis_support_near_N01.json"
renderer.OUTPUT = renderer.ROOT / "videos" / "prompt20_axis_support_near_lift_failure.mp4"
renderer.TITLE = "Prompt 20 near-miss N01"
renderer.EXPECTED_RISE_M = 0.019969417


if __name__ == "__main__":
    renderer.main()
