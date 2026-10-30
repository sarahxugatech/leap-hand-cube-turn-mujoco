"""Render a side-by-side geometry preview of the TEST-only 2x2 cube."""

from __future__ import annotations

import math
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
SCENE = ROOT / "scenes" / "table_2x2_cube_test.xml"
OUTPUT = ROOT / "videos" / "prompt35_2x2_cube_preview.png"


def render_angle(
    model: mujoco.MjModel,
    renderer: mujoco.Renderer,
    angle: float,
) -> np.ndarray:
    data = mujoco.MjData(model)
    key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "start")
    hinge = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "face_turn")
    cube = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")
    mujoco.mj_resetDataKeyframe(model, data, key)
    data.qpos[int(model.jnt_qposadr[hinge])] = angle
    mujoco.mj_forward(model, data)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = data.xpos[cube]
    camera.distance = 0.18
    camera.azimuth = 135
    camera.elevation = -22
    renderer.update_scene(data, camera)
    return renderer.render().copy()


def main() -> None:
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    with mujoco.Renderer(model, height=480, width=480) as renderer:
        frames = [render_angle(model, renderer, angle) for angle in (0.0, math.pi / 2)]
    canvas = Image.new("RGB", (960, 540), (20, 20, 20))
    font = ImageFont.load_default(size=22)
    for index, (frame, label) in enumerate(zip(frames, ("0 deg", "90 deg"), strict=True)):
        canvas.paste(Image.fromarray(frame), (480 * index, 60))
        ImageDraw.Draw(canvas).text((20 + 480 * index, 18), label, font=font, fill="white")
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (250, 510),
        "2x2 TEST geometry preview - top four cubies share face_turn",
        font=font,
        fill=(255, 220, 90),
    )
    canvas.save(OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    main()
