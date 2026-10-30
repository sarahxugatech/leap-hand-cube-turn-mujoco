"""Render a clearly labelled 90-degree geometry preview of the 2x2 face hinge."""

from __future__ import annotations

import math
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
SCENE = ROOT / "scenes" / "table_2x2_cube_test.xml"
OUTPUT = ROOT / "videos" / "prompt35_2x2_face_turn_preview.mp4"
FPS = 30
HOLD_START = 30
TURN_FRAMES = 120
HOLD_END = 60


def overlay(frame: np.ndarray, angle_deg: float) -> np.ndarray:
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image, "RGBA")
    font = ImageFont.load_default(size=20)
    draw.rounded_rectangle((12, 10, 625, 92), radius=10, fill=(0, 0, 0, 190))
    draw.text((24, 18), "2x2 face-turn geometry preview", font=font, fill="white")
    draw.text((24, 45), f"top layer: {angle_deg:6.2f} deg / 90.00 deg", font=font, fill=(255, 220, 70))
    draw.text((24, 70), "PREVIEW ONLY - not a hand-controlled result", font=font, fill=(255, 130, 100))
    return np.asarray(image)


def main() -> None:
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    data = mujoco.MjData(model)
    key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "start")
    hinge = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "face_turn")
    cube = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")
    mujoco.mj_resetDataKeyframe(model, data, key)
    hinge_qpos = int(model.jnt_qposadr[hinge])
    mujoco.mj_forward(model, data)

    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = data.xpos[cube]
    camera.distance = 0.18
    camera.azimuth = 135
    camera.elevation = -22

    total_frames = HOLD_START + TURN_FRAMES + HOLD_END
    with mujoco.Renderer(model, height=480, width=640) as renderer:
        with imageio.get_writer(
            OUTPUT,
            fps=FPS,
            codec="libx264",
            quality=8,
            macro_block_size=None,
        ) as writer:
            for frame_index in range(total_frames):
                if frame_index < HOLD_START:
                    fraction = 0.0
                elif frame_index < HOLD_START + TURN_FRAMES:
                    phase = (frame_index - HOLD_START + 1) / TURN_FRAMES
                    fraction = 0.5 - 0.5 * math.cos(math.pi * phase)
                else:
                    fraction = 1.0
                angle = fraction * math.pi / 2
                data.qpos[hinge_qpos] = angle
                mujoco.mj_forward(model, data)
                camera.lookat[:] = data.xpos[cube]
                renderer.update_scene(data, camera)
                writer.append_data(overlay(renderer.render(), math.degrees(angle)))

    print(OUTPUT)
    print(f"duration_s={total_frames / FPS:.3f}")
    print("face_turn_preview_deg=90.000000")
    print("hand_controlled=NO")


if __name__ == "__main__":
    main()
