from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "scenes" / "table_cube.xml"
OUTPUT_PATH = ROOT / "videos" / "phase0_scene_stability.mp4"
MOUNT_TARGETS = {
    "mount_x_act": -0.0425,
    "mount_y_act": -0.0070,
    "mount_z_act": -0.2190,
    "mount_pitch_act": 0.10,
    "mount_roll_act": -0.10,
}


model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
data = mujoco.MjData(model)
key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "start")
mujoco.mj_resetDataKeyframe(model, data, key_id)
for actuator_name, target in MOUNT_TARGETS.items():
    actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
    data.ctrl[actuator_id] = target
mujoco.mj_forward(model, data)

cube_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "cube_free")
cube_qpos = model.jnt_qposadr[cube_joint]
cube_start = data.qpos[cube_qpos : cube_qpos + 3].copy()

camera = mujoco.MjvCamera()
camera.type = mujoco.mjtCamera.mjCAMERA_FREE
camera.lookat[:] = [-0.02, -0.01, 0.09]
camera.distance = 0.42
camera.elevation = -25

fps = 30
frame_times = np.linspace(0.0, 4.0, fps * 4)
OUTPUT_PATH.parent.mkdir(exist_ok=True)
with mujoco.Renderer(model, height=480, width=640) as renderer:
    with imageio.get_writer(OUTPUT_PATH, fps=fps, codec="libx264", quality=8) as writer:
        for frame_index, target_time in enumerate(frame_times):
            while data.time < target_time:
                mujoco.mj_step(model, data)
            camera.azimuth = 115 + 50 * frame_index / (len(frame_times) - 1)
            renderer.update_scene(data, camera)
            writer.append_data(renderer.render())

cube_end = data.qpos[cube_qpos : cube_qpos + 3]
xy_drift = np.linalg.norm(cube_end[:2] - cube_start[:2])
z_change = abs(cube_end[2] - cube_start[2])
assert xy_drift < 0.002 and z_change < 0.002
print(OUTPUT_PATH)
print(f"cube_xy_drift={xy_drift:.9f} cube_abs_z_change={z_change:.9f}")
