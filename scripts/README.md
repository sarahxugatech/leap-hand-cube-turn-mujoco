# Scripts

## Primary entry points

| Script | Purpose |
|---|---|
| `validate_2x2_hand_turn.py` | Replay the measured trajectory twice and enforce the acceptance checks |
| `render_2x2_hand_turn.py` | Render the validated cube-turn trajectory |
| `render_four_finger_motion_no_cube.py` | Render the unloaded four-finger motion check |

## Supporting pipeline

The primary validator reuses the measured grasp and face-turn pipeline in:

- `validate_four_finger_reverse.py`
- `render_trajectory.py`
- `search_articulated_face_followthrough.py`
- `search_articulated_face_kf03.py`
- `validate_poses.py`

Earlier research tools for pose search, failure diagnosis, and ablation experiments are preserved on the `research-archive-2026-10-30` branch.

