# LEAP Hand 2x2 Cube Face Turn in MuJoCo

Work-in-progress reproduction of a dexterous in-hand manipulation task using one LEAP Hand in MuJoCo. The controller is deterministic and scripted; it does not use reinforcement learning, MPC, VLA, an attachment constraint, or direct face-joint position updates during a run.

## Current result

- Stable grasp and approximately 30 mm lift
- Contact-driven face turn to **15.766 degrees**
- Reverse travel of **15.811 degrees**, ending at approximately **-0.045 degrees**
- Two identical independent validation runs
- No penetration over 2 mm and no MuJoCo solver warning

The final 90-degree face-turn goal is **not yet complete**. The current bottleneck is transferring the pushing role to another finger without destabilizing the core grasp.

## Demonstrations

[Open the browser-based video demo](https://sarahxugatech.github.io/leap-hand-cube-turn-mujoco/)

### Hand-controlled 2x2 face turn

![LEAP Hand turning a 2x2 cube](videos/leap_hand_2x2_turn_preview.gif)

[Watch/download the hand-controlled result](videos/prompt36_leap_hand_2x2_turn.mp4)

### Four-finger unloaded motion

[Watch/download the unloaded finger demo](videos/leap_four_finger_motion_no_cube.mp4)

## Implementation

1. Load and verify the 16-joint LEAP Hand model from MuJoCo Menagerie.
2. Build table, free-cube, articulated-face, and visual 2x2 test scenes.
3. Search for and independently validate a stable grasp and lift.
4. Use cosine-interpolated keyframes and deterministic joint grids to reach and push the passive face layer.
5. Measure contact continuity, object drift, penetration, warnings, and repeatability.

The main controller changes are smooth cosine joint interpolation, staged grasp/lift/turn/reverse control, and automatic joint-angle search. The main simulation change is replacing a rigid cube with a simplified 2x2 model containing a passive rotating layer that can only be moved through contact.

## Setup

```bash
git clone --recurse-submodules <repository-url>
cd <repository-folder>
python3.12 -m venv .venv
.venv/bin/python -m pip install --only-binary=:all: -r requirements.txt
```

## Reproduce the measured result

```bash
.venv/bin/python scripts/validate_2x2_hand_turn.py
```

Expected final line:

```text
HAND_CONTROLLED_2X2_TRAJECTORY=PASS
```

Render the validated trajectory:

```bash
.venv/bin/python scripts/render_2x2_hand_turn.py
```

Render the unloaded four-finger demonstration:

```bash
.venv/bin/python scripts/render_four_finger_motion_no_cube.py
```

## Project layout

- `scenes/`: MuJoCo XML scenes
- `poses/`: measured controller targets and keyframes
- `scripts/`: search, validation, diagnostic, and rendering tools
- `logs/`: experiment evidence, including unsuccessful searches
- `videos/`: demonstrations and selected failure cases
- `PROCESS_LOG.md`: chronological implementation record

## Model attribution

The LEAP Hand model is provided through the [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) submodule. Refer to the Menagerie repository and its LEAP Hand directory for the applicable model license and attribution.

