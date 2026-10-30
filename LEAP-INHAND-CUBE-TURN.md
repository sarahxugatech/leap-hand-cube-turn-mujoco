# LEAP In-Hand Single-Face Cube Turn — Project Guide

**Frozen final goal:**
Use one LEAP Hand in desktop Python MuJoCo simulation to start from a stable held grasp, rotate one cube face by 90 degrees about one axis, and finish in another stable held grasp without dropping the cube or trapping it between the palm and fingers.

**Explicitly out of scope:**
A full Rubik's Cube solve, multi-face planning, color recognition, two-hand manipulation, pen spinning, hammer manipulation, physical hardware, or claims about PPO, VLA, or MPC that were not actually implemented and tested.

**New-project rule:**
The old `dexterous-leap-project` directory is reference-only. Do not delete, overwrite, or modify it. Reuse measured lessons, not old pass claims. Every success criterion must be measured again in this project.

**Companion skill:**
Read `SKILL.md` before changing the project. It defines staged gates, contact-sequence requirements, validation thresholds, and known failure modes.

---

## 1. Lessons from the previous project (2026-10-25 to 2026-10-29)

### Validated components

- **Phase 0 environment and model loading:** Python 3.12.14, MuJoCo 3.17.0, NumPy 2.5.4, and later mink 1.3.0. The Menagerie LEAP `right_hand.xml` model has `nq=16`, `nv=16`, `nu=16`, `nbody=18`, and `ngeom=89`. Index, middle, and ring joints are ordered `mcp`, `rot`, `pip`, `dip`; thumb joints are `cmc`, `axl`, `mcp`, `ipl`. Actuators use the joint name plus `_act`. Position actuators use `kp=3.0`, `kv=0.01`; default sliding friction is 0.2, fingertip friction is 0.5, the contact cone is elliptic, and `impratio=100`.
- **Phase 1 table and cube scene:** `scenes/table_cube.xml` uses a 0.025 m cube half-size, density 1000 kg/m³, mass 0.125 kg, and friction 0.2. The table top is at 0.05 m and the cube starts at `(0, 0, 0.075)`. Measured XY drift was approximately zero and Z change was 0.000108 m, below the 0.002 m gate. The hand base is driven by three slide joints plus pitch and roll, not mocap or a fixed palm.
- **Reusable validator:** `scripts/validate_poses.py` supports batch validation and summary tables. The essential fix was replacing instantaneous pose changes with a 600-step cosine interpolation from open pose followed by a 500-step hold.
- **Keyframe editor:** robot-keyframe-kit 0.3.5 detected all 21 joints and actuators. It must be launched with `--root-body mount`; otherwise the cube free joint is treated as the root and editing moves the cube instead of the hand.

### Solved issues

- **Cube displacement:** `power_wrap_B` reached 2.101 mm maximum and 1.999 mm final displacement. Thumb variant T3 reached 2.006 mm maximum and 1.984 mm final displacement. Similar maximum and final values showed that the cube stopped sliding after closure.
- **Finger crossing:** Separating finger rotation first (index `+0.40`, middle `0.00`, ring `-0.40`) and then applying moderate curl prevented finger-finger, hand-cube, and hand-table penetration in all three power-wrap poses.
- **Reference baseline:** mount `(-0.0425, -0.0070, -0.2190)`, pitch/roll `(+0.10, -0.10)`, carry Z `-0.10`. Joint order below is `(mcp, rot, pip, dip)`:
  - Index `(-0.05, +0.50, -0.45, -0.36)`
  - Middle `(-0.06, -0.45, -0.50, -0.32)`
  - Ring `(-0.25, -0.60, -0.15, -0.30)`
  - Thumb order `(cmc, axl, mcp, ipl)`: `(-0.349, -0.349, +1.530, +0.110)`

### Previously blocked issues

- No tested power-wrap pose produced thumb contact.
- No tested power-wrap pose produced fingertip contact.
- Final contact involved at most one non-tip finger.
- Lift, move, and place were not tested because no pose passed the relaxed entry gate: sustained thumb contact, at least two finger contacts, displacement below 15 mm, and no excessive penetration.
- Two mink IK attempts failed because the target definition was incompatible with this hand and mounting direction, not because of solver tuning. With the mount fixed, all fingertip residuals exceeded the 0.005 m threshold. With all 21 DOFs released, the solution hit the lower mount-Z and roll limits and moved the hand through the table.
- Requiring all four fingertips to meet opposite faces at the same middle height was therefore abandoned in favor of a power grasp plus finger gaiting.

### Engineering lessons

1. The thumb issue was structural, not a one-parameter tuning problem. Pitch, roll, yaw, and XY scans only reduced a 29–39 mm gap by 2–9 mm per scan.
2. Motion order matters: separate `rot` first, then curl `mcp`, `pip`, and `dip` in stages, and move the thumb last. Curling everything at once with zero rotation causes overlap.
3. Instantaneous pose changes are not valid tests. The old teleporting validator produced contact at step 0, 5.246 mm penetration, and a 246 mm ejection. Interpolation restored contact at step 19 with no excessive penetration.
4. Small displacement does not prove grasping. An open hand has zero cube displacement because it never touches the cube. Rank contact first, then displacement, and report `NO PASSING POSE` when no gate passes.
5. Report both maximum and end-of-hold displacement because a cube can look stable during closure and continue sliding during the hold.
6. The five mount joints are not a wrist. They move the thumb and fingers together and cannot independently correct thumb and fingertip height.
7. Do not design a precision thumb-to-single-finger pinch. Use the thumb pad against a multi-finger and palm power grasp.
8. Apply a stop-loss rule: after two structural failures, change the method rather than expanding the same grid.

---

## 2. External references and transferable ideas

### MuJoCo Playground — LeapCubeReorient / LeapCubeRotateZ

- Source: arXiv:2502.08844 and `mujoco_playground/_src/manipulation/leap_hand/reorient.py`.
- Method: parallel MJX/JAX simulation, PPO, asymmetric actor-critic observations, and a geodesic orientation reward.
- Training-scale values differ across available tables and must be checked against the current `manipulation_params.py` before citation. Do not assume this training is practical on a Mac CPU.
- Common failures include trapping the cube between the fingers and palm and finger-thumb interlocking.
- Transferable ideas: start from a held cube, monitor known failure modes, and separate deployable proprioception from privileged simulator state.
- Non-transferable items: PPO checkpoints, MJX batch settings, and claims that Playground training is inexpensive on this hardware.

### dexterous-astra cube project

- Source: <https://github.com/jianglongye/dexterous-astra>.
- It uses a five-finger Wuji hand in MuJoCo, including a two-hand zero-shot scripted controller and learned single-hand turns and rolls.
- Transferable ideas: separate simulation, scripted zero-shot, policy, and checkpoint folders; treat a scripted controller as a legitimate zero-shot baseline.
- Non-transferable items: Wuji 20-DOF joint values and two-hand trajectories cannot be copied to a 16-DOF, four-finger LEAP Hand.

### General in-hand manipulation decomposition

- **Surprisingly Robust In-Hand Manipulation** (arXiv:2201.11503) shows that reusable contact skills can cover cube orientations. A face turn should be decomposed into sliding, pushing, rolling, twisting, pivoting, finger gaiting, and controlled release rather than one monotonic joint trajectory.
- **DROP** (arXiv:2409.14562) demonstrates that many rotations require one contact set for the first part of the turn and another contact gait for completion. This supports a two-stage face-turn sequence.
- **HORA / Rapid Motor Adaptation** (arXiv:2210.04887) shows that stable finger gaiting can emerge from simulation training with proprioceptive history. If a learning route is attempted later, begin with Z-axis rotation and proprioception rather than vision and full-orientation targets.
- Contact-mode planning examples with a LEAP Hand show that a fixed grasp cannot provide large rotation; the task is fundamentally a sequence of contacts being created and released.

### Grasp-pose sources

- **DexGraspNet 2.0** (arXiv:2410.23004) provides force-closure LEAP grasps. Any imported wrist and joint pose must be converted to Menagerie joint order and revalidated in MuJoCo.
- A **LEAP grasp cache** can provide stable initialization poses. Face turning should start from a validated cached grasp rather than from an open hand.

---

## 3. Staged project plan

### Phase 0 — Environment and model

- Verify Python and MuJoCo versions instead of assuming they match the old project.
- Fetch the complete Menagerie `leap_hand` folder with its assets.
- Load the model, print every joint and actuator, and verify joint ranges before writing target angles.
- Deliverables: `logs/phase0_check_only.txt` and the first `PROCESS_LOG.md` entry.

### Phase 1 — Stable held grasp

- Goal: pick the cube up from the table, hold it in free space, and maintain the grasp for 500 steps.
- Relaxed gate: sustained thumb plus at least two finger contacts, displacement below 15 mm, no penetration above 0.002 m, and no ejection.
- Strict gate: sustained thumb plus at least two fingertip contacts and displacement below 5 mm.
- Lift gate: cube bottom at least 0.08 m or center Z at least 0.105 m, with thumb and two-finger contact retained through the final hold.
- Change one variable at a time: begin with `power_wrap_B` / T3, scan mount Y, then mount X, and only then consider palm height, pitch, and roll.
- Phase 2 must not begin until Phase 1 passes.

### Phase 2 — Scripted single-face gait

Define the contact sequence before writing control code. A 90-degree Z-axis example is:

1. **Initial hold:** thumb, at least two fingers, and optionally the palm maintain a 500-step grasp.
2. **Guard:** one finger forms a lateral wall while the thumb supports the opposite side.
3. **Push:** another finger pushes or rolls the face around Z while guarding contacts may slide but do not disappear.
4. **Gait:** the pushing finger lifts, repositions, and establishes a new contact while the thumb and guarding fingers support the cube. This is the highest-risk stage.
5. **Finish:** all fingers return to a power grasp and hold for 500 steps.
6. Stop immediately on a drop, trap, or penetration above 0.002 m. Record which contact failed first, the step, and the geometry name.

The complete gate requires:

- A stable 500-step starting grasp.
- A 90-degree turn within a tolerance frozen before the first test.
- A stable 500-step ending grasp with no trapping.
- No hand-object or finger-finger penetration above 0.002 m.
- Two consecutive passes with the same values.
- Rotation measured from the cube or face joint state, never inferred from finger angles.

Deliverables: one uncut video covering closure, lift, turn, and final hold; a contact-sequence log; and a summary table.

### Phase 3 — Optional learning route

Only begin after the scripted Phase 2 trajectory proves that the task is mechanically feasible.

- Option A: adapt the Playground LeapCubeReorient approach to Z-axis-only proprioceptive control, initializing from the scripted sequence. Rewards should cover orientation error, drop and penetration penalties, torque/action-rate costs, and a stable final grasp.
- Option B: use DROP-style sampled planning with a partial turn followed by a contact change.
- Record available GPU hardware, simulator support, training steps, and time budget before making any training commitment.
- Deliverables: training curves, checkpoints, and validation against the same Phase 2 gates.

### Phase 4 — Documentation

- `PROCESS_LOG.md`: date, change, before/after measurements, observation, and keep/revert decision for every iteration.
- `FINAL_REPORT.md`: project facts only. Do not include task-assignment content, unearned titles, or untested VLA/MPC/learning claims.
- Scripted control must be described as scripted control, simulation as simulation, and a single-face turn as a single-face turn rather than a full cube solve.

---

## 4. Recommended project structure

```text
<project-root>/
  scenes/                 # Table, cube, and hand-base MuJoCo scenes
  poses/                  # One JSON per candidate, with mount and all 16 joints
  scripts/
    validate_poses.py     # Shared batch validator with angle measurement
    render_video.py       # Closure, lift, turn, and final hold
  logs/                   # Detailed evidence; only summary tables in reports
  videos/                 # Complete, truthful result videos
  PROCESS_LOG.md
  FINAL_REPORT.md
```

## 5. Required execution order

1. Phase 0: environment and model load check.
2. Phase 1: start from `power_wrap_B` / T3 and pass the grasp gate with the shared batch validator.
3. Phase 2: document the contact sequence, then implement and validate a scripted face-turn gait.
4. Discuss Phase 3 learning methods only after Phase 2 passes twice.

---

**One-sentence conclusion from the old project:**
Cube displacement and finger crossing had been solved, but thumb contact and a second supporting finger had not; the new project therefore had to pass the held-grasp gate first and then add a finger gait for a single-face turn.
