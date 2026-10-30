---
name: dexterous-astra-leap-method
description: Guide this MuJoCo desktop-Python LEAP articulated cube-face project with grasp-first scripted control, measurable gates, repeatability, and honest attribution.
---

# Dexterous-Astra — Distilled Method for a LEAP Project, Not a Code Copy

- Use this skill when guiding Sarah's LEAP in-hand project (MuJoCo, desktop Python, sim only) with the *method* of Jianglong Ye's dexterous-astra work.
- Source, verified 2026-10-30: profile https://github.com/jianglongye — its top repo is dexterous-astra, https://github.com/jianglongye/dexterous-astra, public, main branch, 2 commits, 115 stars, created 2026-09-17, commit 5e60ae21cd29866aefe3e2cffee9b260c2b0a7b1. Root contains only `assets`, `cube`, `pen`, `README.md`, `.gitignore`, `ruff.toml`. There is NO `LICENSE` file in the root and NO hammer folder — hammer, if it exists anywhere, is tweet/video-only. Do not claim a license, and do not claim hammer code exists.
- The profile's other popular repos (GIFS shape representation, implicit object tracking, continuous grasping function from human demonstrations, FeatureNeRF) are NOT distilled here. They are separate research lines. This skill covers only dexterous-astra, because that is the repo relevant to Sarah's project. If a future task needs one of the others, distill it separately.
- The repo's own note, verbatim in spirit: it contains final artifacts from Astra experiments, rather than a task-specific harness or benchmark provided to Astra, and it provides the zero-shot prompt used for pen spinning for reproducibility. Treat it as a reference artifact, not as a harness to install and claim as your own work.

## 1. What the repo actually contains — task table, not code to paste

| Task | Hand, simulator | Zero-shot | Trained |
| --- | --- | --- | --- |
| Pen spinning | Sharpa Wave hand, Isaac Lab | sampled finger-keyframe planner | PPO policy: 3 turns and a hold |
| Rubik's cube | Wuji hands, MuJoCo | two-hand scripted controller: two moves | one-hand learned turns and rolls: solves `U F L` |

- Every task has the same four folders: `sim/` (environment), `zero_shot/` (script), `policy/` (runs the checkpoints), `checkpoints/`.
- Run entry points, as documented in the README: `pen/run_zero_shot.sh`, `pen/run_policy.sh`, `cube/run_zero_shot.sh`, `cube/run_policy.sh`, each writing to a `runs/` output folder.
- Environment requirements, as documented: pen needs `ISAAC_PY` = Python 3.11 with Isaac Lab 2.3.2 and `rsl-rl-lib==3.0.1`; cube and pen Python otherwise use `pen/requirements.txt` or `cube/requirements.txt` as `PY`. No membership, login, subscription, or API-key step appears in the README/run instructions; the code runs locally. Pen needs an NVIDIA GPU for Isaac Lab; cube (MuJoCo) does not.
- Zero-shot prompt for pen spinning, as published in the README (quote it, do not rephrase it as your own prompt):
  > There's a pen simulator here. A robot hand lies palm-up with a pen on its fingers.
  > Spin it one full turn without dropping it, ending steady. No training.

## 2. Folder structure to borrow — structure, not files

- `pen/sim/`: `grasp.json`, `mj_scene.py`, `pen_env.py`, `render.py`, `spec.py`.
- `pen/zero_shot/`: `export_for_render.py`, `mpc_pen.py`.
- `cube/sim/`: `arm_ik.py`, `arm_planner.py`, `arms.py`, `physics.json`, `physics.py`, `render.py`, `run.py`, `scene.py`.
- `cube/zero_shot/`: `controller.py`, `grasp_planner.py`, `grasps.json`.
- `policy/` and `checkpoints/` in each task: policy code only runs a checkpoint; it does not contain the training harness.
- Lesson for Sarah's LEAP project: keep the same separation, with LEAP names, not Astra names:
  - `scenes/` = scene + physics parameters, separated from controller code (Astra's `physics.json` / `spec.py` pattern).
  - `poses/` = precomputed grasp poses as data files, separated from code (Astra's `grasp.json` / `grasps.json` pattern).
  - `scripts/` = one script per phase (validate, render, plan, control), not one monolithic script.
  - `logs/` + `videos/` = per-run evidence, kept out of the pose/controller code.
  - `PROCESS_LOG.md` = one entry per change, with measured numbers before/after.

## 3. The grasp-first pattern — the most transferable idea

- Both Astra tasks start from a precomputed, settle-validated grasp, they do NOT start rotation/spinning from an open hand or from the table.
- Pen pattern, distilled from `pen/sim/grasp.json` (values are Astra's Sharpa values, listed here only to show the *fields*, never to copy into LEAP):
  - Fields: a grasp label, an object label, a candidate id, a joint list, a joint array, an object position, an object orientation, a layout (including yaw), and a settle report.
  - Settle report fields: number of settle trials, valid fraction, tip count, thumb count, palm count, and a Z drop during settle. The grasp is only used if the settle report passes. A grasp file without a settle report is not a validated grasp.
- Cube pattern, distilled from `cube/zero_shot/`: a `grasp_planner` produces candidate grasps into `grasps.json`, and a separate `controller` executes from a chosen grasp. Planner and executor are separate files, so a bad grasp can be rejected without rewriting the controller.
- Transfer to LEAP, in Sarah's existing gate language:
  - A LEAP pose is only a "held pose" after it passes the new project's grasp gate in the NEW folder: thumb sustained + at least 2 fingers, displacement <0.015 m relaxed / <0.005 m strict, no penetration >0.002 m, 500-step hold, then a lift (cube bottom >=0.08 m / centre Z >=0.105 m, thumb + 2 fingers still in contact, no drop).
  - If the gate fails, the work is grasp work, not rotation work. This is exactly Astra's ordering: grasp file first, rotation/spinning second.
  - Do NOT copy Astra's joint arrays, candidate ids, object positions, or settle numbers into LEAP. LEAP is 16 DOF, 4 fingers; Sharpa is 22 DOF, 5 fingers; Wuji is 20 DOF, 5 fingers. Joint angles do not transfer 1:1, and a Sharpa/Wuji checkpoint does not run on a LEAP hand.

## 4. The zero-shot planner pattern — planner/actor split, scored sampling

- Pen zero-shot, distilled from `pen/zero_shot/mpc_pen.py` (method, not code):
  - It is a closed-loop, non-RL, sampled finger-keyframe planner, NOT a learned policy and NOT a single fixed trajectory.
  - Planner/actor split: the planner pauses simulation while it plans, then the actor executes a small number of real control steps (six in the Astra pen planner), then planning runs again. Do not plan and execute in the same step without a pause, or the timing in the log will not match the physics.
  - Sampling: a population of candidate keyframe sequences, evolved over a few generations, with a fixed seed, a fixed time budget, and a small number of keyframe knots (two in the Astra pen planner). Report population, generations, seed, and time budget for every run; a run without them is not reproducible.
  - Scoring, in Astra's form: reward net object rotation, plus a support term for finger force above a threshold (0.01 in the pen planner), minus penalties for penetration above a threshold (0.002 m), with a large negative score for a bad candidate. A candidate that drops the object, wedges it, or penetrates is not "almost good" — it is bad, and must be scored as such.
  - Precondition: the planner assumes the object is already held and asserts the initial grasp is valid before spinning. If the initial grasp is not valid, the planner must STOP, not spin anyway.
- Cube zero-shot, distilled from `cube/zero_shot/controller.py`:
  - It is a scripted controller, not a learned policy. It executes a small, fixed number of moves (two, in the Astra cube case) from a chosen grasp, in a fixed order, with the grasp planner separate from the controller.
  - Transfer: Sarah's scripted single face turn is the LEAP analogue of this rung — a scripted controller from a held grasp, before any learned policy. Label it "scripted / zero-shot" in every report. Never label it PPO, MPC, VLA, or a learned policy unless that phase actually ran.
- Ladder, in order, for Sarah's project (do not skip a rung):
  1. Scripted / zero-shot controller from a validated held grasp (this is the honest first demo).
  2. Sampled keyframe planner (Astra pen pattern, LEAP version) — only after the scripted controller shows the rotation is feasible.
  3. Learned PPO policy — only after step 2, and only if a GPU/MJX path is actually recorded as available. On a Mac without an NVIDIA GPU, do not promise PPO training as a free step.
  4. Full solve / pen spinning / two-hand work — OUT of scope for the current LEAP single-face-turn goal.

## 5. What to measure, every run — Astra's implicit checklist, made explicit

- Object rotation from object quaternion, not from finger angles. For a face turn, fix the tolerance before the first run (Sarah's current gate: 90 degrees +/- 10 degrees) and never move it after a failure.
- Start-held and end-held, each 500 steps, with thumb + finger contacts listed by exact geom name, tip vs non-tip classified from the geom name only.
- Displacement: report max AND end-of-hold displacement. A cube that looks stable at close and slides during hold is not stable (Sarah's old Prompt 12 V0: 5.118 mm at close, 30.900 mm at hold end).
- Drop / wedge / penetration: drop step and first lost contact, wedge check (object centre vs palm/finger gap), penetration >0.002 m at any step (finger-finger, hand-object, hand-table). Any one of these fails the run, even if rotation looks correct.
- Repeatability: the same sequence passes twice in a row before it is called a pass. One lucky run is not a pass.
- Video: render the real result, including drops. Do not cut out a drop, do not render a lift or rotation that did not happen.
- Logs: per-frame or per-step logs stay in files; only a summary table is pasted back. A summary without the underlying log file is not evidence.

## 6. Honesty and attribution rules — read this before using the skill

- Do NOT copy Astra code, joint arrays, grasp files, or checkpoints into Sarah's LEAP project, and do NOT present Astra results as Sarah's results. The task Jin assigned is a from-scratch re-implementation on a different hand.
- DO cite Astra in `PROCESS_LOG.md` and `FINAL_REPORT.md` in this form: "Method reference: dexterous-astra (Jianglong Ye), sim / zero-shot script / policy folder structure, grasp-first ordering, and sampled keyframe planner pattern. LEAP model, poses, controller, and all numbers were re-implemented and measured in MuJoCo in this project; no Sharpa/Wuji code or checkpoints were copied."
- Do NOT claim: pen spinning, two-hand cube work, hammer work, a full Rubik's solve, a learned policy, MPC/MPPI, VLA, hardware results, or sim-to-real transfer, unless that phase actually ran and was measured in Sarah's project.
- Do NOT tune friction, mass, solver, or object size to force a pass. Astra's physics parameters are in `physics.json` / `spec.py` for a reason: they are fixed before the run, not adjusted after a failure. If a LEAP run needs a physics change, log it as a scope change, re-run the grasp gate, and say so in the report.
- Do NOT delete the old LEAP project to "start clean" from Astra. The old folder contains the only measured evidence for displacement, anti-crossing, mink failure, and validator behaviour. Copy lessons, not files, unless a file is explicitly re-validated in the new folder.

## 7. How to use this skill in a Codex prompt

- Tell Codex, in the prompt, to read this skill first, then to follow Sections 2–5 in order: folder separation, grasp file + settle/gate first, scripted/zero-shot controller second, planner third, learned policy last.
- Require Codex to output, for every phase: the grasp file used, its gate result, the controller/planner parameters (population, generations, seed, time budget, if a planner is used), the summary table (rotation, displacement max/end, contacts, drop/wedge/penetration), and the log/video file paths. If any of those are missing, the phase is not complete.
- If Codex proposes to copy an Astra file, stop it and require a LEAP re-implementation with a LEAP gate result instead. If Codex proposes to start rotation from the table or from an open hand, stop it and require a validated held grasp first.
- This skill does not replace the LEAP in-hand skill (`local_leap_inhand_cube_turn`): that skill owns Sarah's gates, contact sequence, and folder layout; this skill owns Astra's method pattern and attribution. When they conflict on a gate value, Sarah's LEAP gates win, because they were measured in her project.

## 8. Current measured LEAP articulated-face checkpoint

- As of 2026-10-30, the TEST-only articulated-face path has independently passed grasp, 30 mm lift, and two deterministic follow-through keyframes. This is scripted/zero-shot control, not MPC or a learned policy.
- KF02 R4 ends at `9.539155665 degrees` and KF03 ends at `15.766142345 degrees`. Each keyframe passed two clean-state replays with sustained thumb/index/middle core support, no ring-core contact, no warning, and no penetration over 2 mm.
- KF03 uses ring deltas `rf_mcp +0.10`, `rf_pip -0.05`, and `rf_dip -0.10 rad` from the measured KF02 endpoint while keeping `rf_rot` fixed. Its final-100 contact rate is `100%`, maximum palm-relative drift is `0.112637 mm`, and final-100 angular range is `0.048485 degrees`.
- Continue single-ring deterministic follow-through while safe joint space and the frozen gates remain available. Introduce an index/middle gait only after the ring-only search produces no admissible candidate or reaches a joint/contact boundary; do not add cooperative variables preemptively.
- Evidence: `logs/prompt25_face_followthrough_validation_revision_3.txt`, `logs/prompt26_face_kf03_search.txt`, and `logs/prompt26_face_kf03_validation.txt`. The full 90-degree face-turn gate remains unpassed.
