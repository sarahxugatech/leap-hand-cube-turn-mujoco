# Process Log

## 2026-10-29 — Prompt 01 startup

- Scope: Phase 0 environment/model check and table/cube scene stability only.
- Starting cube values (not results): half-size 0.025 m; density 1000 kg/m³; mass 0.125 kg; friction 0.2; start position (0, 0, 0.075) m.
- Starting table value (not a result): table top Z = 0.05 m.
- Starting mount values (not results): position (-0.0425, -0.0070, -0.2190) m; pitch/roll (+0.10, -0.10) rad.
- No grasp, close, lift, move, place, face rotation, finger gaiting, learned policy, VLA, or MPC is permitted in this prompt.

### Environment result

- Python 3.12.14; pip 26.2.1; MuJoCo 3.17.0; NumPy 2.5.4.
- Installed `mujoco` and `numpy` wheel-only in `.venv`; `pip check` reported no broken requirements.
- `MUJOCO_PATH` was not set; `mujoco-py` and source-built MuJoCo were not used.
- The companion skill named by the phase document was missing at `/Users/tianhui/workspace/custom-skills/local_leap_inhand_cube_turn/SKILL.md`; the phase document was used directly as the fallback.

### Phase 0 model check result

- Official Menagerie sparse checkout revision: `f3b6013629c7b94f8ee93fd977b161fa8350f9bc`.
- `menagerie/leap_hand/scene_right.xml` loaded successfully with nq=16, nv=16, nu=16, nbody=18, ngeom=89.
- All 16 joint names/ranges, all actuator names, kp=3.0, kv=0.01, default friction 0.2, fingertip friction 0.5, elliptic cone, and impratio 100 matched the required checks. Full output: `logs/phase0_check_only.txt`.

### Table/cube stability result

- Rebuilt `scenes/table_cube.xml`; compiled cube mass is 0.125 kg.
- Mount holding gains were increased after an initial passive check showed 0.007318260 m gravity sag. Prescribed starting positions and all cube/table properties were unchanged. The final maximum mount drift was 0.000731826 m_or_rad.
- Video preview exposed a missing mount home-body origin. The mount body was corrected to Z=0.25 m, while the prescribed five mount joint values remained (-0.0425, -0.0070, -0.2190, +0.10, -0.10). This places the hand above the table instead of 0.25 m too low.
- Corrected final 2000-step / 4.0 s passive test: cube start (0, 0, 0.075) m; cube end (5.4407e-9, -9.7289e-9, 0.074892245) m; XY drift 1.1147e-8 m; Z change -0.000107755 m.
- Table support contact and incidental open-hand index-base/cube contact occurred. The deepest overlap was -0.000136719 m, below the 0.002 m penetration threshold; no solver/contact warnings occurred.
- Phase stability criterion: PASS. This is not a grasp result.
- Main scene SHA256: `1e1e7666992f8e2c82ed11c05723489df0f9614f5a345021be16360eadf7f2a1`.
- No grasp or rotation was attempted.

## 2026-10-29 — Phase 0 stability video

- Added `scripts/render_stability.py` and rendered `videos/phase0_scene_stability.mp4`.
- Video: 640×480, 30 fps, 4.0 s. Only the camera orbits; no grasp, lift, or rotation control is present.
- Video SHA256: `f8517bd5a43b2aa805148b85222f1a3c97718851f5fd57115f815c14ebd03dfa`.

## 2026-10-29 — Prompt 02 corrected-scene baseline

- Correction: `scenes/table_cube.xml` mount body changed from `<body name="mount">` to `<body name="mount" pos="0 0 0.25">`.
- Reason: the five mount values are joint coordinates relative to a Z=0.25 m home-body origin. Without that origin the complete hand was assembled 0.25 m too low, below the table; the video preview exposed the error.
- SHA256 transition: incorrect `b2bf12bafa80dca7b856d5dd5e19cc889f1ffb9f3f774a9415b9518837cea0ae` → corrected/frozen `1e1e7666992f8e2c82ed11c05723489df0f9614f5a345021be16360eadf7f2a1`.
- Corrected-scene passive stability rerun (2000 steps, 4.0 s): cube start (0, 0, 0.075) m; cube end (5.440749e-9, -9.728904e-9, 0.074892244580) m; XY drift 1.1147e-8 m; Z change -0.000107755420 m; mount maximum drift 0.000731826 m_or_rad.
- Open-pose palm centre: (-0.032566533460, 0.002983341665, 0.130003328892) m. Conservative hand lowest Z (geom centre minus bounding radius): 0.042003269557 m.
- Deepest contact distance: -0.000136718930 m (`cube_geom` / `if_bs_collision_2`); no penetration over 0.002 m and no solver/contact warning.
- Full baseline log: `logs/prompt02_corrected_scene_stability.txt`.

## 2026-10-29 — Prompt 02 grasp gate

- Imported old-folder `power_wrap_B.json` and `power_wrap_B_thumb_T3.json` read-only into `poses/`; semantic JSON comparison against both old files passed. B matched byte-for-byte; T3 differed only by a trailing blank line.
- Verified B/T3 mount: (-0.0425, -0.0070, -0.2190), pitch/roll (+0.10, -0.10). Finger rot values are if +0.40, mf 0.00, rf -0.40. T3 thumb is cmc/axl 0.000/0.000 and mcp/ipl 1.530/0.110.
- Added `scripts/validate_poses.py`: explicit 5 mount + 16 hand joint JSON validation, 600-step cosine close, 500-step hold, 100-step sustained-contact window, frozen-hash checks, temporary test scene only, detailed per-pose logs, and >0.002 m checks for finger-finger, hand-cube, and hand-table penetration. No lift path exists.
- Open self-check: max/end XY displacement 0.000/0.000 mm; no contact, no penetration, anti-crossing PASS.
- T3 re-proof: max/end XY displacement 2.018/2.016 mm; final contact YES with one non-tip finger; thumb ever/sustained NO; tip fingers 0; anti-crossing PASS; Gate A FAIL; Relaxed FAIL.
- Because T3 thumb remained NO, created Y1–Y5 with only mount_y changed to -0.0270, -0.0170, -0.0070, +0.0030, +0.0130 m. A programmatic lock check verified that all 16 hand joints and mount_x/mount_z/pitch/roll were unchanged.
- Mount-Y result: all five had thumb ever/sustained NO, anti-crossing PASS, no >0.002 m hand-cube or hand-table penetration, and Gate A/Relaxed FAIL. No lift was run.
- Required-order ranking selects Y1 because all thumb results tie at NO and Y1 has two any-finger contacts; its max/end displacement is 17.010/17.010 mm. It is not a grasp candidate because it has no thumb contact and exceeds the relaxed displacement threshold.
- Full summary: `logs/prompt02_grasp_gate_summary.txt`; detailed logs: `logs/validate_poses_*.txt`.
- Frozen main scene SHA256 before/after: `1e1e7666992f8e2c82ed11c05723489df0f9614f5a345021be16360eadf7f2a1`.

## 2026-10-29 — Prompt 03 mount-X only

- Re-read `poses/power_wrap_B_thumb_T3.json` and its Prompt 02 detailed log. Control values/results matched: mount (-0.0425, -0.0070, -0.2190), pitch/roll (+0.10, -0.10), max/end displacement 2.018/2.016 mm, one any-segment finger, final contact YES, thumb ever/sustained NO, tip fingers 0, anti-crossing PASS.
- Created X1–X5 with mount_x -0.0625, -0.0525, -0.0425, -0.0325, -0.0225 m. Programmatic lock check confirmed all 16 hand joints and mount_y/mount_z/pitch/roll exactly matched T3.
- One batch validator command ran all five variants using temporary test scene copies. All variants had thumb ever/sustained NO, tip fingers 0, anti-crossing PASS, no >0.002 m finger-finger/hand-cube/hand-table penetration, and Gate A/Relaxed FAIL. No lift was run.
- X4 and X5 ejected with end displacement 25.913 and 32.933 mm. X1/X2/X3 retained one finger with end displacement 16.454/8.269/2.016 mm respectively.
- Required-order ranking (thumb ever, thumb sustained, displacement, finger count) selects X3. It is not a grasp candidate because it has no thumb contact.
- Full summary: `logs/prompt03_mountX_summary.txt`; detailed logs: `logs/validate_poses_power_wrap_T3_mountX_X*.txt`.
- Frozen main scene SHA256 before/after: `1e1e7666992f8e2c82ed11c05723489df0f9614f5a345021be16360eadf7f2a1`.

## 2026-10-29 — Prompt 05 autonomous loop, Step A

- Added and ran `scripts/thumb_distance.py` on a temporary TEST scene copy with T3 mount and all 16 joints set exactly; forward kinematics only, zero physics steps, no lift.
- Thumb tip: (-0.020590486, -0.022320315, 0.152946122) m. Palm centre: (-0.032566533, 0.002983342, 0.130003329) m. Thumb base: (-0.112179176, -0.007894804, 0.160905695) m.
- Thumb closest cube surface: top, distance 0.052946122 m, closest-point delta DX/DY/DZ = (0.000000000, 0.000000000, -0.052946122) m.
- Finger-tip closest distances: index 0.015415534 m, middle 0.022641779 m, ring 0.032029115 m.
- Decision: |DZ| is the largest component. X/Y were already exhausted from T3 with thumb ever NO, so the next allowed variable is mount-Z.
- Main scene remained frozen at `1e1e7666992f8e2c82ed11c05723489df0f9614f5a345021be16360eadf7f2a1`.

## 2026-10-29 — Prompt 05 autonomous loop, variable 1 mount-Z

- Generated Z1–Z5 from T3 with only mount_z changed to -0.239, -0.229, -0.219, -0.209, -0.199 m; programmatic lock check passed for mount_x/mount_y/pitch/roll and all 16 hand joints.
- Batch validation: thumb ever/sustained NO for all five; tip fingers 0; anti-crossing PASS for all; Gate A/Relaxed FAIL for all. No lift.
- Z1/Z2 hand-cube penetration exceeded 0.002 m (-0.017764242/-0.007764242 m) and were rejected. Z3 reproduced the control at max/end 2.018/2.016 mm. Z4 max/end 10.488/10.464 mm. Z5 had no final hand contact at 4.434/4.434 mm.
- Decision: no refinement because thumb ever was NO in all initial variants. Logged `mount-Z: thumb ever NO in all variants` and advanced to pitch-only, with roll fixed, to preserve one-variable-at-a-time.
- Main scene remained frozen at `1e1e7666992f8e2c82ed11c05723489df0f9614f5a345021be16360eadf7f2a1`.

## 2026-10-29 — Prompt 05 autonomous loop, variable 2 pitch-only

- Generated P1–P5 from T3 with only mount_pitch changed to 0.00, 0.05, 0.10, 0.15, 0.20 rad; roll stayed -0.10 rad. Programmatic lock check passed for mount_x/mount_y/mount_z/roll and all 16 hand joints.
- Batch validation: thumb ever/sustained NO for all five; tip fingers 0; anti-crossing PASS for all; Gate A/Relaxed FAIL for all. No lift.
- P1 had no final hand contact at max/end 5.175/5.169 mm. P2/P3/P4 retained one finger at 2.266/2.249, 2.018/2.016, and 3.517/3.517 mm. P5 ejected at 29.035/29.034 mm and was rejected for hand-cube penetration -0.003922936 m.
- Decision: no refinement because thumb ever was NO in all initial variants. Logged `pitch: thumb ever NO in all variants` and advanced to TEST-scene-only yaw, the third and final allowed variable.
- Main scene remained frozen at `1e1e7666992f8e2c82ed11c05723489df0f9614f5a345021be16360eadf7f2a1`.

## 2026-10-29 — Prompt 05 autonomous loop, variable 3 TEST-scene yaw and STOP

- Extended `scripts/validate_poses.py` only for TEST-scene yaw poses: a temporary mount-origin Z-axis hinge/actuator is added to the disposable scene copy when a pose contains `mount_yaw_degrees`. The main scene was not edited. A no-yaw T3 regression reproduced max/end 2.018/2.016 mm before the yaw batch.
- Generated W1–W5 with TEST yaw -30°, -15°, 0°, +15°, +30°. The original five mount values and all 16 hand joints remained exactly T3; programmatic lock check passed.
- Batch validation: thumb ever/sustained NO for all five; tip fingers 0; anti-crossing PASS; no >0.002 m penetration; Gate A/Relaxed FAIL. No lift.
- W1 retained two fingers at 29.693/29.693 mm. W2 retained one at 57.792/55.425 mm. W3 retained one at 1.941/1.887 mm. W4 retained one at 0.295/0.126 mm. W5 ejected at 36.505/36.494 mm.
- Decision: no refinement because thumb ever was NO in all initial variants. Three variables are exhausted, so the loop stopped; no fourth variable, blind grid, or lift was run.
- Best by required order: W4 (+15° TEST yaw), thumb ever/sustained NO, max/end 0.295/0.126 mm, one non-tip finger. W4 is NOT a grasp candidate because thumb contact is NO.
- Final verdict: `NO GRASP CANDIDATE — thumb ever NO in all autonomous variants, next human decision needed: change hand orientation / thumb base geometry, or use a DexGraspNet / grasp-cache pose as the next source`.
- The first final-status apply_patch attempt failed because it tried delete/add on the same file; it made no change. The corrected single-update patch succeeded. Simulation results were unaffected.
- Main scene remained frozen before/after at `1e1e7666992f8e2c82ed11c05723489df0f9614f5a345021be16360eadf7f2a1`; old folder was read-only and untouched.
- Final read-only audit first failed because its generated lookup path used `mountZ_M1` instead of the actual `mountZ_Z1`; no files were changed. After correcting the audit path, all 15 pose files and detailed logs passed consistency checks: thumb ever YES count 0, Gate PASS count 0, lift run count 0, and all single-variable locks intact.

## 2026-10-29 — Prompt 06 official LEAP grasp-cache import

- Read the old project as reference only; it contained no verified grasp cache or pose that passed the current Gate A/Relaxed validator.
- Imported one pose from the official `leap-hand/LEAP_Hand_Sim` cache at commit `150bc3d4b61fd6619193ba5a8ef209f3609ced89`: `leap_hand_in_palm_cube_grasp_50k_s09.npy`, SHA256 `395ef1e0618e342d492d4f4c4cee881b5aea0a00d9cb9929363e54c857935b0e`, row 0. The source cache contains 1024 rows and was generated for a 0.0675 m cube; the current cube side remains 0.0500 m.
- Reconstructed the Isaac Gym simulation DOF order from the source repository's explicit `sim_to_real_indices` / `real_to_sim_indices`: index, thumb, middle, ring. Mapped it to the Menagerie order: index, middle, ring, thumb. All 16 mapped values passed the Menagerie joint-range checks.
- Added `scripts/import_leap_cache_pose.py` to preserve the source hand/cube relative transform and test the cube's 24 rotational symmetries without changing the cached joint angles. The first transform check exposed and corrected a duplicated Menagerie palm-local `pos="0 0 0.1" quat="0 1 0 0"` transform before validation.
- Selected cube symmetry 7 because it had the largest safe table clearance among equivalent orientations. Static target-pose distances were hand-table `+0.018779522 m` and hand-cube `+0.011932301 m`; the latter is consistent with the source cube being 17.5 mm wider than the current cube.
- Wrote the traceable mapped pose to `poses/leap_cache_s09_row0000.json`. Full source, mapping, and 24-orientation scan: `logs/prompt06_import_scan.txt`.
- Extended `scripts/validate_poses.py` only to accept an optional fixed palm-parent quaternion in disposable TEST scenes. The frozen main scene was not edited. A T3 regression before the imported-pose run reproduced max/end displacement `2.018/2.016 mm`, one final finger, thumb NO, and both gates FAIL.
- Imported-pose validation (600-step cosine close + 500-step hold): thumb ever/sustained NO; tip fingers 0; any fingers 0; final contact NO; max/end XY displacement `0.000/0.000 mm`; no ejection; no >0.002 m penetration; anti-crossing PASS; Gate A FAIL; Relaxed FAIL. Minimum hand-table distance during the trajectory was `-0.001689471 m`, below the 0.002 m rejection threshold. No solver warning occurred.
- Verdict: the official source pose is a verified grasp for its 6.75 cm source cube, but it is **not a grasp candidate for this project's 5 cm cube** because it produces no hand-cube contact after mapping and revalidation.
- No lift or rotation was run. `FINAL_REPORT.md` result fields remain TBD. Main scene SHA256 before/after remained `1e1e7666992f8e2c82ed11c05723489df0f9614f5a345021be16360eadf7f2a1`.

## 2026-10-29 — Prompt 07 mirrored 5 cm cache feasibility gate

- Used the public `Msornerrrr/in-hand-rotation-mjlab` 5 cm modified-left-LEAP cache at commit `0ab03eb2244b4a1a020d94ef38a4823bb9e15ba0`: `leap_left_custom_grasp_cache_cube5cm.npz`, SHA256 `a44b7e2af512215acc1750a4f1133f6a10315854c5e17a4c2e2f1cdc10e54eaf`.
- Selected source row 3388 deterministically as the first row nearest the current 0.025 m half-size; its cached half-size is `0.024919355 m` and its joint names already match the 16-joint Menagerie order.
- Applied the source repository's left/right sign convention to `if_rot`, `mf_rot`, `rf_rot`, and `th_axl`. Derived the fixed left/right frame reflection and translation from the three corresponding finger-base origins. Recorded a remaining 0.0285 m custom-thumb geometry mismatch as an explicit approximation risk.
- Initial static mapping exposed a missing MjLab entity-root offset. Source inspection showed that MjLab places the hand entity at `(0, 0, 0.1) m` in addition to the MJCF palm-local transform. With that offset included, a source-model diagnostic reproduced contact at approximately `-0.000205181 m`, confirming that the selected cache row and root convention are internally consistent.
- Added `scripts/import_mjlab_left5_pose.py` with an early static safety gate. It checks all 24 cube-equivalent orientations using the current right-hand geometry before writing any pose JSON.
- Corrected right-hand mapping result: every orientation penetrated the table by `0.016551632–0.071969417 m` and the cube by `0.002132434 m`. The four orientations inside the frozen mount ranges each penetrated the table by `0.026878352 m` and the cube by `0.002132434 m`.
- The importer stopped before writing a pose because no orientation satisfied mount limits and the 0.002 m penetration threshold. The stale pose created during the rejected pre-offset diagnostic was deleted. Full scan: `logs/prompt07_mirrored_import_scan.txt`.
- Structural conclusion: this cache represents a cube already suspended in an in-hand grasp. Preserving that relative grasp with the cube resting on the current table necessarily drives the wrapping fingers through the table. This cannot be corrected by a rigid mount/orientation change alone.
- No dynamic validator, lift, or rotation was run after the static rejection. Main scene SHA256 remained `1e1e7666992f8e2c82ed11c05723489df0f9614f5a345021be16360eadf7f2a1`; `FINAL_REPORT.md` remains TBD.


## Prompt 07 static rejection video

- Rendered `videos/prompt07_static_rejection.mp4` from the four mount-range-compatible mirrored orientations (1, 2, 8, 11).
- Video: 8.0 s, 640x480, 30 fps, H.264; SHA256 `34df6b5f856189b027d9a0e177a31bc652ec50e5ee9adc658287c3bc09178967`. This is a static diagnostic only; no physics stepping, grasp, lift, or rotation was attempted.
- The overlay reports the measured hand-table and hand-cube penetration for each rejected orientation.
- Frozen main scene SHA256 remained `1e1e7666992f8e2c82ed11c05723489df0f9614f5a345021be16360eadf7f2a1`.

## 2026-10-29 — Prompt 08 TEST-scene passive hold

- Added the TEST-only pose `poses/mjlab_left5_held_test.json` and `scripts/validate_held_pose.py`. The MjLab row-3388 mirrored relative pose was translated upward rigidly by 80 mm: cube centre `(0, 0, 0.155) m`, with the hand-cube relative transform unchanged.
- The cube remained a free body; there were zero equality constraints, no cube-hand attachment, and no cube qpos update during the 500-step / 1.0 s run. Hand and mount position actuators only held their initial targets.
- Initialization had only transient palm contact (`palm_collision_6`), no thumb/finger contact. All hand contact was lost after step 5; the cube fell `80.108 mm` back to the table. Maximum/end 3D displacement was `88.806/80.109 mm`.
- Deepest hand-cube distance was `-0.000310799 m`, below the 0.002 m penetration limit; there were no solver warnings. Gate A, Relaxed, and passive-hold stability all failed.
- Verdict: rejected; the main scene was not updated. Full result: `logs/prompt08_passive_hold.txt` and `logs/PROMPT08_STATUS.md`.
- Frozen main scene SHA256 before/after remained `1e1e7666992f8e2c82ed11c05723489df0f9614f5a345021be16360eadf7f2a1`.

## 2026-10-30 — Prompt 09 original modified-left initial-contact gate

- Added `scripts/validate_mjlab_left_source_hold.py` and rebuilt row 3388 in a disposable TEST scene using the source repository's `left_hand_custom.xml`, source cube half-size `0.024919355 m`, source cube mass `0.1 kg`, nominal shared slide friction `1.0`, and source timestep `0.005 s`.
- Corrected an omission in the earlier right-hand feasibility work: the complete cached cube pose was used, including its non-identity WXYZ quaternion `(0.237825811, 0.086552799, 0.021958888, 0.967194617)`, rather than position alone.
- Initial contacts were only `palm_collision_6` and `mf_bs_collision_1`; thumb contact was NO and only one non-thumb finger (`mf`) contacted the cube. Deepest hand-cube distance was `-0.000205187 m`, below the 0.002 m penetration limit.
- The frozen initial gate (thumb + at least two non-thumb fingers) failed, so the script ran zero dynamic steps. No attachment, equality constraint, lift, or rotation was used.
- Source-code inspection explains the mismatch: MjLab's cache reset loads cube size and pose but explicitly does not load robot joints. A cache row is not, by itself, a self-contained verified grasp under this project's contact gate.
- Gate A, Relaxed, and passive hold were not passed. Full result: `logs/prompt09_source_left_passive_hold.txt` and `logs/PROMPT09_STATUS.md`.
- Main scene was not updated; SHA256 before/after remained `1e1e7666992f8e2c82ed11c05723489df0f9614f5a345021be16360eadf7f2a1`.

### Prompt 09 static diagnostic video

- Added `scripts/render_prompt09_initial_gate.py` and rendered `videos/prompt09_initial_gate_failure.mp4`.
- Video: 6.0 s, 640x480, 30 fps, H.264; SHA256 `1ba5df8cf684552d28239aee6bb0d3d2034dd5d044dee7f8c3dfa1bb7c3a2345`.
- The yellow geoms mark the measured initial palm and middle-finger-base contacts. This is static visualization only: zero dynamics steps, no lift, and no rotation.
- Main scene SHA256 remained `1e1e7666992f8e2c82ed11c05723489df0f9614f5a345021be16360eadf7f2a1`.

## 2026-10-30 — Prompt 10 source reset joints with cache row 3388

- Extended `scripts/validate_mjlab_left_source_hold.py` with a `--joint-source source-reset` mode while preserving the default Prompt 09 cache-joint reproduction.
- Replaced row-3388 joint values with the source environment's exact fixed `GRASP_INIT_JOINT_POS`; kept the original left geometry, complete cached cube pose, cube size/mass, nominal friction, and solver settings unchanged.
- Initial contacts remained palm `palm_collision_6` plus middle-finger base `mf_bs_collision_1`. Thumb contact was NO and only one non-thumb finger (`mf`) contacted the cube.
- Minimum hand-cube distance was `-0.002797299 m`, exceeding the 0.002 m penetration limit. The initial contact gate failed, so zero dynamic steps were run; no attachment, lift, or rotation was attempted.
- Full result: `logs/prompt10_source_left_passive_hold.txt` and `logs/PROMPT10_STATUS.md`.
- Main scene was not updated; SHA256 before/after remained `1e1e7666992f8e2c82ed11c05723489df0f9614f5a345021be16360eadf7f2a1`.

## 2026-10-30 — Prompt 11 full 5 cm cache static scan

- Added `scripts/scan_mjlab_left_cache.py` and scanned all 6,764 cached cube sizes/full WXYZ poses against the original modified-left hand at the fixed source `GRASP_INIT_JOINT_POS`.
- The scan performed static forward/collision checks only. Finger-finger minimum distance was `+0.010000000 m`, so the fixed hand pose passed the self-crossing check.
- All rows contacted the hand; 11 contacted the thumb and 2,359 contacted at least two non-thumb fingers, but the sets had no intersection. Raw thumb-plus-two-finger gate passes: 0. Penetration-safe rows: 2,268. Final static candidates: 0.
- Rows 4482 and 5119 were penetration-safe thumb contacts with zero non-thumb fingers. Row 5760 had thumb plus middle finger but no second finger and exceeded the penetration limit at `-0.003264360 m`.
- No dynamic validation, lift, or rotation was run. Full 6,764-row output: `logs/prompt11_cache_static_scan.txt`; status: `logs/PROMPT11_STATUS.md`.
- Main scene remained unchanged at SHA256 `1e1e7666992f8e2c82ed11c05723489df0f9614f5a345021be16360eadf7f2a1`.

## 2026-10-30 — Project method skill adopted

- Added the user-provided Dexterous-Astra method distillation as project-local `SKILL.md` so future phases use the same grasp-first ordering and evidence rules.
- Method reference: dexterous-astra (Jianglong Ye), sim / zero-shot script / policy folder structure, grasp-first ordering, and sampled keyframe planner pattern. LEAP model, poses, controller, and all numbers were re-implemented and measured in MuJoCo in this project; no Sharpa/Wuji code or checkpoints were copied.
- Current phase remains grasp work: no pose has passed thumb-sustained + two-finger contact, 500-step hold, penetration, and lift gates. Therefore scripted rotation, sampled planning, and learned policy remain blocked.
- The next implementation must be a LEAP-specific grasp planner/optimizer that writes candidate pose data separately from the validator/controller. Astra joint arrays, grasp files, controllers, and checkpoints will not be copied.

## 2026-10-30 — Prompt 12 project-native grasp optimization and Phase 1 gate PASS

- Added `scripts/plan_grasp_candidates.py`, a reproducible CEM-style static sampler over the 16 LEAP joints plus initial cube XYZ. Fixed parameters: seed 12030, population 768, 12 generations, elite fraction 0.10, and 45 s time budget. Mount, cube quaternion, cube size, mass, friction, and solver remained frozen. This is sampled grasp search, not MPC and not a learned policy.
- The first objective admitted apparent support contacts that failed the complete penetration gate. The objective was corrected to penalize contact-derived whole-hand/cube and cross-finger penetration during sampling, then rerun with the same seed and budget. The corrected run produced five static candidates within the 0.002 m penetration limit.
- All five static candidates were independently tested for 500 steps with a free cube, zero equality constraints, no attachment, and no cube-qpos updates. They were safe but initially lost thumb contact by steps 4–15, so none passed.
- Added `scripts/refine_grasp_hold.py` and locally refined the lowest-displacement candidate C02 with dynamic screening. Fixed parameters: seed 12031, population 256, 10 generations, elite fraction 0.10, and 100 screen steps. Five candidates passed the short screen.
- Independent 500-step validation of R01–R05: all five passed Relaxed and passive-hold stability, with sustained thumb plus final index and middle contact and no penetration over 0.002 m. Gate A remained FAIL because the contacts were not two fingertip contacts.
- Best held pose: `poses/grasp_hold_refined_R01.json`. Maximum/end displacement `0.435/0.261 mm`; Z change `+0.063 mm`; final contacts palm + index base + middle base + thumb distal; minimum hand-cube distance `-0.001605751 m`; no warnings.
- Added `scripts/validate_lift.py` and tested R01 with a 30 mm cosine mount-Z lift over 300 steps followed by a 500-step hold. The free cube rose `30.063 mm`; maximum/end palm-relative drift was `1.219/0.971 mm`; thumb remained sustained, index and middle remained in contact, penetration stayed within the limit, and the lift gate PASSed.
- No cube rotation was attempted in this phase. The next phase is the scripted/zero-shot contact sequence with a frozen `90 degrees +/- 10 degrees` orientation gate, followed by two consecutive passing runs before any sampled rotation planner or learned policy.
- Main scene stayed unchanged before/after at SHA256 `1e1e7666992f8e2c82ed11c05723489df0f9614f5a345021be16360eadf7f2a1`.

## 2026-10-30 — Prompt 13 Phase 2 first scripted turn and sampled-keyframe stop

- Froze the rotation gate before running: world-Z cube rotation `90 degrees +/- 10 degrees`, start/end 500-step held gates, no penetration over 0.002 m, no drop/wedge, and two identical complete passes required. Wrote the intended contact sequence first in `logs/PROMPT13_CONTACT_SEQUENCE.md`.
- Scripted run 01 used a thumb tangential sweep while palm/index/middle supported the cube. It remained safe and ended held, but produced only `-0.018754 degrees`; the thumb lost contact during the sweep. This sequence was rejected as ineffective.
- Scripted run 02 established a ring guard and applied opposing index/middle rotation commands. It demonstrated real Z torque (`+5.348231 degrees`) but was rejected for finger-finger penetration `-0.003216152 m` and palm-relative drift `0.020414532 m`.
- After two manual structural attempts, stopped manual tuning and added the separate offline sampled keyframe planner `scripts/plan_turn_keyframes.py`. Fixed planner parameters: seed 13001, population 24, 6 generations, elite count 5, 45 s budget; actual elapsed 26.048 s. This was sampled keyframe planning, not MPC and not a learned policy.
- The planner produced two safety-screened data files. Independent actor verification of `poses/turn_keyframes_K01.json` produced `+1.540210 degrees`, tilt `0.043862 degrees`, drift `3.457 mm`, sustained thumb plus index/middle/ring end contact, and no penetration over 0.002 m. It was safe but failed the 90-degree rotation gate.
- Repeating the exact K01 gait showed it does not accumulate: two cycles safely ended at `+1.591946 degrees`, nearly identical to one cycle. A 10-cycle diagnostic was rejected for `-0.002129619 m` self-penetration and `30.105 mm` drift. After adding first-hard-failure termination, the repeated gait stopped at step 3060 (cycle 3 release) when self-penetration reached `-0.002002755 m`.
- Final Phase 2 verdict for this contact sequence: **FAIL / saturated at about 1.59 degrees**. The held grasp and lift remain valid; the 90-degree turn is not achieved and no success or repeatability pass is claimed.
- Rendered the real two-cycle failed result without hiding the failure: `videos/prompt13_K01_two_cycle_failure.mp4`, SHA256 `a9fe9b2be1fc50d1ecac6ac3ff894d6fb13e3b866b6f7cae0b66ca116ced6c8e`.
- Main scene stayed unchanged at SHA256 `1e1e7666992f8e2c82ed11c05723489df0f9614f5a345021be16360eadf7f2a1`. No attach, cube-qpos update during a run, MPC, VLA, or learned policy was used.

## 2026-10-30 — Prompt 14 turn-ready contact-mode transition PASS

- Kept the validated R01 held pose, 30 mm lift, cube physics, solver, and main scene frozen. Defined a turn-ready gate before sampling: sustained thumb, sustained two-or-more non-thumb supports, and at least one sustained non-thumb distal side contact whose contact point is at least 5 mm above the cube bottom; maximum palm-relative drift below 15 mm; no drop or penetration over 2 mm.
- Added the separate planner `scripts/plan_turn_ready_grasp.py`. Fixed parameters: seed 14001, population 64, 8 generations, elite count 8, 45 s budget, 300-step dynamic transition, and 100-step screening hold. Actual elapsed time was 22.970 s. The planner sampled the 12 non-thumb LEAP targets; thumb, mount, cube state, and physics were fixed. This was sampled contact-mode planning, not MPC and not a learned policy.
- The planner wrote three data-only candidates: `poses/turn_ready_T01.json` through `T03.json`. Added the independent actor/validator `scripts/validate_turn_ready.py`, which rebuilt R01, held 500 steps, lifted 30 mm, replayed the 300-step transition, and then held each candidate for 500 steps.
- T01, T02, and T03 all independently passed the turn-ready gate. Every candidate sustained thumb, index, middle, and ring contact; the new side contact was `rf_ds_collision_3`; there was no drop, warning, or penetration over 2 mm.
- Best candidate T01: maximum/end cube displacement `0.132/0.117 mm`; maximum/end palm-relative drift `0.113/0.112 mm`; minimum hand-cube distance `-0.000116970 m`; no finger-finger or hand-table contact. Final contacts were palm, index base, middle base, ring distal, and thumb distal.
- Rendered the real T01 transition and 500-step hold: `videos/prompt14_turn_ready_T01_transition.mp4`, SHA256 `f0d7d6f5bbf5d11f4f2c1054a15ab98467d1aa235c85167061b71e2fdfa5a53d`.
- No rotation was attempted in Prompt 14. Main scene remained unchanged at SHA256 `1e1e7666992f8e2c82ed11c05723489df0f9614f5a345021be16360eadf7f2a1`.

## 2026-10-30 — Prompt 15 two-stage scripted turn FAIL and orientation stop

- Froze a deterministic two-stage sequence before run 01: start from validated T01, use the ring distal contact as a tangential pusher, release/reposition it, repeat, then hold 500 steps. The target remained world-Z `90 degrees +/- 10 degrees`, with thumb plus two non-thumb supports throughout and the existing 2 mm penetration limit.
- Run 01 swept `rf_mcp` from the T01 value to 1.65 rad. It stopped at step 2118, 318 steps into the first sweep, when contact fell to thumb plus middle only. Measured rotation at stop was `+1.235781 degrees`; drift was `0.605 mm`; no penetration or drop occurred. The run failed the support and rotation gates.
- Added `scripts/check_side_contact_reachability.py`. At the exact validated T01 state it sampled the full four-joint ranges independently for index and middle using 20,000 deterministic samples each (seeds 15001 and 15002). It found 40 safe index distal-side configurations and 38 safe middle configurations. The maximum-margin index candidate used `if_tip` at `-0.000086316 m` contact depth.
- Wrote `poses/turn_ready_T04_index_tip.json` from that maximum-margin index sample and independently replayed the R01-to-T04 transition plus 500-step hold. T04 passed with sustained thumb, index-tip side contact, and middle support; maximum/end palm drift `2.852/1.733 mm`; no penetration. The ring contact was lost, so T04 alone was not sufficient for safe finger release.
- Extended the existing turn-ready planner with an explicit dual-side mode instead of blind tuning. Fixed parameters: seed 15003, population 64, 8 generations, elite count 8, 45 s budget; actual elapsed 26.495 s. It produced `dual_side_D01` through D03 with simultaneous `if_tip` and `rf_ds_collision_1` side contacts. All three independently passed a 500-step hold with thumb and three non-thumb contacts and no penetration.
- Run 02 used D01 and swept `rf_mcp` from 1.563941 to 1.900 rad in two stages with release/reset/recontact. It completed safely, but the ring left contact during each sweep; final rotation was only `-0.014108 degrees`, tilt `0.085005 degrees`, maximum palm drift `0.195 mm`, and end hold PASS. Rotation gate FAIL.
- Two structurally different two-stage attempts were exhausted. Stop-loss decision: do not tune or grid-search this contact sequence further. Under the current mount orientation, ring MCP motion moves the distal geom away from the cube after contact loads, so it cannot transmit sustained tangential torque.
- Rendered the real run-02 failure: `videos/prompt15_two_stage_run02_dual_side.mp4`, SHA256 `bc672ec67bb4ba1eb93f5998432d9768a4a333d21c684baf2b29ebbddc938785`.
- Next method change: alter the hand mount orientation in a TEST scene so a controllable distal joint has a tangential path along the cube face, then re-run grasp, lift, and turn-ready gates from the beginning. No 90-degree success is claimed.
- Main scene stayed unchanged at SHA256 `1e1e7666992f8e2c82ed11c05723489df0f9614f5a345021be16360eadf7f2a1`; no attach, cube-qpos update during a run, MPC, VLA, or learned policy was used.

## 2026-10-30 — Prompt 16 TEST-only mount-orientation redesign and torque probe

- Added `scripts/plan_oriented_grasp.py`, a deterministic static CEM search over 16 LEAP joints, mount pitch/roll/yaw, and initial held cube XYZ. Fixed parameters: seed 16001, population 1,024, 14 generations, elite fraction 0.08, and 45 s budget; actual elapsed time was 3.821 s. The mount orientation had to differ from the old orientation by at least 10 degrees. This was sampled search, not MPC or a learned policy.
- The full static gate required thumb contact, at least two non-thumb contacts, at least one distal side contact, and no hand-cube, hand-table, or cross-finger penetration over 2 mm. One candidate, `poses/oriented_grasp_O01.json`, passed.
- Independent 500-step validation of O01 used a free cube, no equality constraint, no attach, and no cube-qpos update during the run. Mount yaw was `-14.316187 degrees`. Maximum/end displacement was `12.609/12.391 mm`; the thumb stayed sustained and middle/ring contacts remained; minimum hand-cube distance was `-0.001064202 m`; no warnings or excessive penetration occurred. Relaxed and passive-hold stability gates PASSed; strict Gate A failed because there were not two fingertip contacts.
- Independent 30 mm lift validation PASSed. Cube rise was `20.573 mm`; maximum/end palm-relative drift was `11.792/11.732 mm`; thumb plus middle/ring support remained; minimum hand-cube distance was `-0.001320928 m`; there were no warnings or excessive penetration.
- Added `scripts/probe_oriented_turn_torque.py` and tested each of the 12 non-thumb joints at `+/-0.10 rad` from the lifted O01 state, using a 150-step cosine ramp plus 100-step hold. Each branch stopped on sustained support loss, 15 mm drift, or penetration over 2 mm. Twenty of 24 probes completed safely.
- Best safe total-angle direction was `rf_rot -0.10 rad`: geodesic angular displacement `5.604469 degrees`, maximum translation `1.787 mm`, with thumb, middle, ring, and palm contacts retained. A subsequent axis-metric audit using the frozen Prompt 13 relative-quaternion calculation measured only `+2.510023 degrees` world-Z yaw and `5.065863 degrees` tilt. This proves torque transfer but is not an acceptable target-axis turn direction.
- Rendered the same oriented-grasp, 30 mm lift, and torque-probe sequence as `videos/prompt16_oriented_grasp_torque_probe.mp4`: 13.0 s, 640x480, 30 fps, H.264; replay geodesic measurement `5.643267 degrees` and `1.767 mm`; SHA256 `a6e63399622bed37be7cc32ba626eba16ff14c0fcb0141e83a9c29a276acb8f7`. The overlay explicitly labels it as a small probe rather than a 90-degree success.
- The next gate was corrected to use world-Z yaw and tilt, not geodesic angle alone. A reset gait must not be built from this direction unless the target-axis metric improves.
- Main scene stayed unchanged at SHA256 `1e1e7666992f8e2c82ed11c05723489df0f9614f5a345021be16360eadf7f2a1`; no attach, cube-qpos update during a run, MPC, VLA, or learned policy was used.

## 2026-10-30 — Prompt 17 bounded amplitude scan and axis-metric correction

- Added `scripts/scan_oriented_rf_rot_amplitude.py` and independently replayed amplitudes from `-0.10 rad` through the exact `rf_rot` lower joint limit. Each run rebuilt O01, lifted 30 mm, and stopped on sustained support loss, 15 mm displacement, warnings, or penetration over 2 mm.
- All six amplitudes completed safely. `rf_rot` started at `-0.904290031 rad`, leaving only `0.142709969 rad` before its `-1.047 rad` lower limit.
- Under the corrected frozen metric, `-0.10 rad` was best: world-Z yaw `+2.512805 degrees`, tilt `5.050316 degrees`, geodesic angle `5.591909 degrees`, and maximum translation `1.776 mm`. At the joint limit, yaw decreased to `+2.249993 degrees` while tilt increased to `7.076657 degrees` and translation to `3.029 mm`.
- Verdict: **safe torque transfer, wrong-axis dominated**. Do not construct a repeated reset gait from this action. The next planner revision must directly maximize world-Z yaw while penalizing tilt, then re-run hold and lift gates for any new mount orientation.
- No 90-degree turn was attempted or claimed. Main scene SHA256 remained `1e1e7666992f8e2c82ed11c05723489df0f9614f5a345021be16360eadf7f2a1`.

## 2026-10-30 — Prompt 18 target-axis mount-orientation search

- Added `scripts/plan_target_axis_grasp.py` with fixed seed 18001, population 64, 8 generations, elite count 8, and 45 s budget. Variables were limited to mount pitch/roll/yaw, the four ring joints, palm-relative cube XYZ offset, and `rf_rot` amplitude; thumb/index/middle joints and cube physics remained fixed. This was dynamic sampled search, not MPC or a learned policy.
- Corrected two planner-screening errors before judging the result: the candidate now runs the real settle-then-300-step-lift sequence instead of starting directly at lifted height, and mount-orientation variants co-transform the cube from the fixed palm-relative grasp pose instead of leaving it at an incompatible world pose.
- Applied the hard support gate only during the turn probe, matching the independently validated startup/hold definition; penetration and 15 mm palm-relative drift limits remained active throughout.
- A deterministic 7x7 pitch/roll grid plus the local dynamic search found no candidate satisfying `abs(world-Z yaw) >= 1 degree` and `abs(yaw) > tilt`. The best safe grid state used pitch `-0.233333 rad`, roll `-0.116667 rad`, produced yaw `-3.274819 degrees`, tilt `3.765448 degrees`, and 13.575 mm maximum palm-relative drift. It was close but correctly rejected because tilt still exceeded yaw.
- Stop-loss decision: do not relax the axis gate or expand blind 16-joint sampling. The existing pitch/roll/yaw joint range is insufficient for a clean target-axis direction in this local grasp family. The next geometry change is a TEST-only fixed base quaternion, with cube pose co-transformed to preserve the grasp, followed by fresh hold and lift validation.
- No candidate pose was promoted, and no 90-degree turn was attempted or claimed. Main scene SHA256 remained `1e1e7666992f8e2c82ed11c05723489df0f9614f5a345021be16360eadf7f2a1`.

## 2026-10-30 — Prompt 19 fixed-base quaternion axis alignment

- Added `scripts/scan_fixed_base_quaternion.py`. It preserved the O01 cube pose relative to the palm, kept all O01 finger joints fixed, used the best Prompt 18 mount pitch/roll, and scanned TEST-only fixed base pitch/roll quaternions. Every case replayed settle, 30 mm lift, and the `rf_rot` probe with the existing support, 15 mm palm-relative drift, warning, and 2 mm penetration stops.
- The coarse 7x7 grid produced no pass. A pre-declared final local refinement used 2.5-degree base-angle steps around the two closest coarse samples and the already-safe `rf_rot` amplitudes `-0.10`, `-0.12`, and `-0.142709969 rad`. Total cases: 154.
- Thirteen screen cases across eight distinct base orientations passed the target-axis condition `abs(world-Z yaw) >= 1 degree` and `abs(yaw) > tilt`. Best screen result used base pitch/roll `0/10 degrees` and probe `-0.12 rad`: yaw `-1.032353 degrees`, tilt `0.772982 degrees`, geodesic angle `1.291139 degrees`, and 14.745 mm palm-relative drift.
- Independent verification rejected all three distinct promoted orientations. Q01 and Q02 passed the 500-step Relaxed hold but their 30 mm lift produced only `16.788` and `16.902 mm` cube rise, below the frozen 20 mm lift gate. Q03 lost middle-finger contact in the 500-step hold (Relaxed FAIL) and produced only `17.025 mm` rise. All remained free bodies with no attach, no run-time cube-qpos updates, no warnings, and no penetration over 2 mm.
- Stop-loss verdict: **axis alignment PASS in short screening, grasp/lift gate FAIL in independent validation**. Do not continue this O01 + `rf_rot` fixed-base family or run a rotation actor from Q01–Q03. The next structural change must jointly optimize support-force margin and target-axis torque instead of preserving O01 finger joints.
- Rendered the independently measured Q01 lift failure as `videos/prompt19_Q01_lift_gate_failure.mp4`: 9.0 s, 640x480, 30 fps, H.264; commanded hand lift 30 mm, measured cube rise `16.788440 mm`, lift gate FAIL; SHA256 `7a7b6d2215306a8ac2a351df2f48e5922ea4c0c146c05f2db2c1a1b70a03d968`. No rotation segment was added because the lift prerequisite failed.
- No 90-degree turn was attempted or claimed. Main scene SHA256 remained `1e1e7666992f8e2c82ed11c05723489df0f9614f5a345021be16360eadf7f2a1`.

## 2026-10-30 — Prompt 20 joint support/target-axis refinement

- Added `scripts/refine_axis_support.py`, a deterministic dynamic CEM refinement around Prompt 19 Q01. Fixed parameters: seed 20001, population 48, 10 generations, elite count 6, and 45 s budget. Variables were all 16 LEAP joints plus a palm-relative cube XYZ offset; the Q01 TEST-only base quaternion, mount pose, cube physics, and `rf_rot -0.12 rad` probe were fixed.
- Every sample replayed settle, 30 mm lift, post-lift hold, and the target-axis probe. The combined hard gate required at least 20 mm cube rise, `abs(world-Z yaw) >= 1 degree`, `abs(yaw) > tilt`, less than 15 mm palm-relative drift, probe-stage thumb plus two-finger support, and no warning or penetration over 2 mm.
- Initial objective run completed all 10 generations in 27.589 s and produced no combined pass. Its closest safe sample reached `19.163 mm` rise (0.837 mm below gate), yaw `+3.734 degrees`, tilt `3.514 degrees`, and 10.653 mm drift.
- Per the pre-declared local refinement rule, the objective was revised once to prioritize lift margin over excess yaw, with the same seed, bounds, population, generations, and gates. The rerun completed in 27.019 s and again produced no candidate. Its best lift/axis tradeoff reached `18.573 mm` rise, yaw `-4.424 degrees`, tilt `3.725 degrees`, and 11.425 mm drift.
- Stop-loss verdict: **Q01 fixed-base ring-pusher family has a structural lift/axis tradeoff**. Do not tune weights or expand sampling further. The next structural change must use a different pushing finger or a new high-margin base grasp, then re-run hold, lift, and target-axis gates.
- Replayed the final fixed-seed objective revision solely to preserve its best safe axis near-miss as `poses/axis_support_near_N01.json`, explicitly marked `near_miss: true` and probe `pass: false`. Independent 500-step validation passed Relaxed hold with 9.995 mm maximum displacement and no excessive penetration. Independent lift produced `19.969417 mm` cube rise—only `0.030583 mm` below the frozen 20 mm gate—with 9.606 mm maximum palm-relative drift, sustained thumb plus middle/ring support, and no warning or excessive penetration. It remains a strict lift FAIL and was not promoted.
- Rendered that independent near-miss lift as `videos/prompt20_axis_support_near_lift_failure.mp4`: 9.0 s, 640x480, 30 fps, H.264; SHA256 `bd0c28fb8f5020c4a0a86bea90ddbb525332fb6a6fe88782ceceef451d707d00`. The video labels the exact failure and contains no rotation segment.
- No pose was promoted and no 90-degree turn was attempted or claimed. Main scene SHA256 remained `1e1e7666992f8e2c82ed11c05723489df0f9614f5a345021be16360eadf7f2a1`.

## 2026-10-30 — Prompt 21 high-margin R01 / middle-finger target-axis probe

- Returned to the high-margin `grasp_hold_refined_R01` instead of continuing the Prompt 19/20 fixed-base ring-pusher family. Re-ran the independent 500-step hold and 30 mm lift. Hold remained Relaxed PASS; lift rose `30.063160 mm` with `1.218986/0.970637 mm` maximum/end palm-relative drift, sustained thumb plus index/middle support, and no warning or excessive penetration.
- Generalized the existing TEST-only small-joint probe to accept a pose, output log, amplitude, single-joint filter, and an optional validated transition. The default Prompt 16 behavior remains available; no new dependency or scene change was introduced.
- Scanned all 12 non-thumb joints in both directions at `0.10`, `0.20`, and `0.30 rad`. No branch met the predeclared promotion gate of at least 3 degrees world-Z yaw with at most 2 degrees tilt while retaining support. The clean R01 direction was middle-finger rotation: `mf_rot +0.60 rad` produced `-1.915586 degrees` yaw, `0.084147 degrees` tilt, and `2.200 mm` displacement with thumb plus index/middle support.
- Replayed the real R01 30 mm lift, the validated 300-step R01-to-T01 transition, and a 500-step hold before probing `mf_rot`. The best clean T01 probe was `+0.60 rad`: yaw `-2.365691 degrees`, tilt `0.044256 degrees`, and `1.078 mm` displacement. It retained thumb plus index/middle but lost ring at the endpoint. At `+0.80 rad`, ring support returned but yaw fell to `-2.118956 degrees`; `+1.00 rad` failed sustained support.
- Stop-loss verdict: amplitude alone cannot reach the 3-degree promotion gate. Do not build a reset gait yet. The next structural change must optimize T01 ring-support/contact placement around the middle-finger pusher, then independently repeat hold, lift, and target-axis gates.
- Detailed logs are `logs/prompt21_R01_joint_probe*.txt`, `logs/prompt21_R01_mf_rot_*.txt`, and `logs/prompt21_T01_mf_rot_*.txt`; summary is `logs/PROMPT21_STATUS.md`.
- No 90-degree turn was attempted or claimed. Main scene SHA256 stayed `1e1e7666992f8e2c82ed11c05723489df0f9614f5a345021be16360eadf7f2a1`; no attach, cube-qpos update during a run, learned policy, MPC, or VLA was used.

## 2026-10-30 — Prompt 22 actual articulated face task model

- The user confirmed that the target is a real face-layer turn, not whole-rigid-cube reorientation. The frozen single-body scene was therefore retained as historical evidence and was not modified.
- Added TEST-only `scenes/table_articulated_face_test.xml`. The minimal task model keeps the cube as a free body and divides its geometry into a `0.083333333 kg` core plus a `0.041666667 kg` top face layer, preserving the original total `0.125 kg` mass and 50 mm outer envelope at the aligned state.
- The face layer uses the unactuated `face_turn` hinge about local Z with a `+/-95 degree` range, damping `0.002`, and friction loss `0.01`. This is a deliberately minimal one-face mechanism, not a complete 27-cubie Rubik's Cube and not a full-solve model.
- Added `scripts/check_articulated_face_scene.py` and ran a 1,000-step/2-second table stability check. The free cube drifted at most `0.119340 mm`; the face joint drifted `0.000115145 rad` (`0.006597 degrees`); there were no MuJoCo warnings. Passive stability PASSed.
- TEST scene SHA256: `baaf5fc65a93bc11ec12b9407e24c70568ed3d5f7bdb00916617b48f18fdacbb`. Detailed output: `logs/prompt22_articulated_face_scene_check.txt`.
- No grasp, lift, or dynamic face turn was attempted. The next required rung is to migrate the R01 grasp state to the articulated object and independently repeat the 500-step hold and lift gates before writing a face-turn actor.
- Frozen main scene SHA256 before/after remained `1e1e7666992f8e2c82ed11c05723489df0f9614f5a345021be16360eadf7f2a1`; no attachment, runtime object-qpos update, learned policy, MPC, or VLA was used.

## 2026-10-30 — Prompt 23 articulated-object R01 grasp migration PASS

- Added the data-only `poses/articulated_grasp_R01.json`, transferring the measured R01 mount, 16 joint targets, and initial free-body pose to the aligned articulated object with `face_turn=0`. The old single-body PASS was not assumed.
- Added `scripts/validate_articulated_grasp.py`. It first runs a 500-step free-object hold and only after that gate passes performs a 30 mm cosine mount lift over 300 steps plus a 500-step suspended hold. It measures core motion, face-joint drift, core-versus-layer hand contacts, thumb/two-finger support, palm-relative drift, cross-finger/hand-object/hand-table penetration, and warnings.
- Ran the exact hold-then-lift sequence independently twice and preserved both logs. Both runs were deterministic and identical.
- Hold result: maximum/end core displacement `0.437866/0.272997 mm`; sustained thumb plus index/middle; maximum passive face drift `0.000481910 rad` (`0.027611 degrees`); no excessive penetration or warning; HOLD_GATE PASS.
- Lift result: core rise `30.000466 mm`; maximum/end palm-relative drift `0.518315/0.096846 mm`; sustained thumb plus index/middle; maximum face drift `0.000059913 rad` (`0.003433 degrees`); LIFT_GATE PASS.
- Final contacts were palm, index base, middle base, and thumb distal on the core. There was no final hand contact on the face layer, so the validated support grasp does not lock the movable layer.
- Evidence: `logs/prompt23_articulated_grasp_R01_run01.txt`, `logs/prompt23_articulated_grasp_R01_run02.txt`, and `logs/PROMPT23_STATUS.md`.
- Rendered the actual 500-step hold, 30 mm lift, and 500-step lifted hold as `videos/prompt23_articulated_grasp_lift.mp4`: 10.0 s, 640x480, 30 FPS, H.264; replay rise `30.000466 mm`, final passive face angle `+0.021780 degrees`; SHA256 `5e5d307c202269e41e95ed1c82cb13d2bd47e7df7d7a93e7451819e730637793`. The overlay explicitly states that no face turn was attempted.
- No face turn was attempted or claimed. Frozen main scene SHA256 remained `1e1e7666992f8e2c82ed11c05723489df0f9614f5a345021be16360eadf7f2a1`; articulated TEST scene SHA256 remained `baaf5fc65a93bc11ec12b9407e24c70568ed3d5f7bdb00916617b48f18fdacbb`.

## 2026-10-30 — Prompt 24 articulated face-layer ring reach

- Added `scripts/plan_articulated_face_reach.py` and scanned 20,000 fixed-seed (`24001`) static configurations of the four ring joints from the independently validated articulated R01 lifted state. Thumb/index/middle core support was mandatory; ring-core contact and penetration over 2 mm were rejected. Physics and both scene files remained frozen.
- The initial `0.4 mm` shallow-contact objective found 66 safe movable-face side contacts and wrote RF01. Independent 300-step transition plus 500-step hold remained safe but lost the face contact, so RF01 failed.
- Performed one bounded objective revision with the same seed, samples, variables, and gates, targeting approximately `1.0 mm` static contact depth. It wrote `poses/articulated_face_reach_RF02.json` with `rf_ds_collision_1` contacting the face-layer side.
- Independent RF02 validation sustained core support, avoided final ring-core contact, limited palm-relative drift to `0.393893 mm`, and had no excessive penetration or warning. Face-side contact began at step 280, was present at the final step, and occupied `87%` of the final 100 steps, below the strict 100% sustained-contact gate.
- Real contact moved the unactuated face joint to `+0.059071674 rad` (`+3.384761 degrees`). This is the first measured articulated-face motion, but FACE_REACH_GATE remains FAIL because contact was intermittent and the reach-only 2-degree drift bound was exceeded.
- Stop static reach-depth tuning. The next structural action is a controlled follow-through trajectory from RF02 geometry, targeting 5–10 degrees while maintaining core support; RF02 is not a promoted stable-contact pose.
- Evidence: `logs/prompt24_articulated_face_reach_planner.txt`, `logs/prompt24_articulated_face_reach_validation.txt`, `logs/prompt24_articulated_face_reach_planner_revision.txt`, `logs/prompt24_articulated_face_reach_RF02_validation.txt`, and `logs/PROMPT24_STATUS.md`.
- No 90-degree face turn was attempted or claimed. Main scene SHA256 remained `1e1e7666992f8e2c82ed11c05723489df0f9614f5a345021be16360eadf7f2a1`; articulated TEST scene SHA256 remained `baaf5fc65a93bc11ec12b9407e24c70568ed3d5f7bdb00916617b48f18fdacbb`.

## 2026-10-30 — Prompt 25 deterministic 5–10 degree follow-through PASS

- Adapted the user's MCP/PIP/DIP delta-grid idea to native MuJoCo. The supplied sketch used non-native `sim.get_state()` calls and placeholder force/angle values; `scripts/search_articulated_face_followthrough.py` instead uses MuJoCo state rollback, real `mj_contactForce` values, the real `face_turn` joint, and the existing support, drift, warning, and 2 mm penetration gates.
- Replayed the full validated R01 hold/lift and RF02 reach before every search. Froze `rf_rot` at its measured `-1.004311796 rad` state and evaluated all 124 combinations of MCP/PIP/DIP deltas in `{-0.10, -0.05, 0, +0.05, +0.10}` over a 150-step cosine transition.
- Two early candidates proved why the 500-step hold must be part of scoring: KF02 retained support/contact but settled to `10.419309 degrees`, above the frozen 10-degree ceiling; R2 settled to `10.317075 degrees`; R3 stayed at `7.788237 degrees` but its effective final-100 contact rate was only `34%`. All were correctly rejected without relaxing the gate.
- Revised the same bounded 124-case screen to include the 500-step hold. Eight candidates passed all search gates. The selected R4 delta was `rf_mcp +0.05`, `rf_pip -0.05`, `rf_dip -0.05 rad`, with `rf_rot` fixed. Its absolute targets are stored in `poses/articulated_face_followthrough_KF02_R4.json`.
- Independent clean-state validation ran the complete sequence twice. Both runs were deterministic and identical: final face angle `9.539155665 degrees`; final-100 angle range `0.080367665 degrees`; transition force-contact rate `99.333333%`; final-100 force-contact rate `100%`; transition/final-100 average normal force `0.355534855/0.015239348 N`; core support present for every step; no ring-core contact; maximum palm-relative drift `0.143383 mm`; minimum contact distance `-0.000076109 m`; no warning or penetration over 2 mm.
- `REPEATABLE_FOLLOWTHROUGH_GATE=PASS`. This is a scripted/zero-shot 5–10 degree articulated-face result, not a 90-degree face turn, learned policy, MPC, VLA, or full Rubik's Cube solve. `rf_ds_collision_1` is a MuJoCo box geom.
- Evidence: `logs/prompt25_face_followthrough_search_revision_3.txt` and `logs/prompt25_face_followthrough_validation_revision_3.txt`. Main and articulated TEST scenes remained frozen at SHA256 `1e1e7666992f8e2c82ed11c05723489df0f9614f5a345021be16360eadf7f2a1` and `baaf5fc65a93bc11ec12b9407e24c70568ed3d5f7bdb00916617b48f18fdacbb`.

## 2026-10-30 — Prompt 26 deterministic 15–20 degree KF03 PASS

- Replayed the complete clean-state R01 hold, 30 mm lift, RF02 reach, and validated KF02 R4 follow-through before taking the KF03 search snapshot. The measured baseline was `9.539155665 degrees`; no object state was set during the run.
- Measured ring state at the baseline: `rf_mcp 1.546438715`, `rf_rot -1.011383471`, `rf_pip -0.117608830`, and `rf_dip 1.217488338 rad`. Remaining space was `0.683561285 rad` to the MCP upper limit, `0.388391170 rad` to the PIP lower limit, and `1.583488338 rad` to the DIP lower limit. `rf_rot` remained the tight joint at only `0.035616529 rad` above its lower limit and was frozen.
- Confirmed `rf_ds_collision_1` is a MuJoCo box with half-size `[0.010, 0.003, 0.015] m`, or full dimensions `20 x 6 x 30 mm`.
- Added `scripts/search_articulated_face_kf03.py`. It deterministically evaluated all 342 MCP/PIP/DIP combinations from `{-0.10, -0.05, -0.02, 0, +0.02, +0.05, +0.10} rad`, with a 150-step cosine transition and a 500-step hold included in every candidate score. The fixed gate required a 15–20 degree endpoint, final-100 contact rate at least 95%, core support at every step, no ring-core contact, less than 0.5 mm palm-relative drift, no warning, and no penetration over 2 mm.
- Four candidates passed. Selected KF03 uses `rf_mcp +0.10`, `rf_pip -0.05`, and `rf_dip -0.10 rad`, with `rf_rot` fixed. The data-only target is `poses/articulated_face_followthrough_KF03.json`.
- Added `scripts/validate_articulated_face_kf03.py` and independently replayed the full sequence twice. Both runs were identical: final angle `15.766142345 degrees`; gain from KF02 `6.226986680 degrees`; transition/final-100 contact rates `99.333333%/100%`; transition/final-100 average normal force `0.328048683/0.062476757 N`; support rate `100%`; maximum palm-relative drift `0.112637 mm`; final-100 angle range `0.048485 degrees`; minimum distance `-0.000086950 m`; no ring-core contact, warning, or penetration over 2 mm.
- `REPEATABLE_KF03_GATE=PASS`. Continue ring-only deterministic follow-through until it produces no admissible candidate or reaches a joint/contact boundary; only then introduce a second pushing finger. No 90-degree face turn is claimed.
- Evidence: `logs/prompt26_face_kf03_search.txt` and `logs/prompt26_face_kf03_validation.txt`. Main and articulated TEST scenes remained frozen at SHA256 `1e1e7666992f8e2c82ed11c05723489df0f9614f5a345021be16360eadf7f2a1` and `baaf5fc65a93bc11ec12b9407e24c70568ed3d5f7bdb00916617b48f18fdacbb`.

## 2026-10-30 — Prompt 27 user-specified fine KF03_R1 grid FAIL

- Preserved the passing KF03 and added an independent `scripts/search_articulated_face_kf03_r1.py` for the requested asymmetric grid. The search evaluated all 125 combinations from MCP `[0, +0.02, +0.04, +0.06, +0.08] rad` and PIP/DIP `[-0.06, -0.04, -0.02, 0, +0.02] rad`, using a 150-step transition plus 500-step hold from the cleanly replayed `9.539155665 degree` KF02 endpoint.
- Fixed the `rf_rot` control target at the requested KF02 value `-1.004311796 rad`. The measured joint state at search start was MCP `1.546438715`, rot `-1.011383471`, PIP `-0.117608830`, and DIP `1.217488338 rad`; MCP retained `0.683561285 rad` to its upper limit.
- Applied the requested hard edge-contact filter: any `rf_ds_collision_1` normal force above `0.35 N` rejected the candidate. Also required final angle 15–20 degrees, final-100 contact rate at least 95%, thumb/index/middle support at every step, no ring-core contact, no warning or penetration over 2 mm, and no world-axis cube-centre drift above 0.3 mm.
- No candidate passed. The closest gate-relevant combination was MCP `+0.06`, PIP `-0.06`, DIP `-0.06 rad`: final angle `15.192065 degrees`, final-100 contact `100%`, support `100%`, but maximum transition axis drift was `[0.344021, 0.258723, 0.022549] mm` and peak normal force was `0.448145 N` over 115 spike steps. It failed both the 0.3 mm X-drift and 0.35 N force gates.
- The maximum-angle combination, MCP `+0.04`, PIP `-0.06`, DIP `-0.06 rad`, reached `15.242807 degrees` but had `1.021326 mm` transition X drift, `0.592415 N` peak force, and only `98.9%` support; it was also rejected.
- `KF03_R1_SEARCH=FAIL`; no pose file or independent replay pass was created. This bounded grid locates a strict single-ring edge-force/drift boundary under the requested thresholds, but it does not invalidate the previously measured KF03 PASS under its earlier acceptance gate.
- Evidence: `logs/prompt27_face_kf03_R1_search.txt`. Main and articulated TEST scene hashes remained unchanged.
- Rendered the complete real sequence (held grasp/lift, RF02, KF02 R4, and the closest KF03_R1 branch) as `videos/prompt27_KF03_R1_strict_failure.mp4`: 12.0 s, 640x480, 30 FPS, H.264. The render replay measured `15.182449 degrees`, peak normal force `0.448026 N`, and maximum XYZ axis drift `[0.349584, 0.257986, 0.023104] mm`; it therefore reproduces the same force/drift failure category and is explicitly labelled `STRICT GATE: FAIL`. Small contact-sensitive replay differences from the search log are shown rather than hidden. Video SHA256: `fb295971395c5c5032b4b1f729e587a2c7f0e9a835b620585b17f25a0a6f2fad`.

## 2026-10-30 — Prompt 28 four-finger articulated-face reversal PASS

- Interpreted "reverse the cube" as reversing the real articulated red face layer, not flipping the entire rigid cube. Replayed the clean R01 hold/lift, RF02 reach, KF02 R4, and KF03 sequence to the measured `15.766142345 degree` starting state.
- Added a deterministic four-finger reverse target. Relative to the KF03 endpoint, index PIP, middle PIP, and thumb IPL each move `+0.20 rad`; ring MCP/PIP/DIP move `-0.40/+0.40/+0.80 rad`. No object state, face-joint qpos, attachment, equality, learned policy, MPC, or VLA was used.
- Independent validation replayed the complete sequence twice with identical results: final face angle `-0.045199283 degrees`, reverse travel `15.811341628 degrees`, support rate `98.75%`, final-100 core support YES, maximum palm-relative drift `2.355730 mm`, minimum contact distance `-0.000201692 m`, no warning, and no penetration over 2 mm. `FOUR_FINGER_REVERSE_GATE=PASS`.
- Data-only target: `poses/articulated_face_four_finger_reverse_R01.json`. Validation evidence: `logs/prompt28_four_finger_reverse_validation.txt`.
- Rendered the actual trajectory as `videos/prompt28_four_finger_face_reverse.mp4`: 14.0 s, 640x480, 30 FPS, H.264. The render replay began its reverse phase at `15.773871701 degrees`, ended at `-0.046478936 degrees`, and reversed `15.820350637 degrees`. Video SHA256: `0857c348d0c7bf1355a10f9abe3e0de8698875dac776ad8791c9da5e50e365d6`.
- This demonstrates a four-finger scripted reversal of approximately 15.8 degrees back to zero; it is not a full 90-degree face turn and not a whole-cube flip. Main and articulated TEST scene hashes remained `1e1e7666992f8e2c82ed11c05723489df0f9614f5a345021be16360eadf7f2a1` and `baaf5fc65a93bc11ec12b9407e24c70568ed3d5f7bdb00916617b48f18fdacbb`.

## 2026-10-30 — Prompt 29 ring breakthrough diagnostic

- Replayed the validated KF02 endpoint (`9.539155665 degrees`) and applied the requested 150-step cosine keyframe: ring MCP `+0.08`, PIP `-0.06`, and DIP `-0.10 rad`, followed by a 500-step hold. Physics, scenes, and actuator gains were unchanged.
- The motion was not actuator-limited. The ring position actuators have no configured `forcelimit`; maximum MCP/PIP/DIP tracking errors were `7.258/6.837/9.493 mrad`, and final errors were `3.088/-0.975/-1.585 mrad`. Actual joint trajectories closely followed their targets.
- The finger did not slide off. Face contact was present for `649/650` steps, with the only missed sample at step 9; hold contact was `100%`. The face advanced to `13.383551 degrees` at the end of the transition and settled to `15.934528 degrees` at the end of hold.
- The trial instead crossed the frozen load/drift gates. Peak normal force was `0.636276 N` at step 119; force exceeded `0.5 N` for 7 steps and the strict `0.35 N` cap for 164 steps. Maximum cube-axis drift was `[0.836175, 0.516308, 0.024418] mm`, exceeding the 0.3 mm axis gate.
- Minimum ring contact distance was only `-0.123040 mm`, far below the 2 mm excessive-penetration threshold. This does not support a pseudo-penetration lock diagnosis; the evidence points to a high-load simultaneous path that moves the face but pushes the whole cube.
- The requested keyframe is therefore a diagnostic FAIL and was not promoted. Evidence: `logs/prompt29_ring_breakthrough_trace.csv`, `logs/prompt29_ring_breakthrough_diagnostic.txt`, and `logs/prompt29_ring_breakthrough_diagnostic.png`.
- Main and articulated TEST scene hashes remained `1e1e7666992f8e2c82ed11c05723489df0f9614f5a345021be16360eadf7f2a1` and `baaf5fc65a93bc11ec12b9407e24c70568ed3d5f7bdb00916617b48f18fdacbb`.

## 2026-10-30 — Prompt 30 DIP-first phased-delay scan FAIL

- Added an independent TEST-only phased trajectory search rather than modifying the validated follow-through searcher. The start state, target deltas (`rf_mcp +0.08`, `rf_pip -0.06`, `rf_dip -0.10 rad`), 150-step transition, 500-step hold, physics, and scene hashes were frozen. Only the MCP/PIP start delay varied over `30/45/60/75` steps (`60/90/120/150 ms` at the measured `0.002 s` timestep).
- No delay passed the combined 15–20 degree, 0.3 mm maximum world-axis drift, and 0.35 N transition-peak force gates. Results by delay were: 30 steps `15.994870 deg / 0.749345 mm / 0.618136 N`; 45 steps `16.009099 deg / 0.716668 mm / 0.613339 N`; 60 steps `15.855180 deg / 0.638490 mm / 0.603927 N`; 75 steps `15.830079 deg / 0.746886 mm / 0.611902 N`.
- The best drift and force result was 60 steps (`120 ms`), improving the simultaneous trajectory's `0.836175 mm / 0.636276 N` to `0.638490 mm / 0.603927 N`, but it remained well outside both safety limits. It also lost core support for 3 of 650 steps.
- The DIP-only prefix did not consistently unload contact. Its peak normal force before MCP/PIP activation rose with delay: `0.330702`, `0.419415`, `0.464659`, and `0.574302 N`. This rejects the assumed fixed `-0.10 rad` DIP/full-150-step profile as a pure unloading motion.
- `PHASED_DELAY_SEARCH=FAIL`; no pose was created or promoted. The next isolated diagnostic should scan DIP-only direction/amplitude and stop at the force minimum before reintroducing MCP/PIP timing, rather than adding more delay to the current profile.
- Evidence: `logs/prompt30_ring_phased_delay_search.txt` and `logs/prompt30_ring_phased_delay_traces.csv`. Main and articulated TEST scene hashes remained unchanged.

## 2026-10-30 — Prompt 31 pure-DIP direction/amplitude scan

- From the cleanly replayed KF02 endpoint, froze ring MCP/PIP/ROT at their measured values and scanned only DIP over 60 cosine-interpolated steps at `[-0.12, -0.09, -0.06, -0.03, 0, +0.03, +0.06, +0.09, +0.12] rad`. Every candidate started from the identical saved MuJoCo integration state.
- Negative DIP motion increased face angle but also sharply increased force and drift. Results for `-0.12/-0.09/-0.06/-0.03 rad` were respectively peak force `0.706/0.656/0.585/0.407 N` and maximum axis drift `1.260/0.848/0.485/0.091 mm`.
- Positive DIP motion was the true unloading direction. Results for `+0.03/+0.06/+0.09/+0.12 rad` were peak force `0.284/0.272/0.261/0.249 N` and maximum axis drift `0.038/0.036/0.042/0.054 mm`; final face angles remained `9.805/9.732/9.697/9.674 degrees`, all above the declared 9.3-degree floor.
- Under the user's angle/drift/peak-force objective, `+0.12 rad` was best: `0.248949 N` peak force and `0.053876 mm` maximum axis drift, with full core support and no warning or excessive penetration.
- Critical caveat: the positive-DIP force reduction is separation rather than contact-preserving tangential sliding. Contact rates fell to `56.7/40.0/33.3/28.3%` for `+0.03/+0.06/+0.09/+0.12 rad`, and all ended with zero ring/face normal force. The `+0.12 rad` result is therefore a diagnostic unload point, not a promoted Phase-1 pose.
- With a 95% contact-preservation requirement, no moving DIP candidate achieved peak force below 0.35 N. `-0.03 rad` retained `98.3%` contact but peaked at `0.406758 N`; the zero-delta control retained `98.3%` contact and peaked at `0.290801 N` but provides no unloading motion.
- Evidence: `logs/prompt31_pure_dip_unload_scan.txt` and `logs/prompt31_pure_dip_unload_traces.csv`. Main and articulated TEST scene hashes remained unchanged.

## 2026-10-30 — Prompt 32 partial-unload / early-reengagement 2D scan FAIL

- Kept the KF02 start state, MCP `+0.08 rad`, PIP `-0.06 rad`, 150-step transition, 500-step hold, physics, and scenes frozen. Scanned positive DIP `[+0.04, +0.06, +0.08, +0.10] rad` against MCP/PIP delays `[15, 25, 35, 45]` steps (`30/50/70/90 ms`) for 16 deterministic combinations.
- No combination crossed the four requested redlines. All 16 kept maximum axis drift between `0.0376` and `0.0400 mm` and transition peak normal force between `0.2880` and `0.2891 N`, but they failed both re-engagement and advancement.
- Transition contact rate ranged only from `26.7%` to `42.0%`; every combination had `0%` face contact during the 500-step hold. Contact ended between transition steps 41 and 64 and never returned, including when MCP/PIP began as early as step 15.
- Final held face angles ranged from `9.952438` to `10.228185 degrees`, far below the required `>15 degrees`. The maximum-angle branch was DIP `+0.04 rad`, delay 15 steps: final `10.228185 degrees`, contact `36.7%`, drift `0.039973 mm`, and peak force `0.289105 N`. The maximum-contact branch was DIP `+0.04 rad`, delay 45 steps: final `10.093110 degrees`, contact `42.0%`, drift `0.039286 mm`, and peak force `0.289105 N`.
- Interpretation: positive DIP cleanly removes the force/drift problem by separating the fingertip, but the fixed MCP+/PIP- endpoint does not create a return path from that separated geometry. This grid therefore demonstrates unload without dynamic re-engagement, not a smooth follow-through.
- `REENGAGEMENT_2D_SEARCH=FAIL`; no candidate was promoted and no repeat-pass run was triggered. The next isolated search must change the re-engagement geometry (MCP/PIP direction or an additional ring joint), rather than extending delay or positive DIP amplitude.
- Evidence: `logs/prompt32_ring_reengagement_search.txt` and `logs/prompt32_ring_reengagement_traces.csv`. Main and articulated TEST scene hashes remained unchanged.

## 2026-10-30 — Prompt 33 MCP/ROT reengagement-geometry scan FAIL

- Froze the KF02 start, DIP `+0.04 rad`, PIP `-0.06 rad`, 30-step delay, 150-step transition, and 500-step hold. Scanned MCP `[+0.10, +0.14, +0.18, +0.22] rad` against ROT `[0, +0.03, +0.06, +0.09] rad` for 16 clean-state combinations.
- No combination met the requested transition gates of at least 85% force contact, more than 13.5 degrees face angle, and less than 0.4 mm maximum axis drift. Transition contact ranged only `28.0%` to `36.7%`; transition-end angle ranged `9.941831` to `10.064462 degrees`; every 500-step hold had zero face contact.
- The highest contact and angle branch was the smallest geometry change, MCP `+0.10`, ROT `0`: transition contact `36.7%`, transition-end angle `10.064462 degrees`, held final angle `10.161375 degrees`, maximum full-run axis drift `0.039286 mm`, and peak force `0.289105 N`.
- Increasing MCP did not restore contact. At ROT `0`, contact declined from `36.7%` at MCP `+0.10` to `32.0%` at MCP `+0.22`; transition angle stayed near 10.05 degrees. This falsifies the assumed positive-MCP "deeper compression" direction in the positive-DIP geometry.
- Positive ROT consistently worsened reengagement. For MCP `+0.10`, contact fell `36.7% -> 32.7% -> 30.0% -> 28.7%` as ROT increased `0 -> +0.03 -> +0.06 -> +0.09`; transition angle fell `10.064462 -> 9.985772 -> 9.958332 -> 9.941831 degrees`. Larger MCP showed the same trend.
- Full-run axis drift remained only `0.0393` to `0.0450 mm`, but this is not evidence that ROT compensated contact load: the fingertip never reengaged and all hold contact rates were zero. At high MCP, positive ROT slightly increased X drift instead of reducing it.
- `REENGAGEMENT_GEOMETRY_SEARCH=FAIL`; no candidate was promoted and no repeat-pass validation ran. The next isolated action should use a local kinematic/Jacobian probe to measure which MCP/PIP/ROT directions actually move the fingertip toward the rotated face before another physics grid.
- Evidence: `logs/prompt33_reengagement_geometry_search.txt` and `logs/prompt33_reengagement_geometry_traces.csv`. Main and articulated TEST scene hashes remained unchanged.
- Rendered the best-contact failure branch (DIP `+0.04`, MCP `+0.10`, PIP `-0.06`, ROT `0`, delay 30) as `videos/prompt33_reengagement_geometry_failure.mp4`: 9.5 s, 640x480, 30 FPS, H.264. The real replay measured final angle `10.144335 degrees`, transition contact rate `36.666667%`, and maximum XYZ drift `[0.025728, 0.039002, 0.018327] mm`; it is explicitly labelled as a detached-finger reengagement failure. Video SHA256: `3abaff2f57623e72747226cee17b0f20322e3208cccf900ec42aa5dd74daf998`.

## 2026-10-30 — Prompt 34 proposed middle-finger handoff skeleton blocked at pre-contact

- Measured the middle distal geometry at the clean KF02 endpoint before implementing the proposed 600–750 handoff. `mf_ds_collision_1` was approximately `[95.3, 24.5, -56.2] mm` from the face-layer body centre in face-local coordinates and had no face contact. The proposed `mf_pre_pose` therefore required a large whole-finger reconfiguration, not a small press delta.
- Added a fixed-seed (`34001`) 20,000-sample kinematic reachability planner over the full legal middle-finger joint ranges. It found 3,867 side-near configurations and 383 that passed the planner's static crossing/core-clearance checks. The top static candidate placed the middle distal geom `1.491 mm` from a face side, but required `1.509 rad` joint-space motion.
- Dynamically replayed the top 20 candidates from the clean KF02 state using the requested 50-step cosine transfer plus 50-step hold. No candidate passed the pre-contact gate (`0.5–3 mm` final gap, <=0.2 degree face change, <0.1 mm drift, continuous thumb/index support, >=95% ring/face support, no excessive penetration/warning).
- The minimum-drift candidate still reached `2.271478 mm` maximum axis drift, only `69%` thumb/index core-support coverage, `16%` ring/face coverage, zero middle/face contact, and shifted the face by `+1.755955 degrees`. Other candidates drifted up to 25 mm; the closest dynamic face gap came with 13.390 mm drift. One candidate produced 2.504 N middle/face peak force and penetration beyond 2 mm.
- This rejects the handoff assumption that ring plus thumb/index can hold the KF02 object while the middle finger leaves its core-support role. At KF02 the middle finger is a load-bearing core support, not a freely transferable redundant finger. Stage 2 fails before a valid `mf_pre_pose`; stages 3–5 were correctly not executed, and no handoff pose was promoted.
- The next structural prerequisite is a new high-margin core grasp with thumb/index/ring support that remains valid while middle is removed, or an index-finger handoff path that preserves the existing middle support. Slowing the same middle transfer cannot plausibly close a 2.27 mm versus 0.1 mm drift gap without first changing support topology.
- Evidence: `logs/prompt34_middle_handoff_planner.txt` and `logs/prompt34_middle_prepose_dynamic_screen.txt`. Main and articulated TEST scene hashes remained unchanged.

## 2026-10-30 — Prompt 35 TEST-only 2x2 cube model

- Added `scenes/table_2x2_cube_test.xml` without modifying the frozen main scene or the earlier articulated TEST scene. The new model has eight visible corner cubies with six sticker colours. Its upper four cubies share the physical `face_turn` hinge; the lower four remain on the free cube core.
- Preserved the 50 mm overall size, friction `0.2`, total mass `0.125 kg`, free cube body, hinge damping `0.002`, friction loss `0.01`, and face range approximately `+/-95 degrees`. The two physical layers each weigh `0.0625 kg`. Decorative cubies and stickers are massless and non-colliding; continuous hidden collision layers prevent fingertips from catching in visual sticker gaps.
- `scripts/check_2x2_cube_scene.py` loaded the compiled model and passed all structural checks: eight visual cubies, equal layer masses, total mass, non-colliding visuals, 90-degree four-cubie layer rotation, no equality/attachment constraint, and valid hinge range. Evidence: `logs/prompt35_2x2_cube_scene_check.txt`.
- Rendered a geometry-only 0/90-degree comparison at `videos/prompt35_2x2_cube_preview.png`. This preview sets the face joint only to inspect geometry; it is not a hand-control result.
- New TEST scene SHA256: `b126a0fa132d4ae1a1752647258ebb841a73ddd15fde1e84e4c335facb6a5831`. Frozen main and prior articulated TEST scene hashes remain `1e1e7666992f8e2c82ed11c05723489df0f9614f5a345021be16360eadf7f2a1` and `baaf5fc65a93bc11ec12b9407e24c70568ed3d5f7bdb00916617b48f18fdacbb`.
- The old grasp and 15.766-degree trajectory have not yet been revalidated against the equal-mass 2x2 layers. No grasp or face-turn success is claimed for the new model.
- Probed the existing four-finger trajectory on the new scene before making a video. It failed to transfer: the face state was `1.642352 degrees` before the reverse phase and `2.436732 degrees` after it, support coverage was `78.75%`, and maximum palm-relative drift was `3.581295 mm`. Evidence: `logs/prompt35_2x2_old_trajectory_probe.txt`.
- Rendered `videos/prompt35_2x2_face_turn_preview.mp4`, a 7.0-second, 640x480 geometry preview that moves the physical `face_turn` coordinate from 0 to 90 degrees. The video is explicitly labelled `PREVIEW ONLY - not a hand-controlled result`; it is visual evidence of the 2x2 layer geometry, not a manipulation result. Video SHA256: `6aa255af7c94a5cb678ed30bba2758317c01e2744e80d36a83815a5f84425a1c`.

## 2026-10-30 — Prompt 36 LEAP-hand-controlled 2x2 visualization PASS

- Added `scenes/table_2x2_cube_control_test.xml`. It preserves the previously validated articulated scene's continuous collision geometry, `0.125 kg` total mass, friction, passive hinge, and actuator setup while adding eight massless, non-colliding visual cubies and six-colour 2x2 stickers. The visible top four cubies are children of `face_layer` and therefore move only when contact dynamics rotate the passive hinge.
- Replayed the complete held-grasp/lift/RF02/KF02/KF03/four-finger reverse sequence twice with `scripts/validate_2x2_hand_turn.py`. Both runs were identical: forward endpoint `15.766142345 degrees`, reverse travel `15.811341628 degrees`, final angle `-0.045199283 degrees`, support coverage `98.75%`, final-100 support sustained, maximum palm-relative drift `2.355730 mm`, minimum contact distance `-0.201692 mm`, no warning, and no penetration over 2 mm. `HAND_CONTROLLED_2X2_TRAJECTORY=PASS` under the existing Prompt 28 reversal gate.
- The run never writes the face-joint qpos and contains no attachment/equality constraint. This is a scripted LEAP contact-control result, not the earlier kinematic 90-degree preview and not a learned policy.
- Rendered the actual trajectory as `videos/prompt36_leap_hand_2x2_turn.mp4`: 14.0 seconds, 640x480, 30 FPS. Render replay: forward/reverse phase start `15.773871701 degrees`, final `-0.046478936 degrees`, reverse travel `15.820350637 degrees`. Video SHA256: `3fd424bbed85d74dc4c403d99717e8fed3c394cbb4736be88f951df07f2a288f`.
- Evidence: `logs/prompt36_2x2_hand_turn_validation.txt`. Control TEST scene SHA256: `f0306a5dd5514030bc93d8e221374838033041cdfb577fe21a4b6ae066ffbe30`. The 90-degree hand-control goal remains unpassed.

## 2026-10-30 — Prompts 37–38 redundant-support search FAIL

- Tested three evidence-driven routes for establishing the redundant support required before finger gaiting. No physics, scene, friction, mass, solver, mount, or existing validated pose was changed.
- KF03 ring-to-core route: 50,000 fixed-seed full-range ring samples produced 154 core-near poses and 11 statically safe candidates. All 11 failed real 150-step transitions plus 300-step holds. No candidate achieved dynamic ring-tip/core contact. The best ranked branch still moved the cube `10.344420 mm` on one axis and let the face fall from `15.766142` to `12.392382 degrees`. Evidence: `logs/prompt37_ring_core_support_planner.txt` and `logs/prompt37_ring_core_support_dynamic_screen.txt`.
- KF03 index-to-face route: 20,000 fixed-seed samples produced 2,230 face-side poses and 939 statically safe candidates. The top 30 all failed 100-step transitions plus 100-step holds. The minimum-drift branch had `1.274922 mm` maximum axis drift, zero index/face contact, only `42.5%` ring/face coverage, and `79.5%` thumb/middle support. Evidence: `logs/prompt37_index_face_handoff_planner.txt` and `logs/prompt37_index_face_handoff_dynamic_screen.txt`.
- Initial lifted-grasp ring-to-core route: 30,000 fixed-seed samples produced 56 core-near poses and 32 statically safe candidates. The top 30 direct 200-step transitions all lost the target as the cube moved; no dynamic ring-tip/core contact occurred. The apparent static solutions require moving `rf_rot` from approximately `+0.368` to approximately `-1.0 rad`, so simultaneous interpolation sweeps through and displaces the object.
- Retested the top 10 initial-grasp targets under six two-stage joint-order schedules (60 paths total, 150 steps per phase plus a 300-step hold). No schedule restored ring/core contact. The best ranked no-penetration route (`curl_then_rot`) still had `2.231530 mm` drift, `77.17%` original support coverage, and `-1.736964 degrees` face change. Evidence: `logs/prompt38_initial_ring_core_planner.txt`, `logs/prompt38_initial_ring_core_dynamic_screen.txt`, and `logs/prompt38_initial_ring_core_phased_search.txt`.
- No support pose was promoted. These results reject timing-only transfer from the existing grasp. The next prerequisite is a new whole-hand grasp whose initial geometry already places the ring near the core, followed by the full hold/lift gate before any face-turn search.
- Frozen main and articulated TEST scene hashes remained unchanged.

## 2026-10-30 — Prompt 39 global redundant-grasp topology search FAIL

- Frozen the existing 15.766-degree forward/reverse assets in `logs/prompt39_presearch_asset_manifest.txt`, then added independent TEST-only scripts `scripts/search_global_redundant_grasp.py` and `scripts/validate_global_redundant_grasp.py`. The scripts do not edit either scene and never write the cube free-joint or face hinge during a run.
- The static search used fixed seed `39001`: 20,000 uniform samples plus 40,000 geometry-guided samples. It scanned cube X/Y offsets over `+/-45 mm`, absolute mount pitch/roll over `+/-0.2 rad`, relative yaw over `+/-0.2 rad`, and all four ring joints over their full legal ranges. Because the scene has no mount-yaw joint, relative yaw was represented by the opposite cube yaw at initialization and recorded explicitly.
- Of 60,000 samples, 37 put `rf_ds_collision_1` within the declared `[-2.0,+1.5] mm` window of both `cube_core_geom` and `face_layer_geom`. Only one also kept thumb/index/middle within 2 mm of the core, but it had `13.740 mm` penetration. No candidate passed the combined dual-layer, original-support, and <=2 mm penetration static gate.
- Re-ranked failed candidates to avoid validating deep-penetration near-misses, then dynamically tested the top 20 penetration-safe candidates for a 500-step hold, 300-step 30 mm lift, and 500-step lifted hold. No candidate sustained healthy ring contact on the face layer; the best face-layer last-100 healthy-contact rate was `0%`. One candidate sustained ring/core force for all of the last 100 steps, but lost all thumb/index/middle support and drifted `18.576 mm` during hold.
- Across the 20 penetration-safe dynamics, minimum hold axis drift was `3.399884 mm` and minimum palm-relative lift drift was `2.740400 mm`, compared with the requested `0.1 mm` limits. The best branch retaining thumb/index/middle in the final 100 steps still had `3.978533 mm` hold drift, zero ring/core and zero ring/face healthy contact.
- `GLOBAL_REDUNDANT_GRASP=FAIL`. No pose was written, the release test was not run, and the previously validated 15.766-degree turn was not transplanted onto an unvalidated topology. Evidence: `logs/prompt39_global_redundant_grasp_planner.txt` and `logs/prompt39_global_redundant_grasp_validation.txt`.
- This search isolates the topology conflict: within the requested cube/palm/ring-only variables, ring dual-layer proximity and the existing three-finger support occupy incompatible or penetrating regions. The next search must include coordinated repositioning of the other support fingers (or redesign the grasp/object placement), not merely enlarge the same ring/cube scan.
- Frozen main and articulated TEST scene hashes remained `1e1e7666992f8e2c82ed11c05723489df0f9614f5a345021be16360eadf7f2a1` and `baaf5fc65a93bc11ec12b9407e24c70568ed3d5f7bdb00916617b48f18fdacbb`.

## 2026-10-30 — Prompt 40 natural-joint cross-seam grasp search FAIL

- Added `scripts/search_natural_redundant_grasp.py` to test the proposed redundant initial topology without reusing the abnormal ring posture. Ring rotation was restricted to `[-0.5,+0.5] rad`, MCP to `[0,1.8] rad`, PIP to `[-0.1,1.5] rad`, and DIP to within `0.7 rad` of PIP. The search retained the frozen mass, friction, solver, scenes, and actuator parameters.
- The first fixed-seed run evaluated 80,000 ring/object/palm samples and then 800 support-joint samples per thumb/index/middle finger for each retained ring configuration. With fixed relative Z it produced 12 near-seam configurations but no true dual-layer + three-support + penetration-safe candidate. The least-bad configuration still had `12.923 mm` penetration.
- Added only the missing relative-height variable (`cube Z +/-30 mm`) and explicit per-segment penetration rejection. The rerun produced 83 raw near-seam configurations; all 83 penetrated the cube by more than 2 mm through another ring collision segment.
- Relaxed the topology definition from one ring box touching both layers to separate ring collision segments touching core and face. Under the same 80,000-sample natural-joint search, no penetration-safe near-seam configuration existed. `NATURAL_CROSS_SEAM_GRASP=FAIL`.
- Per the grasp-first gate, no dynamic hold, lift, finger release, or face-turn test was run, and no pose was promoted. This is a geometry/topology rejection, not a controller failure.
- The evidence rejects using one LEAP ring finger as a simultaneous cross-seam core support and face pusher with the current collision geometry. The next grasp topology should start with all four fingers supporting the core; a pusher may then leave only after the remaining three supports pass the release gate. This replaces the cross-seam assumption rather than loosening penetration limits.
- Evidence: `logs/prompt40_natural_redundant_grasp_planner.txt` and `logs/prompt40_natural_redundant_grasp_iterations.txt`. Frozen main and articulated TEST scene hashes remained unchanged.

## 2026-10-30 — Prompt 41 natural orange true-tip grasp attempt FAIL

- Confirmed the existing 15.766-degree trajectory is driven by `rf_ds_collision_1`, not by the white/orange finger-end collision mesh `rf_tip`. Under the validated old grasp, 100,000 natural ring samples with `|rf_rot| <= 0.5 rad` left the true tip at least `28.063985 mm` from the movable face side.
- Added a global TEST-only topology search that accounts for the true `rf_tip` mesh bounds, scans initial cube/palm placement, keeps natural ring constraints, and re-solves thumb/index/middle core support. From 100,000 samples, 1,266 placed the true tip near the face, 939 were rejected for ring penetration, 327 were ring-safe, and two passed the complete static tip + three-support + penetration gate.
- Neither static candidate survived dynamics. Candidate 97808 lost both true-tip contact and all three-finger support, with `6.994255 mm` hold drift. Candidate 97353 ended with `0%` true-tip contact, `77%` support, and `17.050470 mm` hold drift. Neither entered a turn test.
- Ran two 640-candidate local dynamic refinements around the better topology. The first exposed mutually exclusive branches: a `100%` true-tip / `0%` support branch and a `0%` true-tip / `100%` support branch. Re-running with an objective that explicitly maximized the weaker of the two still produced no overlap; its best ranked result had `0%` true-tip contact, `100%` support, and `14.227612 mm` drift.
- `NATURAL_TRUE_RING_TIP_GRASP=FAIL`. No pose was promoted, no face turn was attempted, and no new video was created. The next attempt must redesign the whole dynamic grasp/contact geometry; continuing to tune a ring pushing keyframe is not justified until a true-tip held grasp passes.
- Evidence: `logs/prompt41_natural_ring_tip_face_search.txt`, `logs/prompt41_global_natural_ring_tip_grasp.txt`, `logs/prompt41_global_natural_ring_tip_grasp_validation.txt`, `logs/prompt41_natural_ring_tip_hold_refinement.txt`, and `logs/prompt41_natural_ring_tip_iteration_summary.txt`. Frozen scene hashes remained unchanged.
