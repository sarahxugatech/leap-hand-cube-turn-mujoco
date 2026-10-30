# Technical Report

## Objective

Use one LEAP Hand in desktop Python MuJoCo simulation to turn one face of a cube by 90°, starting and ending in a stable held grasp.

Full cube solving, two-hand manipulation, hardware, learned policies, VLA, and MPC are outside the project scope.

## Implementation

- Python 3.12.14, MuJoCo 3.17.0, NumPy 2.5.4
- 16-DoF LEAP Hand model from MuJoCo Menagerie
- Position actuators with cosine-interpolated targets
- Staged grasp, lift, reach, face-turn, hold, and reverse controller
- Simplified articulated 2×2 cube with an unactuated rotating layer
- Deterministic pose search and two-run replay validation

## Controller and simulation modifications

1. Abrupt joint targets were replaced by smooth cosine interpolation.
2. The controller was separated into explicit phases so each transition could be measured independently.
3. Joint-space grid searches were added to find contact-rich reach and follow-through keyframes.
4. The original rigid cube was replaced by a simplified 2×2 cube with a passive face joint.
5. The validator checks contact support, cube drift relative to the palm, penetration, solver warnings, and deterministic repeatability.

No equality constraint attaches the cube to the hand, and the face-joint position is not written during a trajectory.

## Validated result

| Field | Measurement |
|---|---:|
| LEAP model counts | `nq=16`, `nv=16`, `nu=16`, `nbody=18`, `ngeom=89` |
| Passive table-scene XY drift | 0.000011 mm |
| Passive table-scene Z change | −0.108 mm |
| Lift | approximately 30 mm |
| Forward face angle | 15.766142345° |
| Reverse travel | 15.811341628° |
| Final face angle | −0.045199283° |
| Core-support rate | 98.75% |
| Maximum palm-relative drift | 2.356 mm |
| Minimum contact distance | −0.202 mm |
| Penetration over 2 mm | No |
| Solver/contact warnings | None |
| Repeatability | Two identical passing runs |

Primary evidence: [`logs/validated_2x2_trajectory.txt`](../logs/validated_2x2_trajectory.txt).

## Main challenge

The pushing ring finger approaches a geometric limit near 15°. Attempts to reposition another finger for continued rotation reduce support of the cube core and cause whole-object translation. The next engineering step is a redundant initial grasp or a contact-aware finger-handoff strategy.

## Status

The project demonstrates a repeatable contact-driven partial face turn and reversal. The complete 90° turn and stable final grasp remain open work.

## Method reference

The project was informed by the dexterous-astra manipulation workflow described by Jianglong Ye, particularly its simulation structure, grasp-first ordering, and sampled-keyframe pattern. The LEAP model integration, poses, controller, measurements, and validation were independently implemented in MuJoCo; no controller code or checkpoints were copied.

