# Prompt 23 status — articulated object R01 hold and lift

- Pose: `poses/articulated_grasp_R01.json`.
- Scene: `scenes/table_articulated_face_test.xml`, SHA256 `baaf5fc65a93bc11ec12b9407e24c70568ed3d5f7bdb00916617b48f18fdacbb`.
- The exact hold-then-lift sequence was run independently twice; both runs produced identical results.
- 500-step hold: maximum/end core displacement `0.437866/0.272997 mm`; sustained thumb plus index/middle support; maximum passive face-layer drift `0.000481910 rad` (`0.027611 degrees`); HOLD_GATE PASS.
- 30 mm mount lift plus 500-step hold: core rise `30.000466 mm`; maximum/end palm-relative drift `0.518315/0.096846 mm`; sustained thumb plus index/middle support; maximum face-layer drift `0.000059913 rad` (`0.003433 degrees`); LIFT_GATE PASS.
- Final support contacts were core-only: palm, index base, middle base, and thumb distal. No hand geom contacted the face layer at the endpoint, leaving the layer free for a future pusher.
- Minimum hand-object distance was `-0.001605751 m`; no finger-finger or hand-table penetration over 2 mm and no MuJoCo warning occurred.
- No face turn was attempted. No attachment, equality constraint, runtime object-qpos update, learned policy, MPC, or VLA was used.
- Evidence: `logs/prompt23_articulated_grasp_R01_run01.txt` and `logs/prompt23_articulated_grasp_R01_run02.txt`.

