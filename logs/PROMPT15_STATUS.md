# Prompt 15 status — two-stage scripted turn

Result: **FAIL — no sustained tangential push under the current mount orientation**

| Run | Start contact mode | Z rotation | Drift | Safety | Result |
| --- | --- | ---: | ---: | --- | --- |
| 01 | T01 ring-side | +1.236 deg at stop | 0.605 mm | support lost | FAIL |
| 02 | D01 index+ring dual-side | -0.014 deg | 0.195 mm | safe, end-held | FAIL |

Run 01 stopped immediately when only thumb + middle remained. Run 02 completed, but the ring left the cube during both sweeps, so the trajectory transferred essentially zero Z torque.

Intermediate verified results:

- T04 index-tip pose: PASS, 500-step hold, max/end drift 2.852/1.733 mm.
- Dual-side planner: seed 15003, population 64, 8 generations, elite count 8, 45 s budget, 26.495 s elapsed.
- D01/D02/D03: all PASS 500-step holds with `if_tip` + `rf_ds_collision_1` side contacts.

Evidence:

- `logs/PROMPT15_CONTACT_SEQUENCE.md`
- `logs/prompt15_two_stage_run01.txt`
- `logs/prompt15_side_contact_reachability.txt`
- `logs/validate_turn_ready_T04_index_tip.txt`
- `logs/prompt15_dual_side_planner.txt`
- `logs/validate_dual_side_D01.txt`
- `logs/validate_dual_side_D02.txt`
- `logs/validate_dual_side_D03.txt`
- `logs/prompt15_two_stage_run02_dual_side.txt`
- `videos/prompt15_two_stage_run02_dual_side.mp4`

Next required change: TEST-scene hand mount orientation redesign, followed by complete re-validation of grasp, lift, and turn-ready contact gates. Do not continue tuning the current ring-sweep sequence.
