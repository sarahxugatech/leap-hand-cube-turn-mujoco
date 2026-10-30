# Prompt 14 status — turn-ready contact mode

Result: **PASS**

| Candidate | Max/end palm drift | Thumb sustained | Supports sustained | Distal side contact | Penetration | Gate |
| --- | ---: | --- | --- | --- | --- | --- |
| T01 | 0.113/0.112 mm | yes | index + middle + ring | `rf_ds_collision_3` | no | PASS |
| T02 | 0.218/0.214 mm | yes | index + middle + ring | `rf_ds_collision_3` | no | PASS |
| T03 | 0.222/0.203 mm | yes | index + middle + ring | `rf_ds_collision_3` | no | PASS |

Planner parameters: seed 14001, population 64, 8 generations, elite count 8, 45 s budget, 22.970 s elapsed, 300-step transition, 100-step screen hold.

Selected candidate: `poses/turn_ready_T01.json`.

Evidence:

- `logs/prompt14_turn_ready_planner.txt`
- `logs/validate_turn_ready_T01.txt`
- `logs/validate_turn_ready_T02.txt`
- `logs/validate_turn_ready_T03.txt`
- `videos/prompt14_turn_ready_T01_transition.mp4`

No rotation was run. Next phase: Prompt 15, a two-stage scripted/zero-shot turn beginning from T01, with the existing frozen 90 +/- 10 degree gate and two consecutive full passes required.
