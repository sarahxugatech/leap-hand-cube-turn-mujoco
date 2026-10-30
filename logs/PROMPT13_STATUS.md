# Prompt 13 status — first Phase 2 contact sequence

Result: **FAIL — safe gait saturated at 1.592 degrees, target remains 90 +/- 10 degrees**

| Run | Z rotation | Tilt | Max palm-relative drift | Self penetration | End held | Verdict |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| Scripted thumb sweep | -0.019 deg | 0.001 deg | 1.133 mm | safe | yes | no effective torque |
| Scripted differential | +5.348 deg | 1.093 deg | 20.415 mm | -3.216 mm | yes | rejected |
| Planned K01, one cycle | +1.540 deg | 0.044 deg | 3.457 mm | -1.996 mm | yes | safe, target fail |
| Planned K01, two cycles | +1.592 deg | 0.044 deg | 3.457 mm | -1.996 mm | yes | safe, saturated |
| Planned K01, repeated | stopped at step 3060 | 1.409 deg at stop | 3.457 mm | -2.003 mm | no | hard safety stop |

Planner parameters: seed 13001, population 24, 6 generations, elite count 5, time budget 45 s, elapsed 26.048 s.

Evidence:

- `logs/PROMPT13_CONTACT_SEQUENCE.md`
- `logs/prompt13_scripted_turn_run01.txt`
- `logs/prompt13_scripted_turn_run02.txt`
- `logs/prompt13_turn_keyframe_planner.txt`
- `logs/prompt13_scripted_turn_run03_K01.txt`
- `logs/prompt13_scripted_turn_run05_K01_early_stop.txt`
- `logs/prompt13_scripted_turn_run06_K01_2cycles.txt`
- `videos/prompt13_K01_two_cycle_failure.mp4`

Next method change required: redesign the contact mode so the pusher can reposition on a new cube face while thumb + two supports remain active. Do not add more cycles or continue tuning this saturated keyframe sequence.
