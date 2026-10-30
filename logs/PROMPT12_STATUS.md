# Prompt 12 status — Phase 1 grasp and lift

Result: **PASS (Relaxed held-pose gate + lift gate)**

- Held pose: `poses/grasp_hold_refined_R01.json`
- 500-step hold: max/end displacement 0.435/0.261 mm
- End contacts: thumb sustained; index + middle; palm support
- Penetration over 0.002 m: NO
- Lift: commanded 30.000 mm; measured cube rise 30.063 mm
- Lift max/end palm-relative drift: 1.219/0.971 mm
- Attach/equality constraint: NO
- Rotation attempted: NO
- Main scene SHA256: `1e1e7666992f8e2c82ed11c05723489df0f9614f5a345021be16360eadf7f2a1`

Evidence:

- `logs/prompt12_grasp_optimizer.txt`
- `logs/prompt12_hold_refinement.txt`
- `logs/hold_grasp_hold_refined_R01.txt`
- `logs/lift_grasp_hold_refined_R01.txt`

Next gate: write the contact sequence first, then implement the scripted/zero-shot 90-degree reorientation. Freeze tolerance at 90 degrees +/- 10 degrees and require two consecutive complete passes.
