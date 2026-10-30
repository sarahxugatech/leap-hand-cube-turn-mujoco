# Prompt 06 Status

- Source: official `leap-hand/LEAP_Hand_Sim` grasp cache
- Source commit: `150bc3d4b61fd6619193ba5a8ef209f3609ced89`
- Source cache: `leap_hand_in_palm_cube_grasp_50k_s09.npy`, row 0
- Source cache SHA256: `395ef1e0618e342d492d4f4c4cee881b5aea0a00d9cb9929363e54c857935b0e`
- Source cube side: 0.0675 m
- Current cube side: 0.0500 m
- Imported pose: `poses/leap_cache_s09_row0000.json`
- Thumb ever / sustained: NO / NO
- Final finger contacts: 0
- Maximum / end XY displacement: 0.000 / 0.000 mm
- Penetration over 0.002 m: NO
- Gate A: FAIL
- Relaxed entry: FAIL
- Lift: NOT RUN
- Rotation: NOT RUN
- Main scene SHA256 before/after: `1e1e7666992f8e2c82ed11c05723489df0f9614f5a345021be16360eadf7f2a1`

## Verdict

**NO GRASP CANDIDATE — the imported official pose is stable for a 6.75 cm source cube but leaves the current 5 cm cube without hand contact.**

The next source must match both the right-hand Menagerie geometry and the 5 cm cube. A public 5 cm cache was found only for a modified left LEAP hand, so mirroring it would be a new human-approved approximation rather than a verified right-hand import.
