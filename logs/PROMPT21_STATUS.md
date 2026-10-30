# Prompt 21 status — high-margin R01 and middle-finger target-axis probe

- Main scene SHA256 before/after: `1e1e7666992f8e2c82ed11c05723489df0f9614f5a345021be16360eadf7f2a1`.
- R01 independent 500-step hold: Relaxed PASS; maximum/end displacement `0.435/0.261 mm`; sustained thumb plus index/middle; no penetration over 2 mm.
- R01 independent lift: PASS; cube rise `30.063160 mm`; maximum/end palm-relative drift `1.218986/0.970637 mm`; sustained thumb plus index/middle; no penetration or warning.
- R01 all-joint probes at `0.10`, `0.20`, and `0.30 rad`: no action met the promotion gate of `abs(world-Z yaw) >= 3 degrees` and tilt `<= 2 degrees`.
- Clean R01 direction: `mf_rot +0.60 rad` produced yaw `-1.915586 degrees`, tilt `0.084147 degrees`, and `2.200 mm` maximum displacement with thumb plus index/middle support.
- Replayed the validated R01 lift and R01-to-T01 300-step transition, then held 500 steps before each T01 probe. T01 `mf_rot +0.60 rad` produced the largest clean yaw: `-2.365691 degrees`, tilt `0.044256 degrees`, and `1.078 mm` maximum displacement. Thumb plus index/middle remained, but ring contact was absent at the endpoint.
- T01 `mf_rot +0.80 rad` retained thumb plus index/middle/ring contact but yaw fell to `-2.118956 degrees`; tilt was `0.086182 degrees`, displacement `1.112 mm`. At `+1.00 rad`, the probe failed the sustained-support stop.
- Verdict: measurable, clean world-Z torque was found, but the predeclared 3-degree promotion gate did not pass. No reset gait and no 90-degree actor was run.
- Next structural step: optimize the T01 ring-support/contact placement while keeping the validated R01 startup and `mf_rot` as the pusher; require independent hold/lift and at least 3 degrees clean yaw before building a push-release-reset gait.
- No attachment, equality constraint, cube-qpos update during a run, learned policy, MPC, or VLA was used.

