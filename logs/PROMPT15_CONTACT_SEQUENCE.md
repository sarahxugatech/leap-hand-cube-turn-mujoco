# Prompt 15 two-stage scripted / zero-shot turn sequence

Frozen before run 01:

- Source held pose: `poses/grasp_hold_refined_R01.json`
- Turn-ready transition: `poses/turn_ready_T01.json`
- Target: world-Z cube rotation 90 degrees +/- 10 degrees
- Start and end holds: 500 steps each
- Safety: thumb plus at least two non-thumb fingers during the turn, no drop/wedge, no penetration over 0.002 m
- Repeatability: two identical complete passes required

Fixed two-stage sequence:

1. Rebuild R01, hold, lift 30 mm, transition to T01, and hold 500 steps.
2. **Stage 1 sweep:** keep thumb/index/middle support; increase `rf_mcp` from the T01 value to 1.65 rad. The ring distal side contact moves along negative X to create positive Z torque.
3. **Release:** move `rf_rot` outward to -0.70 rad while thumb/index/middle remain in contact.
4. **Reset/recontact:** return `rf_mcp` to the T01 value, then return `rf_rot` to the T01 value.
5. **Stage 2 sweep:** repeat the same `rf_mcp` sweep from the new cube orientation.
6. Hold the final command for 500 steps and measure the cube free-joint quaternion.

This is one deterministic scripted trajectory. It is not MPC, VLA, or a learned policy. The target and safety gates are not changed after failure.
