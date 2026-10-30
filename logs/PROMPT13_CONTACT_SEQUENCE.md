# Prompt 13 scripted / zero-shot contact sequence

Frozen before the first rotation run:

- Start pose: `poses/grasp_hold_refined_R01.json`
- Rotation target: cube orientation about world Z = 90 degrees +/- 10 degrees
- Start hold: 500 steps
- End hold: 500 steps
- Required throughout: no penetration over 0.002 m, no drop/wedge, and no cube-qpos update after initialization
- Required at end: sustained thumb plus at least two non-thumb fingers
- Final success requires the identical script to pass twice consecutively

Planned fixed sequence:

1. **Start held:** palm + index base + middle base support the lower face; thumb distal segment guards the negative-Y side.
2. **Lift/clear:** raise mount Z by 30 mm while preserving the validated grasp.
3. **Guard:** palm/index/middle remain support contacts. The thumb remains the side wall.
4. **Push/sweep:** increase `th_axl` while reducing `th_mcp`. The paired motion is intended to move the thumb tangentially in X while approximately preserving its normal Y pressure, producing Z torque.
5. **Gait/reset:** reduce `th_mcp` to release thumb pressure, return `th_axl`, then re-close `th_mcp`. Palm/index/middle must support the cube while the thumb resets.
6. **Finish:** restore the held command and hold for 500 steps. Measure rotation from the cube free-joint quaternion, never from finger angles.

This is a single deterministic scripted diagnostic, not sampled planning, MPC, VLA, or a learned policy. Failure is retained and reported; the target tolerance and safety gates are not changed after the run.
