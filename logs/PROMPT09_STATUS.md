# Prompt 09 Status

- Scope: original modified-left LEAP geometry in a disposable TEST scene only.
- Source commit: `0ab03eb2244b4a1a020d94ef38a4823bb9e15ba0`.
- Source model: `left_hand_custom.xml`, SHA256 `39972137f563b0e01ee8034a7820d66870c4e6d4b5fb24c9195212a50cd4e0a0`.
- Cache: `leap_left_custom_grasp_cache_cube5cm.npz`, SHA256 `a44b7e2af512215acc1750a4f1133f6a10315854c5e17a4c2e2f1cdc10e54eaf`.
- Source row: 3388; cube half-size `0.024919355 m`.
- The complete source cube pose was used, including quaternion `(w, x, y, z) = (0.237825811, 0.086552799, 0.021958888, 0.967194617)`.
- Initial contacts: palm `palm_collision_6` and middle-finger base `mf_bs_collision_1` only.
- Initial thumb contact: NO. Initial non-thumb fingers: 1 (`mf`).
- Deepest hand-cube distance: `-0.000205187 m`, within the 0.002 m penetration limit.
- Initial contact gate: FAIL.
- Dynamic steps: 0, as required by the gate. No attachment, equality constraint, lift, or rotation was used.
- Gate A: FAIL. Relaxed: FAIL. Passive hold: FAIL / NOT RUN beyond the initial gate.
- Main scene SHA256 before/after: `1e1e7666992f8e2c82ed11c05723489df0f9614f5a345021be16360eadf7f2a1`.

## Verdict

**STOP — row 3388 is not an initial thumb-plus-two-finger grasp even in the original modified-left geometry.**

The exact source geometry and full cached cube orientation reproduce the previously observed shallow contact, but they do not satisfy this project's grasp-entry requirement. The cache is used by an RL environment whose reset code loads cube size and pose while keeping robot joint initialization separate; treating one cache row as a self-contained verified grasp was therefore not justified.

## Video

- `videos/prompt09_initial_gate_failure.mp4`: 6.0 s, 640x480, 30 fps, H.264.
- SHA256: `1ba5df8cf684552d28239aee6bb0d3d2034dd5d044dee7f8c3dfa1bb7c3a2345`.
- Static visualization only; the camera moves, but physics does not step.
