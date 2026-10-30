# Prompt 24 status — ring-finger movable-face reach

- Fixed static search: seed `24001`, 20,000 samples over the four ring joints, with thumb/index/middle required to remain on the core, ring-core contact rejected, and the 2 mm penetration limit retained.
- The initial shallow-contact objective found 66 safe side-contact samples. RF01 was safe dynamically but lost the face-layer contact and failed the sustained-contact gate.
- One predeclared objective revision retained the same seed, samples, variables, physics, and gates while changing the target static contact depth from about `0.4 mm` to `1.0 mm`. It produced `poses/articulated_face_reach_RF02.json` with `rf_ds_collision_1` as the side-contact geom.
- Independent RF02 replay: core support was sustained; ring-core contact was absent in the last 100 steps; maximum palm-relative core drift was `0.393893 mm`; minimum hand-object distance was `-0.210145 mm`; no excessive penetration or warning occurred.
- RF02 first reached the movable face at step 280 and had side contact through the final step, but contact occupied only `87%` of the last 100 steps. The strict sustained-contact condition therefore failed.
- The unactuated face layer nevertheless moved to `+0.059071674 rad` (`+3.384761 degrees`) under real ring-finger contact. This is physical feasibility evidence, not a reach-gate pass and not a 90-degree face turn.
- Verdict: FACE_REACH_GATE FAIL because contact was intermittent and face drift exceeded the 2-degree reach-only limit. Do not promote RF02 as a stable contact pose. Use it only as the start geometry for a follow-through small-angle pusher trajectory.
- No attachment, equality constraint, runtime object-qpos update, learned policy, MPC, or VLA was used. Main and articulated TEST scene hashes stayed frozen.

