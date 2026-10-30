# Prompt 08 Status

- Scope: TEST scene only; frozen main scene was not edited.
- Source pose: mirrored MjLab 5 cm cache row 3388, cube symmetry 1.
- Held initialization: cube centre `(0, 0, 0.155) m`, 80 mm above the table-start centre; hand translated by the same 80 mm relative to Prompt 07.
- Cube-hand attachment: none. The cube retained its free joint, the compiled model had zero equality constraints, and cube qpos was not changed during the 500-step run.
- Initial hand contact: palm only (`palm_collision_6`); no thumb or finger contact.
- Last hand contact: step 5 of 500. Last thumb contact: none.
- Cube end centre: `(-0.000453915, -0.000054165, 0.074892245) m`.
- Maximum / end 3D displacement: `88.806 / 80.109 mm`.
- Z change: `-80.108 mm`; the cube fell back to the table.
- Deepest hand-cube distance: `-0.310799 mm`, within the 2 mm penetration limit.
- Solver warnings: none.
- Gate A: FAIL. Relaxed: FAIL. Passive-hold stability: FAIL.
- Main scene SHA256 before/after: `1e1e7666992f8e2c82ed11c05723489df0f9614f5a345021be16360eadf7f2a1`.

## Verdict

**REJECTED — the mirrored cache mapping is not a held grasp in the current right-hand geometry. Do not update the main scene.**

The initialization produced only transient palm contact, not thumb-plus-finger support. The 0.0285 m custom-thumb geometry mismatch recorded in Prompt 07 remains a material mapping risk.
