# Prompt 11 Status

- Scope: static scan of all 6,764 rows in `leap_left_custom_grasp_cache_cube5cm.npz`.
- Hand: original modified-left geometry with the source environment's fixed `GRASP_INIT_JOINT_POS`.
- Per row: original cube half-size and complete WXYZ pose.
- Dynamic steps: 0. No attachment, lift, or rotation was used.
- Finger-finger minimum distance: `+0.010000000 m`; self-crossing check passed.
- Rows with any hand contact: 6,764.
- Rows with thumb contact: 11.
- Rows with at least two non-thumb fingers: 2,359.
- Rows satisfying thumb plus at least two non-thumb fingers: 0.
- Rows within the 2 mm penetration limit: 2,268.
- Static candidates satisfying all gates: 0.
- Full scan log: `logs/prompt11_cache_static_scan.txt`, SHA256 `f71031da89020fdbe3cb91ab072683fa4d382a77e978f400c7c83b5e05be4455`.
- Main scene SHA256 before/after: `1e1e7666992f8e2c82ed11c05723489df0f9614f5a345021be16360eadf7f2a1`.

## Thumb-contact rows

- Rows 4482 and 5119 are penetration-safe but have zero non-thumb finger contacts.
- Row 5760 is closest to the contact gate with thumb plus middle finger, but it still lacks a second finger and penetrates the hand by `0.003264360 m`.
- The other eight thumb-contact rows have zero non-thumb fingers and exceed the 2 mm penetration limit.

## Verdict

**NO STATIC CANDIDATE — the cache does not contain a cube pose that forms a thumb-plus-two-finger grasp with the source reset hand pose.**

Do not run passive hold, lift, or rotation from this cache/reset combination.
