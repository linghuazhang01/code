# Current-Step → Next-Step config migration

The V4 Math5%/Code1% Fixed4 run started on physical GPUs 0/1/2/4 before this local migration. Its full resolved configuration remains identical to the frozen launch snapshot. No remote sync or restart was performed during this migration.

- Source commit: `806795e` (`feat(tokens): adopt revision 3 V4/V5 taxonomy`).
- Active scope: 36 V4/V5 public profiles, one regular TopLoss head-only profile, and eight helper bases. These 45 YAMLs now have canonical `next_step` filenames and compatibility aliases under their prior `current_step` names. Each alias resolves to `next_step`.
- Retired scope: four non-default tail/TC examples cannot preserve their algorithms under the implemented lagged selector. Their active YAMLs were removed; original bytes remain in source commit `806795e` and the checked-in compressed source fixture. They were not converted into a different ranking or a disabled tail.
- Every migrated public profile retains `token_id` TopLoss, strictly no tail, interval/window 1, original taxonomy and position gates, raw Control/Structure weighting, GPU topology, batch 528, and 60 steps. Only timing and six run/audit/eval/checkpoint identity fields change.
- All new public namespaces contain `-next-`; old Current-Step namespaces are never resumed.
- During migration, local backups were saved in untracked `backup/`; a clean checkout does not need them. The checked-in immutable source and resolved snapshots are compressed in `tests/fixtures/next_step_migration/`. `migration-map.json` records each migrated or archived YAML.
- `tests/test_next_step_config_migration.py` checks all 37 active configurations, aliases, exact full resolved parameter preservation, unique namespaces, retired examples, and the launched V4 snapshot. Existing runtime and taxonomy tests cover source statistics, gates, checkpoints, and same-step unit fixtures with explicit timing.
