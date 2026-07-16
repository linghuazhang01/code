# General-Reasoner official data

The benchmark root keeps the complete official datasets and reproducible
paper-scale subsets side by side:

```text
MMLU-Pro/
  test.parquet                                  # full test set (12,032)
  subsets/openprm_style_500_seed42/
    test.parquet                                # reproducible random 500
    manifest.json                               # IDs, indices, hashes, distribution
SuperGPQA/
  test.parquet                                  # full release (26,529)
  subsets/rsa_1000_seed42/
    test.parquet                                # exact RSA public subset (1,000)
    manifest.json                               # IDs, indices, hashes, distribution
subset_summary.json
```

Generate or verify the subsets from the repository root:

```bash
python -m eval.domains.greasoner.prepare_subsets
```

Use `--force` only when intentionally regenerating outputs after a full dataset
change. The MMLU-Pro subset follows OpenPRM's reported random-500 protocol, but
OpenPRM did not publish its sample IDs or seed; therefore it is named
`openprm_style_500_seed42` rather than presented as an exact paper sample. The
SuperGPQA subset exactly follows RSA's released 1,000-example selection.
