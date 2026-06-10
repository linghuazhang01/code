# Math Eval Domain

Math evaluation data lives here, separate from MOPD math training data.

## Data

- `data/AIME24/test.parquet`
- `data/AIME25/test.parquet`
- `data/HMMT25Feb/test.parquet`
- `data/HMMT25Nov/test.parquet`
- `data/AIME2024/test.parquet`
- `data/AIME2025/test.parquet`

The first four files are the paper-eval validation set used by the MOPD
configs. `AIME2024` and `AIME2025` are kept as legacy G-OPD validation files.

## Code

- `__init__.py`: boxed-answer extraction and fallback math scoring helpers.
- `eval/data_prep/paper_eval.py`: JSONL-to-parquet conversion for AIME/HMMT.
