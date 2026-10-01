#!/usr/bin/env bash
set -euo pipefail

cd /home/shuang_qiu/mopd_code

OUTPUT_ROOT="$PWD/experiments_records/eval"
QUEUE_NAME="mc_eval8_v4_v5_step60_20260929"
FAILED_MARKER="$OUTPUT_ROOT/${QUEUE_NAME}.FAILED"
SUCCESS_MARKER="$OUTPUT_ROOT/${QUEUE_NAME}.SUCCESS"

on_exit() {
  status=$?
  if [[ "$status" -ne 0 ]]; then
    printf 'status=%s finished=%s\n' "$status" "$(date -Is)" > "$FAILED_MARKER"
  fi
}
trap on_exit EXIT

export PYTHON=/root/miniconda3/envs/mopd-verl/bin/python
export PYTHONPATH="$PWD:$PWD/third_party/verl:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export GPU_IDS=0,1,2,3,4,5,6,7

GOPD_DIR="$PWD/data/G-OPD-Training-Data/.eval-source/G-OPD"
DATASETS="aime24,aime25,hmmt25feb,hmmt25nov,humaneval_plus,mbpp_plus,lcb_v5,lcb_v6"
HUMANEVAL_SOURCE="$GOPD_DIR/code_eval/data/HumanEvalPlus.jsonl"
MBPP_SOURCE="$GOPD_DIR/code_eval/data/MbppPlus.jsonl"

archive_incomplete_rescore() {
  local output_dir="$1"
  if [[ -d "$output_dir" && ! -f "$output_dir/SUCCESS" ]]; then
    local archive_path="${output_dir}.failed-$(date +%Y%m%d_%H%M%S)"
    mv "$output_dir" "$archive_path"
    printf '[%s] archived incomplete rescore %s -> %s\n' \
      "$(date -Is)" "$output_dir" "$archive_path"
  fi
}

run_one() {
  local run_tag="$1"
  local model_path="$2"
  local suite_root="$OUTPUT_ROOT/$run_tag"
  local records_path="$suite_root/huggingface/code/thinking_eval_samples.jsonl"
  local rescore_root="$suite_root/huggingface/evalplus_official"

  if [[ -f "$suite_root/SUCCESS" ]]; then
    printf '[%s] reusing completed generation %s\n' "$(date -Is)" "$run_tag"
  else
    printf '[%s] starting %s model=%s\n' "$(date -Is)" "$run_tag" "$model_path"
    eval_args=(
      bash start.sh --eval --local
      --model_path "$model_path"
      --run_tag "$run_tag"
      --datasets "$DATASETS"
      --gopd_dir "$GOPD_DIR"
      --gpus 8
      --gpu_ids "$GPU_IDS"
      --shards_per_dataset 16
    )
    [[ ! -d "$suite_root" ]] || eval_args+=(--resume)
    "${eval_args[@]}"
  fi

  [[ -f "$records_path" ]]
  if [[ -f "$rescore_root/SUCCESS" ]]; then
    printf '[%s] reusing completed EvalPlus rescore for %s\n' "$(date -Is)" "$run_tag"
  else
    archive_incomplete_rescore "$rescore_root"
    printf '[%s] starting EvalPlus rescore for %s\n' "$(date -Is)" "$run_tag"
    HUMANEVAL_SOURCE="$HUMANEVAL_SOURCE" \
    MBPP_SOURCE="$MBPP_SOURCE" \
    SLURM_CPUS_PER_TASK=64 \
      bash scripts/slurm_evalplus_rescore_worker.sh \
        "$run_tag" \
        "$records_path" \
        "$rescore_root"
  fi
  printf '[%s] completed %s\n' "$(date -Is)" "$run_tag"
}

run_one \
  "eval8_mc_k8_v4_4g_step60_20260929" \
  "$PWD/checkpoints/MOPD/q1p7b4g-mc-v4-r3-cs-next-m05c01-f4-b528-s60/global_step_60/actor/huggingface"
run_one \
  "eval8_mc_k8_v5_8g_step60_20260929" \
  "$PWD/checkpoints/HF_RESTORED/q1p7b8g-mc-v5-r3-cs-next-m05c01-f4-b528-s60/global_step_60/actor/huggingface"

printf 'finished=%s\n' "$(date -Is)" > "$SUCCESS_MARKER"
trap - EXIT
