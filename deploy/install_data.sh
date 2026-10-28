#!/usr/bin/env bash
# Run ON the machine (inside the code checkout): download training data, evaluation data (G-OPD prompts at commit
# 37371a4c, LiveCodeBench v5/v6), STUDENT_MODELS / TEACHER_MODEL and EVAL_CHECKPOINTS, all checked against the hub.
# Settings: deploy/target.env and deploy/network.env if present, otherwise everything goes next to the checkout.
#   bash deploy/install_data.sh           # = base, data, models, ckpts (+ evaluation runtime once the env is installed too)
# From your own computer over SSH use: bash deploy/bootstrap.sh setup data
exec bash "$(dirname "${BASH_SOURCE[0]}")/remote/setup.sh" data "$@"
