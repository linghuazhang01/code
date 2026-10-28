#!/usr/bin/env bash
# Run ON the machine (inside the code checkout): install the one training + evaluation environment
# (torch 2.6.0, vLLM 0.8.5.post1, transformers 4.51.3, pinned by deploy/env.lock.txt) and check it on the GPUs.
# Settings: deploy/target.env and deploy/network.env if present, otherwise everything goes next to the checkout.
#   bash deploy/install_env.sh            # = base, env, verify (+ evaluation runtime once data is installed too)
# From your own computer over SSH use: bash deploy/bootstrap.sh setup env
exec bash "$(dirname "${BASH_SOURCE[0]}")/remote/setup.sh" env "$@"
