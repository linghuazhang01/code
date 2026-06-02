# Paper Settings for math + code Multi-Teacher OPD

## Chosen implementation setting

Use the public **G-OPD / ExOPD** math+code recipe as the first implementation
target.

- Paper: `Learning beyond Teacher: Generalized On-Policy Distillation with Reward Extrapolation`, arXiv:2602.12125
- Code: `https://github.com/RUCBM/G-OPD`
- Framework: G-OPD's verl fork, based on verl v0.6.1
- Student / reference: `Qwen/Qwen3-4B`
- Math teacher: `Qwen3-4B-Non-Thinking-RL-Math`
- Code teacher: `Qwen3-4B-Non-Thinking-RL-Code`
- Training data: `G-OPD-Training-Data/math_and_code/train.parquet`
- Validation: `AIME2024`, `AIME2025`, `Eurus/code_validation`
- Teacher routing: `extra_info.opd_teacher` is `math` or `code`
- Objective: reverse-KL OPD with ExOPD reward scaling, `lambda_vals=1.25`

This setting is selected because it is public, text-only, directly covers both
math reasoning and code generation, and already exposes the required verl hooks:
`only_reverse_kl_advantages`, `lambda_vals`, `multi_teacher_distill`, and
`opd_teacher`.

## Paper-by-paper extraction

| Source | Student | Training data | Teacher choice | Implementation role |
| --- | --- | --- | --- | --- |
| G-OPD / ExOPD, arXiv:2602.12125 | Same-size multi-teacher setting uses `Qwen3-4B-Non-Thinking` / `Qwen/Qwen3-4B` as the original student; strong-to-weak variants also evaluate smaller students. | Math uses DeepMath filtered data; code uses Eurus RL code data; official repo also provides merged `math_and_code/train.parquet`. | Domain-specific RL variants from the same base: math RL teacher for math samples, code RL teacher for code samples. | Primary runnable recipe. |
| Uni-OPD, arXiv:2605.03677 | Covers `Qwen3-4B` and smaller students across OPD settings. | Reuses similar text math/code domains and adds data-balancing / calibration recipes. | Includes single-teacher and multi-teacher OPD evaluations. | Second-stage enhancement, not first implementation baseline. |
| MiMo-V2-Flash, arXiv:2601.02780 | Large MiMo-V2-Flash MoE student. | Private/system-scale post-training data for reasoning, code, tools, safety, and agentic tasks. | Domain-specialized teachers from SFT/RL/self variants. | Industrial evidence for MOPD, not a reproducible math+code coding baseline. |
| KAT-Coder-V2, arXiv:2603.27703 | KAT-Coder-V2 built from KAT-Coder-V1 via continued post-training. | Agentic coding data across SWE, WebCoding, Terminal, WebSearch, and General. | Five domain experts independently SFT/RL trained, then unified via OPD. | Related work for code-agent specialization, not the math+code text recipe. |
| CaMOPD, arXiv:2605.27115 | Domain-specialized model as student initialization in general recovery tasks. | Proxy general prompts plus retained domain prompts. | General teacher and domain teacher with alternating updates and gap-based sample selection. | Important direct prior for conflict/control claims, but not a math+code verl recipe. |

## Conservative boundaries

- Do not claim this project is the first multi-teacher OPD or first MOPD.
- Keep code-generation settings separate from agentic coding settings.
- Treat G-OPD's "student surpasses teacher" behavior as setting-specific, not universal.
- Standard logit-based OPD requires white-box teacher log-probabilities. For API-only teachers, use a black-box-compatible alternative such as rubric-based OPD instead.
- This implementation starts with fixed domain routing. Learned teacher routing, difficulty balancing, offline teacher-logprob caching, and gradient conflict control should be separate ablations.
