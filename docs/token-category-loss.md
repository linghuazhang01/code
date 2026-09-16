# Domain × token-category Top-32 KL

The optional actor field below replaces the KL direction at each response
position. It does not add a second loss or change token-selection budgets.

```yaml
actor:
  distill_loss_builder: topk_kl
  distill_mode: topk_renormalized_reverse_kl
  topk_distill_enabled: true
  topk_distill_k: 32
  topk_distill_support_source: teacher
  topk_distill_tail_bucket: false
  topk_distill_loss_by_domain:
    default:
      control: forward
      structure: reverse
      other: reverse
    # Optional per-domain overrides (omitted keys inherit default/global).
    # code:
    #   structure: forward
```

`control` means the active frozen GlobalControl (175 IDs), not the legacy
PDTB-only Connective set. `structure` means GlobalStructure (634 IDs).
`other` is their full-vocabulary complement. See Token.md. Domain candidate
subsets still govern selection/weighting; they do not redefine these types.

The actual generated response token determines the position's category.
Both directions use the SAME teacher Top-32 token IDs, with teacher and
student separately renormalized on that support. Forward is KL(T || S);
reverse is KL(S || T). Neither is full-vocabulary KL. Teacher scores are
detached. No extra teacher forward pass is required.

Routing applies to all valid response occurrences in each configured category,
not only occurrences chosen for increased weight. Existing masks, weights,
rollout importance sampling and reductions are applied afterwards. In
particular, TopLoss selection sees the routed raw loss, so changing the
direction can also change the selected tokens. It does not freeze the old
reverse-KL ranking.

An omitted/empty map preserves the old global loss. Missing category settings
inherit `default`, then the global `distill_mode`. Unknown domains/categories,
non-Top32 support, non-teacher support, non-renormalized objectives and teacher
prefix combinations fail closed. Runtime requires response-aligned token IDs
and math/code/science domain labels. This feature does not enable native EOPD
or chosen-token PG mixing, and does not change auxiliary KL/entropy penalties.

Training and full-gradient replay share the routed actor loss. Configured-loss
audit labels use `topk32_domain_category_renormalized_kl`. Older standalone
historical audit loss reconstructions are not category-routed; use the shared
actor's configured-token-loss output for the routed objective.

Run validation locally before any remote launch. Remote training still uses
`start.sh --local` under the project's AGENTS.md rules.
