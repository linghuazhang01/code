"""Frequency-matched control pool for the list-12 ignore / only runs (Math-only 1.7B, author 2026-11-01).

Keeps the 8 list-12 termination IDs (EOS and conclusion words) so that both pools carry the same stopping signal, and replaces
the other 70 IDs with IDs of about the same frequency that never passed the selection rule. Frequency = occurrences over steps
1-60 of the equal-weight run M17-native. One draw with a fixed seed; the draw is not repeated or inspected for a better one.
Needs the analysis directory behind analysis-output (token-set-toklog-20261019 caches, list12-by-size-20261101 tables).
Writes configs/token_selection/list12_matched_pool_math.json, and matched_pairs_math.csv and matched_share_math.csv under
analysis-output/token-importance-controls-20261101/tables/."""
import hashlib, json, sys
from pathlib import Path
import numpy as np, pandas as pd

CODE = Path(__file__).resolve().parents[1]
A = CODE / "analysis-output"
TABLES = A / "token-importance-controls-20261101/tables"
PROVENANCE = CODE / "configs/token_selection/list12_matched_pool_math.json"
sys.path.insert(0, str(A / "token-set-toklog-20261019"))
import core
from c5_scope import load

SEED = 20261101
CALIPER = 1.25  # candidate count within this ratio of the list ID's count
FIRST_SPECIAL = 151643


def sha(ids):
    return hashlib.sha256(",".join(map(str, sorted(ids))).encode()).hexdigest()


def main():
    L = sorted(map(int, json.load(open(A / "token-set-toklog-20261019/tables/d1_token_list_ids.json"))["math"]["shared"]))
    g = pd.read_csv(A / "list12-by-size-20261101/tables/groups_math.csv", keep_default_na=False)
    flag = lambda c: g[c].astype(str).eq("True")
    term = sorted(g.id[flag("in_list12") & (g["sub"] == "终止")].astype(int))
    selected = set(g.id[flag("in_1p7B") | flag("in_4B") | flag("in_list12")].astype(int))
    roles = json.load(open(CODE / "configs/token_selection/list12_role_supplements.json"))["domains"]["math"]
    length_words = {int(i) for k in ("terminate", "lookback") for i in (roles[k] if isinstance(roles[k], list) else roles[k]["ids"])}

    run, _ = load("M17-native", "math", None)
    cnt = run.cnt.sum(0, dtype=np.float64); tot = cnt.sum()
    eligible = np.zeros(core.V, bool); eligible[:FIRST_SPECIAL] = True
    eligible[list(selected | length_words)] = False
    eligible &= cnt > 0

    rng = np.random.default_rng(SEED); used = np.zeros(core.V, bool); rows = []
    for i in sorted(set(L) - set(term), key=lambda i: -cnt[i]):
        ratio = cnt / cnt[i]
        c = np.nonzero(eligible & ~used & (ratio >= 1 / CALIPER) & (ratio <= CALIPER))[0]
        inside = len(c) > 0
        j = int(rng.choice(c)) if inside else int(np.nonzero(eligible & ~used)[0][np.argmin(np.abs(np.log(cnt[eligible & ~used] / cnt[i])))])
        used[j] = True
        rows.append(dict(list_id=i, list_token=core.show(i), list_per_1000=1000 * cnt[i] / tot, matched_id=j, matched_token=core.show(j),
                         matched_per_1000=1000 * cnt[j] / tot, ratio=cnt[j] / cnt[i], candidates=len(c), inside_caliper=inside))
    P = pd.DataFrame(rows); P.to_csv(TABLES / "matched_pairs_math.csv", index=False)
    pool = sorted(term + P.matched_id.tolist())
    assert len(pool) == len(L) == 78 and not set(pool) & (set(L) - set(term))

    share = []
    for key in ("M17-native", "MC17n-math"):
        r, _ = load(key, "math", None)
        for k, n in (("list-12", L), ("matched", pool)):
            s = r.cnt[:, n].sum(1, dtype=np.float64) / r.cnt.sum(1, dtype=np.float64)
            share.append(dict(run=key, pool=k, all_steps=r.cnt[:, n].sum(dtype=np.float64) / r.cnt.sum(dtype=np.float64), step1=s[0], step15=s[14], step30=s[29], step60=s[59]))
    S = pd.DataFrame(share); S.to_csv(TABLES / "matched_share_math.csv", index=False)
    json.dump(dict(domain="math", seed=SEED, caliper=CALIPER, frequency_run="M17-native", list12_sha256=sha(L), termination_ids=term,
                   matched_ids=sorted(P.matched_id.tolist()), pool=pool, pool_sha256=sha(pool),
                   excluded=dict(selected_in_any_size=len(selected), terminate_lookback_candidates=len(length_words), first_special_id=FIRST_SPECIAL)),
              open(PROVENANCE, "w"), indent=1)
    print("termination", [core.show(i) for i in term])
    print("pool", len(pool), sha(pool)); print("eligible", int(eligible.sum()), "outside caliper", int((~P.inside_caliper).sum()))
    print(P.ratio.describe().round(3).to_string()); print(S.round(4).to_string(index=False))
    print(P.sort_values("list_per_1000", ascending=False).head(26)[["list_token", "list_per_1000", "matched_token", "matched_per_1000", "candidates"]].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
