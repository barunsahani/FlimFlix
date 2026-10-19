"""
Build + evaluate + save the recommender.

    python train.py --csv movies_metadata.csv

Evaluation (no user ratings exist in this dataset, so we use two honest proxies):
  * Franchise Hit@10 / Recall@10 : query a movie, count other films of the same TMDB collection
    (Toy Story -> Toy Story 2/3) retrieved in the top 10.
  * Genre Jaccard@10 : average genre overlap between the query and its top-10 recommendations.
Weights are tuned on one half of the franchises and reported on the held-out half.
During tuning the 'franchise' and 'title' blocks are switched OFF so the benchmark can't be gamed
by the very feature that defines the ground truth.
"""
import argparse, json, time
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

import recommender as R


def topk_matrix(X, qidx, bias, k=10, chunk=400):
    out = []
    XT = X.T.tocsc()
    for s in range(0, len(qidx), chunk):
        q = qidx[s:s + chunk]
        S = np.asarray((X[q] @ XT).todense())
        S = S + bias[None, :]
        S[np.arange(len(q)), q] = -1e9
        top = np.argpartition(-S, k, axis=1)[:, :k]
        order = np.argsort(-np.take_along_axis(S, top, 1), axis=1)
        out.append(np.take_along_axis(top, order, 1))
    return np.vstack(out)


def evaluate(top, qidx, d, groups, gsets):
    hit, rec, jac = [], [], []
    coll = d["collection_id"].values
    for row, q in zip(top, qidx):
        sib = groups[coll[q]] - {q}
        got = len(sib & set(row.tolist()))
        hit.append(got > 0)
        rec.append(got / min(len(sib), 10))
        gq = gsets[q]
        jac.append(np.mean([len(gq & gsets[j]) / max(1, len(gq | gsets[j])) for j in row]))
    return dict(hit10=float(np.mean(hit)), recall10=float(np.mean(rec)), genre_jaccard10=float(np.mean(jac)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="movies_metadata.csv")
    ap.add_argument("--out", default="model")
    ap.add_argument("--no-tune", action="store_true")
    a = ap.parse_args()

    t0 = time.time()
    d = R.load_catalog(a.csv)
    print(f"catalog: {len(d)} movies")
    blocks = R.build_blocks(d)
    print({k: v.shape for k, v in blocks.items()}, f"{time.time()-t0:.0f}s")

    # ---------- benchmark setup
    coll = d["collection_id"].values
    groups = {}
    for i, c in enumerate(coll):
        if c > 0:
            groups.setdefault(c, set()).add(i)
    groups = {c: s for c, s in groups.items() if len(s) >= 2}
    cids = sorted(groups)
    rng = np.random.RandomState(0)
    rng.shuffle(cids)
    half = len(cids) // 2
    tune_c, test_c = set(cids[:half]), set(cids[half:])
    q_tune = np.array(sorted(i for c in tune_c for i in groups[c]))
    q_test = np.array(sorted(i for c in test_c for i in groups[c]))
    if len(q_tune) > 1500:
        q_tune = rng.choice(q_tune, 1500, replace=False)
    gsets = [set(g) for g in d["genres_list"]]
    prior = d["prior"].values.astype(np.float32)
    votes = d["vote_count"].values
    print(f"franchises: {len(groups)} | tune queries {len(q_tune)} | held-out queries {len(q_test)}")

    def run(weights, lam, queries, X=None):
        X = R.combine(blocks, weights) if X is None else X
        return evaluate(topk_matrix(X, queries, R.make_bias(prior, votes, lam)), queries, d, groups, gsets)

    # ---------- baseline = the ORIGINAL notebook recipe, on the same catalog
    base_text = (d["overview"] + " " + d["genres_list"].apply(" ".join) + " " + d["tagline"]).apply(R._clean)
    Xb = normalize(TfidfVectorizer(max_features=50000, ngram_range=(1, 2), stop_words="english",
                                   dtype=np.float32).fit_transform(base_text))
    baseline = evaluate(topk_matrix(Xb, q_test, np.zeros(len(d), dtype=np.float32)), q_test, d, groups, gsets)
    print("BASELINE (original recipe)  held-out:", baseline)

    # ---------- tuning (franchise/title blocks OFF)
    content_only = {k: v for k, v in R.DEFAULT_WEIGHTS.items()}
    content_only["franchise"] = 0.0
    content_only["title"] = 0.0
    content_only["genres"] = 0.25
    lam = R.DEFAULT_PRIOR_LAMBDA
    best_w = dict(content_only)

    def objective(m):
        return 0.5 * m["recall10"] + 0.5 * m["genre_jaccard10"]

    if not a.no_tune:
        grid = {
            "genres": [0.25, 0.4, 0.55],
            "companies": [0.0, 0.12, 0.3],
            "language": [0.0, 0.08, 0.2],
            "era": [0.0, 0.1, 0.25],
        }
        cur = run(best_w, lam, q_tune)
        print("start", cur)
        for _ in range(2):  # two coordinate-descent passes
            for k, vals in grid.items():
                for v in vals:
                    w = dict(best_w); w[k] = v
                    m = run(w, lam, q_tune)
                    if objective(m) > objective(cur) + 1e-4:
                        best_w, cur = w, m
                        print(f"  {k}={v}  -> {m}")
        for l in [0.0, 0.03, 0.06, 0.1, 0.15]:
            m = run(best_w, l, q_tune)
            if objective(m) > objective(cur) + 1e-4:
                lam, cur = l, m
                print(f"  prior_lambda={l} -> {m}")
    print("tuned weights:", best_w, "lambda", lam)

    tuned_content = run(best_w, lam, q_test)
    print("TUNED content-only  held-out:", tuned_content)

    final_w = dict(best_w)
    final_w["title"] = R.DEFAULT_WEIGHTS["title"]
    final_w["franchise"] = R.DEFAULT_WEIGHTS["franchise"]
    Xf = R.combine(blocks, final_w)
    final = run(final_w, lam, q_test, X=Xf)
    print("FINAL (+title +franchise)   held-out:", final)

    model = R.Recommender(d, Xf, lam)
    model.save(a.out)
    json.dump({
        "catalog_size": len(d), "franchises": len(groups), "heldout_queries": int(len(q_test)),
        "baseline_original_recipe": baseline, "tuned_content_only": tuned_content,
        "final_model": final, "weights": final_w, "prior_lambda": lam,
        "note": "Baseline re-implemented on the same cleaned catalog (without NLTK lemmatisation).",
    }, open(f"{a.out}/report.json", "w"), indent=2)
    print(f"saved to {a.out}/  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
