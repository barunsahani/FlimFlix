"""
Content-based movie recommender (v2).

Improvements over the original notebook model (TF-IDF over overview+genres+tagline):
  * Data cleaning  - broken ids / duplicate ids removed, unreleased & adult titles dropped,
                     movies without a poster or overview dropped.
  * Multi-block features, each block L2-normalised and weighted, so the final cosine
    similarity is a weighted blend of:
        text      TF-IDF (1-2 grams, sublinear tf, df-pruned) on overview + tagline
        genres    TF-IDF weighted one-hot (rare genres count more)
        companies production companies
        language  original language
        era       release decade (+ half weight for neighbouring decades)
        title     title words (catches sequels / remakes: "Toy Story 2" ~ "Toy Story 3")
        franchise TMDB collection id (explicit sequel / franchise link)
  * Quality prior  - Bayesian weighted rating blended into the ranking so obscure,
                     barely-rated titles don't crowd the results.
  * Taste profiles - recommend from a *set* of movies (My List) by averaging vectors.
"""
from __future__ import annotations

import ast
import gzip
import json
import os
import re
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

IMG_BASE = "https://image.tmdb.org/t/p"

# Tuned on the franchise benchmark (see train.py / model/report.json)
DEFAULT_WEIGHTS = {
    "text": 1.0,
    "genres": 0.25,
    "companies": 0.12,
    "language": 0.08,
    "era": 0.0,
    "title": 0.12,
    "franchise": 0.60,
}
DEFAULT_PRIOR_LAMBDA = 0.03


def make_bias(prior, vote_count, lam, floor: int = 25, penalty: float = 0.10):
    """Additive ranking bias: quality prior, minus a soft penalty for barely-rated titles."""
    return (lam * np.asarray(prior) - penalty * (np.asarray(vote_count) < floor)).astype(np.float32)


# --------------------------------------------------------------------------- data
def _safe_list(x, key="name") -> list:
    if not isinstance(x, str) or not x.strip():
        return []
    try:
        v = ast.literal_eval(x)
        return [i[key] for i in v if isinstance(i, dict) and key in i]
    except Exception:
        return []


def _safe_dict(x) -> dict:
    if not isinstance(x, str) or not x.strip():
        return {}
    try:
        v = ast.literal_eval(x)
        return v if isinstance(v, dict) else {}
    except Exception:
        return {}


def load_catalog(csv_path: str, min_votes: int = 5) -> pd.DataFrame:
    d = pd.read_csv(csv_path, low_memory=False)
    d["id"] = pd.to_numeric(d["id"], errors="coerce")
    d = d.dropna(subset=["id"]).copy()
    d["id"] = d["id"].astype(int)
    d = d.drop_duplicates("id")
    for c in ["vote_count", "vote_average", "popularity", "runtime"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d["vote_count"] = d["vote_count"].fillna(0)
    d["vote_average"] = d["vote_average"].fillna(0)
    d["popularity"] = d["popularity"].fillna(0)

    d = d[d["status"].eq("Released")]
    d = d[d["adult"].astype(str).str.lower().ne("true")]
    d = d[d["poster_path"].notna() & d["title"].notna()]
    d["overview"] = d["overview"].fillna("").astype(str)
    d["tagline"] = d["tagline"].fillna("").astype(str)
    d = d[d["overview"].str.len() > 20]
    d = d[d["vote_count"] >= min_votes]
    # same title + same year = duplicate listing; keep the best-voted one
    d["year"] = pd.to_datetime(d["release_date"], errors="coerce").dt.year
    d = d.sort_values("vote_count", ascending=False)
    d = d.drop_duplicates(["title", "year"]).reset_index(drop=True)

    d["genres_list"] = d["genres"].apply(_safe_list)
    d["companies_list"] = d["production_companies"].apply(_safe_list)
    coll = d["belongs_to_collection"].apply(_safe_dict)
    d["collection_id"] = coll.apply(lambda c: c.get("id", -1)).astype(int)
    d["collection_name"] = coll.apply(lambda c: c.get("name", ""))
    d["backdrop_path"] = coll.apply(lambda c: c.get("backdrop_path"))
    # Bayesian weighted rating (IMDB formula)
    C = d["vote_average"].mean()
    m = d["vote_count"].quantile(0.60)
    v, R = d["vote_count"], d["vote_average"]
    d["wr"] = v / (v + m) * R + m / (v + m) * C
    # quality prior in [0,1]: blend of rating & log popularity (vote count) ranks
    d["prior"] = 0.6 * d["wr"].rank(pct=True) + 0.4 * np.log1p(d["vote_count"]).rank(pct=True)
    return d


# ------------------------------------------------------------------- features
_clean_re = re.compile(r"[^a-z0-9\s]")


def _clean(t: str) -> str:
    return _clean_re.sub(" ", str(t).lower())


def _title_key(t: str) -> str:
    # strip digits / roman numerals so sequels share tokens
    t = _clean(t)
    t = re.sub(r"\b(ii|iii|iv|v|vi|vii|viii|ix|x|part|chapter|the)\b", " ", t)
    t = re.sub(r"\d+", " ", t)
    return t


def build_blocks(d: pd.DataFrame) -> Dict[str, sp.csr_matrix]:
    blocks: Dict[str, sp.csr_matrix] = {}

    text = (d["overview"] + " " + d["tagline"]).apply(_clean)
    blocks["text"] = TfidfVectorizer(
        ngram_range=(1, 2), stop_words="english", sublinear_tf=True,
        min_df=2, max_df=0.4, max_features=120_000, dtype=np.float32,
    ).fit_transform(text)

    ident = lambda x: x  # noqa: E731
    blocks["genres"] = TfidfVectorizer(
        analyzer=ident, lowercase=False, dtype=np.float32
    ).fit_transform(d["genres_list"])
    blocks["companies"] = TfidfVectorizer(
        analyzer=ident, lowercase=False, min_df=2, dtype=np.float32
    ).fit_transform(d["companies_list"])
    blocks["language"] = TfidfVectorizer(
        analyzer=ident, lowercase=False, dtype=np.float32
    ).fit_transform(d["original_language"].fillna("xx").apply(lambda x: [x]))

    def era_tokens(y):
        if pd.isna(y):
            return []
        dec = int(y) // 10 * 10
        return [f"d{dec}", f"d{dec}", f"d{dec-10}", f"d{dec+10}"]  # neighbours get half weight

    blocks["era"] = TfidfVectorizer(
        analyzer=ident, lowercase=False, use_idf=False, dtype=np.float32
    ).fit_transform(d["year"].apply(era_tokens))

    blocks["title"] = TfidfVectorizer(
        ngram_range=(1, 1), min_df=2, max_df=0.05, dtype=np.float32
    ).fit_transform(d["title"].apply(_title_key))

    has_coll = d["collection_id"].values > 0
    coll_tokens = [[f"c{c}"] if h else [] for c, h in zip(d["collection_id"], has_coll)]
    blocks["franchise"] = TfidfVectorizer(
        analyzer=ident, lowercase=False, use_idf=False, dtype=np.float32
    ).fit_transform(coll_tokens)
    return blocks


def combine(blocks: Dict[str, sp.csr_matrix], weights: Dict[str, float]) -> sp.csr_matrix:
    parts = []
    for name, w in weights.items():
        if w > 0 and name in blocks:
            parts.append(normalize(blocks[name]) * np.sqrt(w))
    X = sp.hstack(parts, format="csr")
    return normalize(X).astype(np.float32)


# ---------------------------------------------------------------------- model
class Recommender:
    def __init__(self, meta: pd.DataFrame, X: sp.csr_matrix,
                 prior_lambda: float = DEFAULT_PRIOR_LAMBDA):
        self.meta = meta.reset_index(drop=True)
        self.X = X.tocsr()
        self.XT = self.X.T.tocsc()
        self.prior = self.meta["prior"].values.astype(np.float32)
        self.votes = self.meta["vote_count"].values
        self.prior_lambda = prior_lambda
        self.bias = make_bias(self.prior, self.votes, prior_lambda)
        self.id_to_idx = {int(i): k for k, i in enumerate(self.meta["id"].values)}
        self._titles_norm = self.meta["title"].str.lower().values

    # ---- persistence
    def save(self, folder: str):
        os.makedirs(folder, exist_ok=True)
        sp.save_npz(os.path.join(folder, "features.npz"), self.X, compressed=True)
        keep = ["id", "title", "year", "overview", "tagline", "poster_path", "backdrop_path",
                "vote_average", "vote_count", "popularity", "runtime", "wr", "prior",
                "collection_id", "collection_name", "original_language"]
        out = self.meta[keep].copy()
        out["genres"] = self.meta["genres_list"].apply(json.dumps)
        out.to_csv(os.path.join(folder, "movies.csv.gz"), index=False, compression="gzip")
        with open(os.path.join(folder, "config.json"), "w") as f:
            json.dump({"prior_lambda": self.prior_lambda}, f)

    @classmethod
    def load(cls, folder: str) -> "Recommender":
        X = sp.load_npz(os.path.join(folder, "features.npz"))
        meta = pd.read_csv(os.path.join(folder, "movies.csv.gz"))
        meta["genres_list"] = meta["genres"].apply(json.loads)
        meta["collection_name"] = meta["collection_name"].fillna("")
        cfg = json.load(open(os.path.join(folder, "config.json")))
        return cls(meta, X, cfg.get("prior_lambda", DEFAULT_PRIOR_LAMBDA))

    # ---- scoring
    def _rank(self, qvec: sp.csr_matrix, exclude: Iterable[int], n: int,
              prior_lambda: Optional[float] = None):
        bias = self.bias if prior_lambda is None else make_bias(self.prior, self.votes, prior_lambda)
        sims = np.asarray((self.X @ qvec.T).todense()).ravel()
        score = sims + bias
        for e in exclude:
            score[e] = -1e9
        top = np.argpartition(-score, min(n, len(score) - 1))[:n]
        top = top[np.argsort(-score[top])]
        return top, sims[top]

    def similar(self, movie_id: int, n: int = 20) -> List[dict]:
        idx = self.id_to_idx.get(int(movie_id))
        if idx is None:
            return []
        # also hide duplicate listings of the same title
        same_title = np.where(self._titles_norm == self._titles_norm[idx])[0].tolist()
        top, sims = self._rank(self.X[idx], same_title, n)
        return self._cards_with_match(top, sims)

    def for_profile(self, movie_ids: List[int], n: int = 20) -> List[dict]:
        idxs = [self.id_to_idx[int(i)] for i in movie_ids if int(i) in self.id_to_idx]
        if not idxs:
            return []
        q = sp.csr_matrix(self.X[idxs].mean(axis=0))
        q = normalize(q)
        top, sims = self._rank(q, idxs, n)
        return self._cards_with_match(top, sims)

    # ---- movies that are NOT in the training catalog (new releases from TMDB)
    def _light(self):
        """Small text+genre index over the catalog, built lazily, used to place unseen movies."""
        if getattr(self, "_light_idx", None) is None:
            text = (self.meta["overview"].fillna("") + " " + self.meta["tagline"].fillna("")).apply(_clean)
            tv = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", sublinear_tf=True,
                                 min_df=2, max_df=0.4, max_features=120_000, dtype=np.float32)
            Xt = normalize(tv.fit_transform(text))
            gv = TfidfVectorizer(analyzer=lambda x: x, lowercase=False, dtype=np.float32)
            Xg = normalize(gv.fit_transform(self.meta["genres_list"]))
            X = normalize(sp.hstack([Xt, Xg * np.sqrt(0.4)]).tocsr())
            self._light_idx = (tv, gv, X)
        return self._light_idx

    def similar_to_new(self, overview: str, tagline: str, genres: List[str], n: int = 18,
                       exclude_ids: Iterable[int] = ()) -> List[dict]:
        tv, gv, X = self._light()
        qt = normalize(tv.transform([_clean((overview or "") + " " + (tagline or ""))]))
        qg = normalize(gv.transform([[g for g in genres if g]])) * np.sqrt(0.4)
        q = normalize(sp.hstack([qt, qg]).tocsr())
        sims = np.asarray((X @ q.T).todense()).ravel()
        score = sims + self.bias
        for i in exclude_ids:
            if int(i) in self.id_to_idx:
                score[self.id_to_idx[int(i)]] = -1e9
        top = np.argpartition(-score, n)[:n]
        top = top[np.argsort(-score[top])]
        return self._cards_with_match(top, sims[top])

    # ---- presentation
    def _cards_with_match(self, top, sims) -> List[dict]:
        """'Match %' is a *relative* score inside the list (best = 99), like streaming UIs show."""
        smax = float(max(sims.max(), 1e-6)) if len(sims) else 1.0
        return [self.card(i, similarity=(0.60 + 0.39 * float(s) / smax)) for i, s in zip(top, sims)]

    def card(self, i: int, similarity: Optional[float] = None) -> dict:
        r = self.meta.iloc[int(i)]
        c = {
            "id": int(r["id"]),
            "title": r["title"],
            "year": None if pd.isna(r["year"]) else int(r["year"]),
            "poster": f"{IMG_BASE}/w342{r['poster_path']}",
            "rating": round(float(r["vote_average"]), 1),
            "genres": list(r["genres_list"]),
        }
        if similarity is not None:
            c["match"] = int(round(max(0.0, min(1.0, similarity)) * 100))
        return c

    def detail(self, movie_id: int) -> Optional[dict]:
        idx = self.id_to_idx.get(int(movie_id))
        if idx is None:
            return None
        r = self.meta.iloc[idx]
        c = self.card(idx)
        c.update({
            "overview": r["overview"],
            "tagline": r["tagline"] if isinstance(r["tagline"], str) else "",
            "runtime": None if pd.isna(r["runtime"]) else int(r["runtime"]),
            "votes": int(r["vote_count"]),
            "collection": r["collection_name"] or None,
            "original_language": r["original_language"] if isinstance(r["original_language"], str) else None,
            "poster_large": f"{IMG_BASE}/w500{r['poster_path']}",
            "backdrop": (f"{IMG_BASE}/w1280{r['backdrop_path']}"
                         if isinstance(r["backdrop_path"], str) and r["backdrop_path"] else None),
        })
        return c
