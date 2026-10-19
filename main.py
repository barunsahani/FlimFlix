"""
FlimFLix - Netflix-style movie web app.

    uvicorn main:app --reload
    open http://127.0.0.1:8000

Everything is served from the local dataset + trained model (no TMDB API key needed;
poster images are loaded straight from TMDB's public image CDN by the browser).
"""
import datetime as dt
import json
import os
import random
import re
import threading
import time
from typing import List

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from recommender import Recommender

BASE = os.path.dirname(os.path.abspath(__file__))


def _load_env():
    """Tiny .env reader so no extra dependency is needed."""
    p = os.path.join(BASE, ".env")
    if os.path.exists(p):
        for line in open(p, encoding="utf-8"):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_env()
TMDB_API_KEY = os.environ.get("TMDB_API_KEY", "").strip()
POSTER_CACHE_FILE = os.path.join(BASE, "poster_cache.json")
try:
    POSTER_CACHE = json.load(open(POSTER_CACHE_FILE))
except Exception:
    POSTER_CACHE = {}

app = FastAPI(title="FlimFlix", version="4.0")

model: Recommender = Recommender.load(os.path.join(BASE, "model"))
meta = model.meta

GENRE_ROWS = ["Action", "Comedy", "Drama", "Science Fiction", "Thriller", "Horror",
              "Animation", "Romance", "Family", "Crime", "Adventure", "Fantasy",
              "Mystery", "War", "Documentary"]
GENRE_LABEL = {"Science Fiction": "Sci-Fi"}

# ------------------------------------------------------------- precomputed pools
_pop_rank = meta["popularity"].rank(pct=True).values
_quality = 0.5 * meta["prior"].values + 0.5 * _pop_rank          # "feels well-known & good"
_known = meta["vote_count"].values >= 200


def _pool(mask, score, size=60):
    idx = np.where(mask)[0]
    idx = idx[np.argsort(-score[idx])][:size]
    return idx.tolist()


POOLS = {
    "Trending Now": _pool(_known & (meta["year"].fillna(0).values >= 2012), _pop_rank),
    "Popular on FlimFlix": _pool(_known, meta["popularity"].values),
    "Top Rated": _pool(meta["vote_count"].values >= 1500, meta["wr"].values),
    "Critically Acclaimed Classics": _pool(
        _known & (meta["year"].fillna(9999).values < 1990), meta["wr"].values),
    "Hidden Gems": _pool(
        (meta["vote_count"].values >= 100) & (meta["vote_count"].values <= 1200)
        & (meta["wr"].values >= 7.0), meta["wr"].values),
}
for g in GENRE_ROWS:
    gm = np.array([g in gl for gl in meta["genres_list"]]) & _known
    POOLS[g] = _pool(gm, _quality)

HERO_POOL = [i for i in _pool(_known & meta["backdrop_path"].notna().values, _quality, 40)]


def _sample(pool, n=20):
    pool = list(pool)
    top = pool[:8]                       # always keep a few of the very best
    rest = random.sample(pool[8:], min(len(pool) - 8, n - len(top)))
    out = top + rest
    random.shuffle(out)
    return out


def _cards(idxs):
    return [model.card(i) for i in idxs]



# ===================================================================== LIVE TMDB DATA
# The training dataset is a 2017 snapshot, so new / upcoming films come straight from TMDB.
IMG = "https://image.tmdb.org/t/p"
_TTL = 6 * 3600
_tmdb_cache: dict = {}


_http = None
_http_lock = threading.Lock()


def _client():
    """One shared keep-alive HTTP client (opening a new TLS connection per call was timing out)."""
    global _http
    with _http_lock:
        if _http is None:
            import httpx
            _http = httpx.Client(timeout=httpx.Timeout(15.0, connect=8.0),
                                 limits=httpx.Limits(max_connections=8, max_keepalive_connections=8))
        return _http


def tmdb(path: str, **params):
    """Cached TMDB v3 GET with retries. Raises HTTPException on problems."""
    if not TMDB_API_KEY:
        raise HTTPException(404, "no_key")
    key = (path, tuple(sorted(params.items())))
    hit = _tmdb_cache.get(key)
    if hit and time.time() - hit[0] < _TTL:
        return hit[1]
    last = "TMDB unreachable"
    for attempt in range(3):
        try:
            r = _client().get(f"https://api.themoviedb.org/3{path}",
                              params={**params, "api_key": TMDB_API_KEY, "language": "en-US"})
        except Exception as e:
            last = f"TMDB unreachable ({type(e).__name__})"
            time.sleep(0.5 * (attempt + 1))
            continue
        if r.status_code == 401:
            raise HTTPException(401, "bad_key")
        if r.status_code == 404:
            raise HTTPException(404, "not_found")
        if r.status_code == 429 or r.status_code >= 500:
            last = f"TMDB busy ({r.status_code})"
            time.sleep(0.8 * (attempt + 1))
            continue
        if r.status_code != 200:
            raise HTTPException(502, f"TMDB {r.status_code}")
        data = r.json()
        _tmdb_cache[key] = (time.time(), data)
        return data
    raise HTTPException(502, last)


def _genre_map() -> dict:
    try:
        return {g["id"]: g["name"] for g in tmdb("/genre/movie/list")["genres"]}
    except HTTPException:
        return {}


_live_seen: dict = {}


def live_card(j: dict) -> dict:
    if j.get("id") is not None:
        _live_seen[int(j["id"])] = j
    gm = _genre_map()
    rd = j.get("release_date") or ""
    if j.get("genres"):
        genres = [g["name"] for g in j["genres"]]
    else:
        genres = [gm[g] for g in j.get("genre_ids", []) if g in gm]
    return {
        "id": int(j["id"]), "title": j.get("title") or "",
        "year": int(rd[:4]) if rd[:4].isdigit() else None,
        "poster": f"{IMG}/w342{j['poster_path']}" if j.get("poster_path") else "",
        "rating": round(float(j.get("vote_average") or 0), 1),
        "genres": genres, "release_date": rd, "live": True,
    }


def live_detail(j: dict) -> dict:
    c = live_card(j)
    coll = j.get("belongs_to_collection") or {}
    c.update({
        "overview": j.get("overview") or "", "tagline": j.get("tagline") or "",
        "runtime": j.get("runtime") or None, "votes": int(j.get("vote_count") or 0),
        "collection": coll.get("name"), "original_language": j.get("original_language"),
        "poster_large": f"{IMG}/w500{j['poster_path']}" if j.get("poster_path") else "",
        "backdrop": f"{IMG}/w1280{j['backdrop_path']}" if j.get("backdrop_path") else None,
    })
    return c


def _live_list(path: str, limit: int = 20, upcoming_only: bool = False):
    data = tmdb(path)
    items = [x for x in data.get("results", []) if x.get("poster_path") and x.get("title")]
    if upcoming_only:
        today = dt.date.today().isoformat()
        items = sorted([x for x in items if (x.get("release_date") or "") >= today],
                       key=lambda x: x.get("release_date") or "9999")
    return [live_card(x) for x in items[:limit]]


@app.get("/api/live")
def live():
    if not TMDB_API_KEY:
        return {"enabled": False, "rows": [], "hero": None}
    rows, hero = [], None
    for title, path, up in [("Now Playing in Theatres", "/movie/now_playing", False),
                            ("Coming Soon", "/movie/upcoming", True),
                            ("Trending This Week", "/trending/movie/week", False)]:
        try:
            cards = _live_list(path, upcoming_only=up)
        except HTTPException as e:
            if e.detail == "bad_key":
                return {"enabled": False, "rows": [], "hero": None, "error": "bad_key"}
            continue
        if cards:
            rows.append({"title": title, "movies": cards})
    try:
        for x in tmdb("/trending/movie/week").get("results", []):
            if x.get("backdrop_path") and x.get("overview") and x.get("poster_path"):
                hero = live_detail(x)
                hero["collection"] = "Trending this week"
                break
    except HTTPException:
        pass
    return {"enabled": bool(rows), "rows": rows, "hero": hero}


if TMDB_API_KEY:   # warm the text index in the background so the first "new movie" click is fast
    threading.Thread(target=model._light, daemon=True).start()

# ------------------------------------------------------------------------- API
@app.get("/api/health")
def health():
    return {"status": "ok", "movies": len(meta), "tmdb_key": bool(TMDB_API_KEY)}


@app.get("/api/home")
def home():
    hero_idx = random.choice(HERO_POOL)
    hero = model.detail(int(meta.iloc[hero_idx]["id"]))
    rows = []
    for name in ["Trending Now", "Popular on FlimFlix", "Top Rated"]:
        rows.append({"title": name, "movies": _cards(_sample(POOLS[name]))})
    for g in random.sample(GENRE_ROWS[:8], 4):
        rows.append({"title": f"{GENRE_LABEL.get(g, g)} Movies", "genre": g,
                     "movies": _cards(_sample(POOLS[g]))})
    rows.append({"title": "Critically Acclaimed Classics",
                 "movies": _cards(_sample(POOLS["Critically Acclaimed Classics"]))})
    rows.append({"title": "Hidden Gems", "movies": _cards(_sample(POOLS["Hidden Gems"]))})
    for g in GENRE_ROWS[8:]:
        rows.append({"title": f"{GENRE_LABEL.get(g, g)} Movies", "genre": g,
                     "movies": _cards(_sample(POOLS[g]))})
    return {"hero": hero, "rows": rows,
            "genres": [{"name": g, "label": GENRE_LABEL.get(g, g)} for g in GENRE_ROWS]}


@app.get("/api/genre/{name}")
def genre(name: str, limit: int = Query(60, ge=1, le=100)):
    if name not in POOLS or name not in GENRE_ROWS:
        raise HTTPException(404, "Unknown genre")
    return {"title": f"{GENRE_LABEL.get(name, name)} Movies", "movies": _cards(POOLS[name][:limit])}


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()


_titles_norm = np.array([_norm(t) for t in meta["title"]])


@app.get("/api/search")
def search(q: str = Query(..., min_length=1), limit: int = Query(30, ge=1, le=60)):
    qn = _norm(q)
    if not qn:
        return {"query": q, "movies": []}
    starts = np.char.startswith(_titles_norm, qn)
    contains = np.char.find(_titles_norm, qn) >= 0
    words = qn.split()
    allw = np.ones(len(meta), dtype=bool)
    for w in words:
        allw &= np.char.find(_titles_norm, w) >= 0
    score = (3.0 * starts + 2.0 * contains + 1.0 * allw) * 10 + _quality
    score[~(contains | allw)] = -1
    top = np.argsort(-score)[:limit]
    local = [model.card(int(i)) for i in top if score[i] > 0]
    if TMDB_API_KEY:   # also search TMDB so brand-new releases are findable
        try:
            res = tmdb("/search/movie", query=q, include_adult="false")["results"]
            live = [x for x in res if x.get("poster_path") and x.get("title")][:20]
            seen, merged = set(), []
            for x in live:
                i = int(x["id"])
                merged.append(model.card(model.id_to_idx[i]) if i in model.id_to_idx else live_card(x))
                seen.add(i)
            merged += [c for c in local if c["id"] not in seen]
            local = merged[:limit]
        except HTTPException:
            pass
    return {"query": q, "movies": local}


@app.get("/api/movie/{movie_id}")
def movie(movie_id: int, n: int = Query(18, ge=1, le=40)):
    d = model.detail(movie_id)
    if d is not None:
        d["similar"] = model.similar(movie_id, n)
        return d
    # not in the 2017 training catalog -> new release: details from TMDB, picks from the model
    try:
        j = tmdb(f"/movie/{movie_id}")
    except HTTPException as e:
        raw = _live_seen.get(int(movie_id))      # fall back to what the list already told us
        if raw is None or e.detail in ("no_key", "bad_key"):
            raise
        j = raw
    d = live_detail(j)
    try:
        d["similar"] = model.similar_to_new(d["overview"], d["tagline"], d["genres"], n)
    except Exception:
        d["similar"] = []
    try:
        rec = tmdb(f"/movie/{movie_id}/recommendations").get("results", [])
        if not rec:
            rec = tmdb(f"/movie/{movie_id}/similar").get("results", [])
        d["tmdb_similar"] = [live_card(x) for x in rec if x.get("poster_path") and x.get("title")][:12]
    except HTTPException:
        d["tmdb_similar"] = []
    return d


class Profile(BaseModel):
    ids: List[int]
    n: int = 20


@app.post("/api/profile")
def profile(p: Profile):
    return {"movies": model.for_profile(p.ids[-30:], min(max(p.n, 1), 40))}


@app.post("/api/cards")
def cards(p: Profile):
    out = []
    for i in p.ids:
        idx = model.id_to_idx.get(int(i))
        if idx is not None:
            out.append(model.card(idx))
        elif TMDB_API_KEY:
            try:
                out.append(live_card(tmdb(f"/movie/{int(i)}")))
            except HTTPException:
                pass
    return {"movies": out}


def poster_info(movie_id: int):
    """Current poster/backdrop paths for a movie (dataset paths date from 2017 and many are dead)."""
    key = str(movie_id)
    if key in POSTER_CACHE:
        return POSTER_CACHE[key]
    j = tmdb(f"/movie/{movie_id}")
    if not j.get("poster_path"):
        raise HTTPException(404, "no_poster")
    out = {
        "poster": f"{IMG}/w342{j['poster_path']}",
        "poster_large": f"{IMG}/w500{j['poster_path']}",
        "backdrop": f"{IMG}/w1280{j['backdrop_path']}" if j.get("backdrop_path") else None,
    }
    POSTER_CACHE[key] = out
    if len(POSTER_CACHE) % 5 == 0:
        try:
            json.dump(POSTER_CACHE, open(POSTER_CACHE_FILE, "w"))
        except Exception:
            pass
    return out


@app.get("/api/poster/{movie_id}")
def fresh_poster(movie_id: int):
    return poster_info(movie_id)


class IdList(BaseModel):
    ids: List[int]


@app.post("/api/posters")
def fresh_posters(p: IdList):
    """Batch version: the browser asks for many dead posters at once."""
    if not TMDB_API_KEY:
        return {"results": {}, "error": "no_key"}
    from concurrent.futures import ThreadPoolExecutor
    ids = list(dict.fromkeys(p.ids))[:40]

    def one(i):
        try:
            return i, poster_info(i)
        except HTTPException as e:
            return i, ({"_error": e.detail} if e.detail == "bad_key" else None)

    with ThreadPoolExecutor(max_workers=6) as ex:
        res = dict(ex.map(one, ids))
    if any(isinstance(v, dict) and v.get("_error") == "bad_key" for v in res.values()):
        return {"results": {}, "error": "bad_key"}
    return {"results": {str(k): v for k, v in res.items()}, "error": None}


# use already-known fresh poster paths in every card the API returns (fewer dead images)
_orig_card, _orig_detail = model.card, model.detail


def _card_fixed(i, similarity=None):
    c = _orig_card(i, similarity)
    f = POSTER_CACHE.get(str(c["id"]))
    if f and f.get("poster"):
        c["poster"] = f["poster"]
    return c


def _detail_fixed(movie_id):
    d = _orig_detail(movie_id)
    f = POSTER_CACHE.get(str(movie_id)) if d else None
    if f:
        d["poster"], d["poster_large"] = f["poster"], f["poster_large"]
        d["backdrop"] = f["backdrop"] or d.get("backdrop")
    return d


model.card, model.detail = _card_fixed, _detail_fixed


# ---------------------------------------------------------------------- frontend
app.mount("/static", StaticFiles(directory=os.path.join(BASE, "static")), name="static")


@app.get("/")
def index():
    return FileResponse(os.path.join(BASE, "static", "index.html"))
