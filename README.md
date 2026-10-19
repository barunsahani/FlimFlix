# FLIMFLIX – Netflix-style movie recommender

A Netflix-like web app (hero banner, scrolling rows, search, detail modal, **More Like This**, My List
with personalised picks) powered by an improved content-based recommendation model.

## Run it

```bash
pip install -r requirements.txt
uvicorn main:app --reload
# open http://127.0.0.1:8000
```

The trained model is already in `model/`. All metadata comes from the local dataset.

### Posters (important)
Poster paths in the 2017 dataset are often dead now (TMDB shows "File Not Found"). To load current posters,
add a **free TMDB API key**: create an account at themoviedb.org -> Settings -> API -> request a key
(copy the "API Key" v3), then copy `.env.example` to `.env` and paste it:

```
TMDB_API_KEY=your_key_here
```
Restart the server. Without a key the app still works, but outdated posters show as title tiles.
(Backdrop images exist only for franchise movies; others get a blurred-poster hero.)

## Retrain (optional)

Put `movies_metadata.csv` (Kaggle "The Movies Dataset") next to `train.py`, then:

```bash
python train.py --csv movies_metadata.csv      # ~1 min; re-tunes weights, rewrites model/ + model/report.json
```

## What changed from the original repo

| Area | Before | Now |
|---|---|---|
| Data | raw CSV; broken ids, duplicates, unreleased/no-poster rows kept | cleaned catalog of 30,266 released movies with posters |
| Features | one TF-IDF over overview + genres + tagline | 7 weighted blocks: text TF-IDF, genres, production companies, language, era, title words, franchise (TMDB collection) |
| Ranking | pure cosine similarity | cosine + quality prior + soft penalty for barely-rated titles (obscure <25-vote titles are ~3-11% of results instead of ~36-48%) |
| Personalisation | none | *Top Picks for You* from the average taste vector of your My List |
| Backend | needed TMDB key for every request + a Streamlit client | self-contained FastAPI (`/api/home`, `/api/search`, `/api/movie/{id}`, `/api/profile`, `/api/genre/{g}`) |
| Frontend | Streamlit | Netflix-style SPA in `static/` (vanilla JS, no build step) |

## Model accuracy

The dataset has no user ratings, so accuracy is measured with two proxies on **held-out franchises**
(weights were tuned on a different half; the franchise/title features are switched off during
tuning so they can't inflate the score):

| Model | Franchise Recall@10 | Franchise Hit@10 | Genre Jaccard@10 | Median votes of recs |
|---|---|---|---|---|
| Original recipe (TF-IDF on overview+genres+tagline) | 0.49 | 0.64 | 0.39 | – |
| New model, content features only | 0.46 | 0.63 | **0.85** | **246** |
| New model as served (+ franchise & title features) | ~0.99* | ~1.00* | 0.80 | 172 |

\* by construction – the franchise feature *is* the ground truth, so treat the content-only row as the honest figure.
Full numbers: `model/report.json`. Baseline was re-implemented on the same catalog, without NLTK lemmatisation.

Caveats: franchise membership is a narrow proxy for "similar"; true quality needs user feedback
(clicks / ratings), which would be the next step (e.g. add collaborative filtering on the MovieLens
`ratings.csv` that ships with the same Kaggle dataset). The "% Match" badge is a *relative* score within
each list (best = 99), not a probability.

## Files

```
main.py          FastAPI server + API
recommender.py   data cleaning, feature blocks, Recommender class
train.py         build + tune + evaluate + save model
static/          index.html, style.css, app.js
model/           features.npz, movies.csv.gz, config.json, report.json
Procfile         for Render/Heroku-style deploys
```

### A note on the trade-off
The franchise benchmark rewards finding *obscure* sequels, so its optimum was `prior_lambda = 0`. I set it to **0.1**
after reading actual recommendations (Inception → Minority Report, Interstellar, Watchmen, Oblivion; Pulp Fiction →
Snatch, Reservoir Dogs) because it surfaces well-known, relevant titles. The cost is slightly lower franchise recall
in content-only mode (0.46 vs 0.49 for the original), which the explicit franchise feature more than makes up for in
the served model. Change `prior_lambda` in `model/config.json` to tune this (0 = pure similarity).
