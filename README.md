# Movie Hub — A Collaborative Filtering Movie Recommendation System

A Flask web application that recommends films using **item-based collaborative
filtering with cosine similarity**, trained on the **MovieLens 20M** dataset
(20,000,263 ratings from 138,493 users).

Unlike a content-based recommender, which compares plot summaries or genres,
this system learns from behaviour: two movies are similar when the same people
tended to rate them the same way. The result is that *The Dark Knight* leads to
*Batman Begins*, *Inception* and *The Prestige* — a connection no genre tag
would ever produce.


## What it does

| Feature | Description |
|---|---|
| **Similar movies** | Every movie page lists the films whose audiences overlap most with it |
| **Recommender** | Type a film you like, get ranked recommendations with a match percentage |
| **Personal recommendations** | `/for-you` ranks the whole catalog against your own ratings and watchlist |
| **Star ratings** | Rate any film 1–5; your profile immediately changes what you are recommended |
| **Typo-tolerant search** | `godfater` finds *The Godfather*, `matrx` finds *The Matrix* |
| **Watchlist** | Save films to a personal list |
| **Accounts** | Registration, login, profile pictures, password rules |
| **Admin console** | Dashboard, catalog CRUD, user management, watchlist administration |



## The recommendation algorithm

### Why item-based

User-based collaborative filtering compares the current user against all
138,493 MovieLens users on every request. Item-based CF pre-computes
movie-to-movie similarity **once, offline**, so serving a recommendation only
means adding up a few dozen numbers. Item similarities are also far more stable
over time than user similarities, which is why production systems (Amazon's
"customers who bought this also bought") use this form.

### Training — `scripts/build_model.py`

```
data/raw/*.csv  (~900 MB)
      │
      │  python scripts/build_model.py      (~30–50 seconds)
      ▼
data/processed/movies.csv            catalog metadata      (~1 MB)
data/processed/item_similarity.npz   top-50 neighbours     (~1.2 MB)
data/processed/model_meta.json       how it was built
```

**Step 1 — Filter.** Keep movies with at least 500 ratings. This reduces
26,744 movies to **4,489**, retaining 18,717,467 of the 20M ratings.
Similarity computed from a handful of ratings is noise, not signal.

**Step 2 — Build the matrix.** The ratings become a sparse `movie × user`
matrix `R` of shape 4,489 × 138,493 with 18.7M non-zero entries. Stored densely
this would be 2.5 GB; as a CSR sparse matrix it is about 150 MB.

**Step 3 — Adjust.** Subtract each movie's mean rating from its own row:

```
R'[i, u] = R[i, u] − mean(R[i])
```

This is the *adjusted* part of adjusted cosine similarity. It removes the bias
of films almost everyone likes, leaving only how much **more or less than usual**
each viewer enjoyed that film. Without it, every popular movie looks similar to
every other popular movie.

**Step 4 — Cosine similarity.** Normalise each row to unit length, then:

```
S = R'_norm · R'_normᵀ
```

Because the rows are unit vectors, this product *is* the cosine of the angle
between every pair of movies — a score from −1 to 1. This single matrix
multiplication takes about 14 seconds.

**Step 5 — Prune.** Keep only each movie's top 50 neighbours. The full
4,489 × 4,489 matrix is 80 MB; the top-50 lists are **1.2 MB** and give
identical results for the top-N recommendations actually served.

### Serving — `app/services/recommender.py`

**Similar to one movie** — a dictionary lookup into the neighbour lists.

**Personalised** — for a user who rated movies `R`, each candidate `c` scores:

```
score(c) = Σ  sim(i, c) × (rᵢ − 3.0)     for i in R
           ─────────────────────────
              Σ  |rᵢ − 3.0|
```

Ratings are centred on 3.0, the midpoint of the scale, so a 5-star rating pulls
its neighbours **up** and a 1-star rating pushes them **down**. A 3-star rating
contributes nothing, which is correct: it carries no preference information.
Watchlisted-but-unrated films count as a weak positive (4.0). Anything the user
has already rated or saved is removed before ranking.

The divisor is constant for a given user, so it cannot change the ordering — it
just keeps the score inside 0–1 and comparable between users with different
numbers of ratings.

### The cold-start problem

Two cases, both handled explicitly rather than hidden:

- **New user, no ratings** → `/for-you` falls back to the most-rated titles in
  the dataset and says so on the page.
- **Movie with no rating history** (only possible for entries an admin added by
  hand) → the recommender raises `ColdStartError` and the UI explains that there
  is nothing to collaborate on yet.


---

## Architecture

```
Browser
   │
   ▼
Flask blueprints          main / auth / admin        app/views/
   │
   ▼
Service layer             MovieCatalog               app/services/catalog.py
                          RecommendationEngine       app/services/recommender.py
   │                      SimilarityIndex
   ├─────────────► data/processed/   read-only trained model (loaded once)
   └─────────────► instance/site.db  users, ratings, watchlists (SQLite)
```

**Two data stores, on purpose.** The catalog and the trained model are
read-mostly artifacts that every user shares, so they are held in memory and
loaded once at start-up. Users, ratings and watchlists are per-user mutable
state, so they live in SQLite behind SQLAlchemy.

**Nothing expensive happens in a request.** The catalog parses `movies.csv`
once and keeps three indexes: `movie_id → Movie` for O(1) detail lookups,
normalised title → movie for exact matches, and an inverted `token → movie ids`
index for search. It reloads only when the file's modification time changes,
which happens right after an admin edit.

Measured on the pages themselves: home 4 ms, movie detail 2.5 ms, search
0.1–6 ms, personalised recommendations 0.2 ms.

---

## Project structure

```
movie-recommendation-system/
├── app/
│   ├── __init__.py             create_app() factory, error handlers, template globals
│   ├── config.py               environment-driven configuration
│   ├── extensions.py           db / bcrypt / login_manager singletons
│   ├── bootstrap.py            first-run tables, default admin, orphan cleanup
│   ├── models.py               User, UserWatchlist, UserRating
│   ├── forms.py                Flask-WTF forms and shared validators
│   ├── decorators.py           @admin_required
│   ├── services/
│   │   ├── catalog.py          in-memory catalog, indexes, search, admin writes
│   │   └── recommender.py      SimilarityIndex + RecommendationEngine
│   ├── utils/
│   │   ├── text.py             normalisation and tokenising (LRU-cached)
│   │   ├── posters.py          generated SVG poster placeholders
│   │   ├── media.py            profile picture uploads
│   │   └── youtube.py          trailer lookup, degrades gracefully offline
│   ├── views/
│   │   ├── main.py             browsing, search, recommender, ratings, watchlist
│   │   ├── auth.py             register, login, logout, account
│   │   └── admin.py            dashboard and CRUD
│   ├── static/
│   └── templates/              Jinja2 HTML Templates
├── data/
│   ├── raw/                    ← put the MovieLens CSVs here (not shipped)
│   └── processed/              trained model artifacts (shipped, ~2.2 MB)
├── instance/
│   └── site.db
├── scripts/
│   └── build_model.py          the offline training pipeline
├── .vscode/                    launch configs and editor settings
├── requirements.txt
├── run.py
└── README.md
```

---

## Setup in VS Code

### Prerequisites

- **Python 3.11 or newer** — check with `python --version`
- **VS Code** with the **Python** extension (`ms-python.python`)

### 1. Open the project

Unzip the archive, then in VS Code: **File → Open Folder…** and pick the
`movie-recommendation-system` folder (the one containing `run.py`).

VS Code will offer the recommended extensions from `.vscode/extensions.json` —
accepting them gives you Jinja template highlighting and Tailwind class
completion.

### 2. Create a virtual environment

Open the integrated terminal with **Ctrl + `** (backtick), then:

**Windows (PowerShell)**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

> If PowerShell blocks the script, run this once:
> `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned`

**macOS / Linux**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

Your prompt should now start with `(.venv)`.

### 3. Select the interpreter

Press **Ctrl + Shift + P** → type `Python: Select Interpreter` → choose the one
inside `./.venv`. This is what makes debugging and the Test Explorer work.

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Create your `.env`

**Windows**
```powershell
copy .env.example .env
```

**macOS / Linux**
```bash
cp .env.example .env
```

The defaults work as-is for local development.

### 6. Run it

```bash
python run.py
```

Open **http://127.0.0.1:5000/**.

Or press **F5** and choose **"Flask: run the app"** to run with the debugger
attached, so you can set breakpoints in the recommender and step through it.

### 7. Log in

The database and a default administrator are created automatically on first
run:

- **Email:** `admin@moviehub.com`
- **Password:** `Admin123!`

Change `ADMIN_PASSWORD` in `.env` before using this anywhere but your own
machine.

### Try it out

1. Register a normal account (the password needs upper, lower, a digit and a symbol).
2. Open a few films you like and rate them.
3. Visit **For you** — the recommendations will have changed.
4. Go to **Recommender** and type a title, deliberately misspelled.
5. Log in as the admin to see the dashboard and the model's statistics.

---

## Rebuilding the model

**You do not need to do this to run the project** — the trained artifacts ship
in `data/processed/`. Rebuild only if you want to change the parameters or
regenerate from scratch.

1. Download **MovieLens 20M** from
   <https://grouplens.org/datasets/movielens/20m/>
2. Copy the CSV files directly into `data/raw/` (see `data/raw/README.md`)
3. Run:

```bash
python scripts/build_model.py
```

Tunable parameters:

```bash
python scripts/build_model.py --min-ratings 1000   # smaller, more popular catalog
python scripts/build_model.py --neighbours 100     # deeper neighbour lists
python scripts/build_model.py --keywords 20        # more tags per movie
```

The script streams `rating.csv` in chunks and peaks at roughly 1.5 GB of RAM.



## Configuration

Everything is environment-driven; see `.env.example` for the full list.

| Variable | Default | Purpose |
|---|---|---|
| `FLASK_ENV` | `development` | `development`, `production` or `testing` |
| `SECRET_KEY` | dev placeholder | **required in production** — the app refuses to start otherwise |
| `DATABASE_URL` | `sqlite:///instance/site.db` | any SQLAlchemy URL |
| `DATA_DIR` | `./data/processed` | where the trained artifacts live |
| `ADMIN_USERNAME` / `ADMIN_EMAIL` / `ADMIN_PASSWORD` | `admin` / `admin@moviehub.com` / `Admin123!` | bootstrap administrator |
| `WATCHLIST_MAX_ITEMS` | `20` | per-user watchlist cap |
| `RECOMMENDER_RESULT_LIMIT` | `12` | recommendations per query |
| `TMDB_API_KEY` | empty | enables `scripts/fetch_posters.py` |


---

## Design decisions

**Why the training is a separate script.** Parsing 20M ratings and multiplying
a 4,489 × 138,493 matrix takes tens of seconds and over a gigabyte of RAM. A web
request cannot afford either. Splitting "train offline, serve online" is how
real recommender systems are built, and it keeps the app's start-up under a
second.

**Why only the top 50 neighbours are stored.** The full similarity matrix is
80 MB. Serving a top-12 recommendation never looks past the top 50, so storing
more costs memory and buys nothing.

**Why no pandas at runtime.** The catalog is 4,489 rows. A DataFrame offers
nothing at that size, and dropping the import removes several seconds from
start-up. The web app uses the standard library's `csv` module; pandas and scipy
are only needed by `scripts/build_model.py`.

**Why `fuzzywuzzy` was removed.** The previous search ran five
`partial_ratio` calls against every row of the catalog for every query. Search
now generates candidates through an inverted token index, and only falls back to
fuzzy matching when that misses — correcting the query's words against the
vocabulary of words that actually appear in titles, which is both faster and more
accurate than comparing against whole titles. Typical search: 0.1 ms literal,
5 ms with a typo.

**Why ratings are their own table.** The watchlist alone is a weak, binary
signal. Explicit 1–5 ratings on MovieLens' own scale can be compared directly
against the trained model with no rescaling, and let a user express dislike —
which is what makes the negative half of the scoring formula meaningful.

**Why MovieLens keywords replace plot summaries.** MovieLens has no overview,
cast or director fields. Rather than leave the detail page empty or invent data,
the build script extracts each film's most relevant tags from the *tag genome*
(1,128 curated tags scored for relevance across the dataset). These are the
dataset's own descriptive layer and are honest about their provenance.

**Security.** Passwords are bcrypt-hashed. All state-changing routes are
POST-only and CSRF-protected. `?next=` and `request.referrer` redirects are
validated against the current host, closing the open-redirect hole in the
original code. Uploaded filenames are never trusted — only an allow-listed
extension is kept and the stored name is random. Login failures return one
message for both "no such email" and "wrong password", so the form cannot be
used to enumerate accounts. The last administrator cannot be deleted or demoted.

---

## Dataset credit

F. Maxwell Harper and Joseph A. Konstan. 2015. *The MovieLens Datasets: History
and Context.* ACM Transactions on Interactive Intelligent Systems 5, 4:
19:1–19:19. <https://doi.org/10.1145/2827872>
