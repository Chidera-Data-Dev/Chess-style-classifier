# ♟️ Chess Player Style Classifier

A data science/ML project that classifies Lichess rapid players into playing-style
archetypes based on their opening and development behavior, built end to end,
from data collection through an interactive Streamlit app.

<img width="918" height="506" alt="image" src="https://github.com/user-attachments/assets/6c21cada-a116-471d-b31f-a184f38fe253" />


Enter any Lichess username and get a
style profile based on their recent rated rapid games.

## What it does

Most chess analysis tools tell you *how good* a move was (via engine
evaluation). This project asks a different question: **how does this player
tend to behave in the opening and early middlegame?**

Using unsupervised clustering (K-Means) on 9,684 rated rapid players, two
distinct style archetypes emerged from the data:

- 🐇 **Rapid Developer** — pushes pawns into contested territory early,
  castles sooner, and develops more of their minor pieces quickly.
- 🐢 **Deliberate Developer** — takes longer to castle (or skips it more
  often), develops pieces more gradually, and keeps the queen home longer.

Player rating was deliberately **excluded** from the clustering features —
the goal was to find genuine stylistic variation, not just re-discover skill
level. (Notably: the two clusters show only a mild rating difference,
suggesting style and skill are related but distinct — plenty of strong
players are deliberate developers, and vice versa.)

## How it works

1. **Data collection** — usernames are mined directly from Lichess's monthly
   PGN archive dumps, filtered to rated rapid games, giving a naturally
   diverse pool across the full rating spectrum (not just top-leaderboard
   players).
2. **Feature extraction** — for each player, their own moves (not their
   opponents') are walked game-by-game with `python-chess` to compute six
   behavioral features:
   - `pct_games_never_pushed_contested_pawn`
   - `avg_castle_phase`
   - `pct_games_never_castled`
   - `pct_games_queen_never_moved`
   - `avg_minor_dev_phase`
   - `avg_minor_pieces_developed`
3. **Clustering** — features are standardized and clustered with K-Means
   (k=2, chosen via silhouette score analysis).
4. **Inference** — a Streamlit app pulls a new player's recent games live,
   computes the same features, and assigns them to the nearest trained
   cluster.

## Brief Repo content Explanation

| File | Purpose |
|---|---|
| `mine_rapid_usernames.py` | Mines a diverse pool of rated-rapid usernames from a monthly Lichess archive |
| `player_style_pipeline.py` | Fetches a player's games, extracts features, aggregates to one row per player, checkpoints to SQLite |
| `players.sqlite` | Snapshot dataset — 9,684 players' aggregated features (see note below) |

> **Note on `test_players.sqlite`:** this is a fixed snapshot collected on
> [9/20/2026] — it does not update automatically. The app uses it only to hold
> the pre-trained model artifacts' training data for reference; live
> lookups in the app pull fresh data from Lichess at query time.

## Getting the dataset

You don't need to re-run the full scraping pipeline to explore the data —
just load the included snapshot:

```python
import sqlite3
import pandas as pd

conn = sqlite3.connect("test_players.sqlite")
df = pd.read_sql("SELECT * FROM player_features", conn)
```

## Running the app

Requires `frontend.py`, `scaler_.pkl`, and `model.pkl` in the same folder.

```bash
pip install streamlit joblib pandas requests python-chess
streamlit run app.py
```

Enter any Lichess username with enough recent rated rapid games (10+) to get
a live style classification.

## Rebuilding the dataset from scratch

Only needed if you want to mine a fresh set of players rather than use the
included snapshot.

- `Usernames.ipynb` run this in your notebook to get usernames for extracting games (you could change the year and month in this)
- `game_extractor.ipynb` run this in your notebook to get extracted and aggregated games from your mined usernmes

Note that rebuilding the dataset means having all initial features which are about (16)

Then retrain K-Means on the resulting dataset and re-save `scaler_.pkl` /
`model.pkl` before redeploying the app.

## Tech stack

Python · python-chess · pandas · scikit-learn (StandardScaler, KMeans) ·
SQLite · Streamlit · Lichess public API

## Limitations & honest notes

- Style labels are based on a small, interpretable feature set focused on
  opening/development behavior, they describe a *pattern*, not a
  comprehensive personality profile.
- Cluster assignments are fixed to this specific training run; a different
  training set could shift cluster boundaries slightly.
- Lichess does not support user profile pictures, the app uses
  cluster-themed avatars in place of real photos.
