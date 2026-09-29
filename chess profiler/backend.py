import os
import joblib
import pandas as pd
import requests
import streamlit as st

from extract_games.player_style_pipeline import (fetch_player_games, compute_single_game_stats, 
                                                 aggregate_player_stats)


# Config — fixing features to match your actual training results

FEATURE_COLUMNS = [
    "pct_games_never_pushed_contested_pawn",
    "avg_castle_phase",
    "pct_games_never_castled",
    "pct_games_queen_never_moved",
    "avg_minor_dev_phase",
    "avg_minor_pieces_developed"
]

# Fill these in yourself based on inspecting your own cluster means table —
# the keys must match the cluster numbers KMeans actually assigns (0, 1, 2, ...).
CLUSTER_LABELS = {
    0: "Deliberate Developer",
    1: "Rapid Developer",
}
CLUSTER_MESSAGES = {
    0: "**Observations:** You tend to develop more cautiously and are less likely to castle early or move the queen early. "
       "You also make fewer early contested-pawn pushes. This suggests a more restrained opening pattern, "
       "although the slower castling can leave the king exposed for longer.",
    1: "**Observations:** You tend to get your pieces into play quickly, castle earlier, "
       "and develop more of your minor pieces. You also push contested pawns more often "
       "during your games. This points toward a more active opening approach, although "
       "faster development doesn't necessarily mean better development.",
}
CLUSTER_AVATARS = {
    0: {"emoji": "🐢", "color": "#89f4ac"},
    1: {"emoji": "🐇", "color": "#ed6563"},
}
DEFAULT_AVATAR = {"emoji": "♟️", "color": "#cccccc"}

def render_avatar(cluster_id):
    avatar = CLUSTER_AVATARS.get(cluster_id, DEFAULT_AVATAR)
    st.markdown(
        f"<div style='width:100px;height:100px;border-radius:50%;"
        f"background:{avatar['color']};display:flex;align-items:center;"
        f"justify-content:center;font-size:42px;'>{avatar['emoji']}</div>",
        unsafe_allow_html=True,
    )
 
 
def render_placeholder_avatar():
    """Shown before classification has run yet (we don't know the cluster)."""
    st.markdown(
        "<div style='width:100px;height:100px;border-radius:50%;"
        "background:#e0e0e0;display:flex;align-items:center;"
        "justify-content:center;text-align:center;font-size:11px;"
        "color:#666;'>Analyzing...</div>",
        unsafe_allow_html=True,
    )

MAX_GAMES = 50
MIN_GAMES = 10 # lesser than training's min_games, since a live lookup should still try to return something

# Cached loaders — model/scaler only need to load once per session
@st.cache_resource
def load_model_artifacts():
    scaler_path = os.path.join("deliverables", "scaler_.pkl")
    kmeans_path = os.path.join("deliverables", "model.pkl")
    
    # Load and return the artifacts
    scaler = joblib.load(scaler_path)
    kmeans = joblib.load(kmeans_path)
    
    return scaler, kmeans


@st.cache_data(ttl=3600)
def fetch_lichess_profile(username):
    """Pulls basic profile info (rating, title, account info) from
    Lichess's public user endpoint. Returns None if the user doesn't exist."""
    resp = requests.get(f"https://lichess.org/api/user/{username}")
    if resp.status_code != 200:
        return None
    return resp.json()


# Core classification logic
def classify_player(username, scaler, kmeans):
    """Fetches a player's games, extracts + aggregates features, scales
    them with the SAME fitted scaler from training, and predicts their
    cluster with the SAME fitted kmeans model (never refit at inference
    time). Returns (cluster_id, feature_row_dict) or (None, error_message)."""
    games = list(fetch_player_games(username, max_games=MAX_GAMES))
    if not games:
        return None, "No rated rapid games found for this username."
 
    per_game_stats = []
    for game in games:
        stats = compute_single_game_stats(game, username)
        if stats is not None:
            per_game_stats.append(stats)
 
    if len(per_game_stats) < MIN_GAMES:
        return None, (
            f"Only found {len(per_game_stats)} usable games "
            f"(need at least {MIN_GAMES}) — try a more active player."
        )
 
    row = aggregate_player_stats(username, per_game_stats)
    feature_df = pd.DataFrame([row])[FEATURE_COLUMNS]
 
    if feature_df.isnull().any().any():
        # a feature never occurred across ALL of this player's games
        # (e.g. never castled in any game) — fill with the column's
        # training-time mean via the scaler's stored mean, so the point
        # doesn't break scaling/prediction.
        for col in FEATURE_COLUMNS:
            if feature_df[col].isnull().any():
                feature_df[col] = feature_df[col].fillna(0)
 
    scaled = scaler.transform(feature_df)
    cluster_id = int(kmeans.predict(scaled)[0])
    return cluster_id, row