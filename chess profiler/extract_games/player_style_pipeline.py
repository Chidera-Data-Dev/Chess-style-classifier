"""
Chess player style-clustering pipeline
========================================
For each username: pulls their recent rated rapid games from Lichess,
extracts per-game style features (pawn push timing, castling, queen/minor
piece development, capture rate, check rate, game length, opening
diversity, rating), aggregates to ONE row per player, and checkpoints
progress to SQLite so a crash partway through a large batch doesn't lose
already-processed players.

Install dependencies first:
    pip install python-chess requests

Usage:
    python player_style_pipeline.py --usernames usernames.txt --db players.sqlite --max-games 50 --min-games 20

`usernames.txt` = one Lichess username per line.
"""

import argparse
import io
import sqlite3
import sys
import time

import chess
import chess.pgn
import requests

LICHESS_GAMES_URL = "https://lichess.org/api/games/user/{username}"

CONTESTED_RANK_WHITE = 4  # squares.rank index (0-based) — rank 5 on the board
CONTESTED_RANK_BLACK = 3  # rank 4 on the board


# ---------------------------------------------------------------------------
# Stage 1: fetch one player's games
# ---------------------------------------------------------------------------

def fetch_player_games(username, max_games=50, perf_type="rapid", max_retries=3):
    """Streams one player's rated games from Lichess, yields parsed
    chess.pgn.Game objects. Retries with backoff on rate limiting (429)."""
    params = {
        "max": max_games,
        "rated": "true",
        "perfType": perf_type,
        "clocks": "true",
        "opening": "true",
        "pgnInJson": "false"
    }
    url = LICHESS_GAMES_URL.format(username=username)

    for attempt in range(max_retries):
        resp = requests.get(url, params=params,
                             headers = {"Accept": "application/x-chess-pgn", "User-Agent": "ChessPlayerStyleAnalysis/1.0 (personal research)"})
        if resp.status_code == 429:
            wait = 2 ** (attempt + 1)
            print(f"  rate limited on {username}, waiting {wait}s...", file=sys.stderr)
            time.sleep(wait)
            continue
        if resp.status_code != 200:
            print(f"  {username}: HTTP {resp.status_code}, skipping", file=sys.stderr)
            return
        break
    else:
        print(f"  {username}: gave up after {max_retries} retries", file=sys.stderr)
        return

    text_stream = io.StringIO(resp.text)
    while True:
        game = chess.pgn.read_game(text_stream)
        if game is None:
            break
        yield game


# ---------------------------------------------------------------------------
# Stage 2: per-game feature extraction, filtered to ONE player's moves
# ---------------------------------------------------------------------------

def compute_single_game_stats(game, username):
    """Walks one game, extracting style signals ONLY from the target
    player's own moves. Returns a dict of raw per-game stats, or None if
    the game can't be attributed to this player (shouldn't normally
    happen given the API call is scoped to this user)."""
    headers = game.headers
    if headers.get("White", "").lower() == username.lower():
        target_is_white = True
        rating = headers.get("WhiteElo")
    elif headers.get("Black", "").lower() == username.lower():
        target_is_white = False
        rating = headers.get("BlackElo")
    else:
        return None

    try:
        rating = int(rating)
    except (TypeError, ValueError):
        rating = None

    board = game.board()
    total_plies = sum(1 for _ in game.mainline_moves())
    if total_plies == 0:
        return None

    ply = 0
    target_move_count = 0
    captures = 0
    checks = 0

    pawn_push_phase = None       # first advanced pawn push, this player only
    castle_phase = None          # None until they castle
    castled = False
    queen_dev_phase = None
    minor_dev_phases = []        # one entry per minor piece that develops

    # starting squares for minor pieces / queen, by color
    if target_is_white:
        queen_start = chess.D1
        minor_starts = [chess.B1, chess.G1, chess.C1, chess.F1]  # N,N,B,B
    else:
        queen_start = chess.D8
        minor_starts = [chess.B8, chess.G8, chess.C8, chess.F8]

    developed_minor_starts = set()

    node = game
    while node.variations:
        next_node = node.variations[0]
        move = next_node.move
        mover_is_white = board.turn
        is_target_move = (mover_is_white == target_is_white)

        piece = board.piece_at(move.from_square)
        is_capture = board.is_capture(move)
        is_check = board.gives_check(move)
        is_kingside = board.is_kingside_castling(move)
        is_queenside = board.is_queenside_castling(move)

        if is_target_move:
            ply_local = ply + 1
            phase = ply_local / total_plies
            target_move_count += 1

            if is_capture:
                captures += 1
            if is_check:
                checks += 1

            if is_kingside or is_queenside:
                castled = True
                if castle_phase is None:
                    castle_phase = phase

            if piece is not None:
                if piece.piece_type == chess.PAWN and pawn_push_phase is None:
                    to_rank = chess.square_rank(move.to_square)
                    contested = (to_rank >= CONTESTED_RANK_WHITE if target_is_white
                                 else to_rank <= CONTESTED_RANK_BLACK)
                    if contested:
                        pawn_push_phase = phase

                if (piece.piece_type == chess.QUEEN
                        and move.from_square == queen_start
                        and queen_dev_phase is None):
                    queen_dev_phase = phase

                if (piece.piece_type in (chess.KNIGHT, chess.BISHOP)
                        and move.from_square in minor_starts
                        and move.from_square not in developed_minor_starts):
                    developed_minor_starts.add(move.from_square)
                    minor_dev_phases.append(phase)

        board.push(move)
        ply += 1
        node = next_node

    if target_move_count == 0:
        return None

    return {
        "rating": rating,
        "eco": headers.get("ECO"),
        "total_player_moves": target_move_count,
        "capture_rate": captures / target_move_count,
        "check_rate": checks / target_move_count,
        "pawn_push_phase": pawn_push_phase,       # None if never occurred
        "castled": castled,
        "castle_phase": castle_phase,             # None if never castled
        "queen_dev_phase": queen_dev_phase,       # None if never moved
        "minor_dev_phase_avg": (sum(minor_dev_phases) / len(minor_dev_phases)
                                 if minor_dev_phases else None),
        "minor_dev_count": len(minor_dev_phases),  # out of 4 possible
    }


# ---------------------------------------------------------------------------
# Stage 3: aggregate one player's per-game stats into a single row
# ---------------------------------------------------------------------------

def _safe_mean(values):
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None


def aggregate_player_stats(username, per_game_stats):
    n_games = len(per_game_stats)
    ratings = [g["rating"] for g in per_game_stats if g["rating"] is not None]
    ecos = [g["eco"] for g in per_game_stats if g["eco"]]
    castled_games = [g for g in per_game_stats if g["castled"]]

    return {
        "username": username,
        "n_games": n_games,
        "avg_rating": _safe_mean(ratings),
        "avg_capture_rate": _safe_mean([g["capture_rate"] for g in per_game_stats]),
        "avg_check_rate": _safe_mean([g["check_rate"] for g in per_game_stats]),
        "avg_moves_per_game": _safe_mean([g["total_player_moves"] for g in per_game_stats]),
        # phase averages computed ONLY over games where the event happened —
        # "never occurred" is tracked separately as a rate, not folded into the average
        "avg_pawn_push_phase": _safe_mean([g["pawn_push_phase"] for g in per_game_stats]),
        "pct_games_never_pushed_contested_pawn": sum(
            1 for g in per_game_stats if g["pawn_push_phase"] is None) / n_games,
        "avg_castle_phase": _safe_mean([g["castle_phase"] for g in per_game_stats]),
        "pct_games_never_castled": (n_games - len(castled_games)) / n_games,
        "avg_queen_dev_phase": _safe_mean([g["queen_dev_phase"] for g in per_game_stats]),
        "pct_games_queen_never_moved": sum(
            1 for g in per_game_stats if g["queen_dev_phase"] is None) / n_games,
        "avg_minor_dev_phase": _safe_mean([g["minor_dev_phase_avg"] for g in per_game_stats]),
        "avg_minor_pieces_developed": _safe_mean([g["minor_dev_count"] for g in per_game_stats]),
        "opening_diversity": (len(set(ecos)) / len(ecos)) if ecos else None,
        "n_distinct_openings": len(set(ecos)),
    }


# ---------------------------------------------------------------------------
# Stage 4: SQLite checkpointing
# ---------------------------------------------------------------------------

SCHEMA = """
CREATE TABLE IF NOT EXISTS player_features (
    username TEXT PRIMARY KEY,
    n_games INTEGER,
    avg_rating REAL,
    avg_capture_rate REAL,
    avg_check_rate REAL,
    avg_moves_per_game REAL,
    avg_pawn_push_phase REAL,
    pct_games_never_pushed_contested_pawn REAL,
    avg_castle_phase REAL,
    pct_games_never_castled REAL,
    avg_queen_dev_phase REAL,
    pct_games_queen_never_moved REAL,
    avg_minor_dev_phase REAL,
    avg_minor_pieces_developed REAL,
    opening_diversity REAL,
    n_distinct_openings INTEGER
);
CREATE TABLE IF NOT EXISTS skipped_usernames (
    username TEXT PRIMARY KEY,
    reason TEXT
);
"""


def init_db(db_path):
    conn = sqlite3.connect(db_path)
    conn.executescript(SCHEMA)
    conn.commit()
    return conn


def already_processed(conn, username):
    cur = conn.execute(
        "SELECT 1 FROM player_features WHERE username = ? "
        "UNION SELECT 1 FROM skipped_usernames WHERE username = ?",
        (username, username),
    )
    return cur.fetchone() is not None


def save_player_row(conn, row):
    cols = ", ".join(row.keys())
    placeholders = ", ".join("?" for _ in row)
    conn.execute(
        f"INSERT OR REPLACE INTO player_features ({cols}) VALUES ({placeholders})",
        list(row.values()),
    )
    conn.commit()


def save_skipped(conn, username, reason):
    conn.execute(
        "INSERT OR REPLACE INTO skipped_usernames (username, reason) VALUES (?, ?)",
        (username, reason),
    )
    conn.commit()


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def build_player_dataset(usernames, db_path, max_games=50, min_games=20):
    conn = init_db(db_path)

    for i, username in enumerate(usernames, 1):
        username = username.strip()
        if not username:
            continue
        if already_processed(conn, username):
            print(f"[{i}/{len(usernames)}] {username}: already done, skipping")
            continue

        print(f"[{i}/{len(usernames)}] {username}: fetching...")
        games = list(fetch_player_games(username, max_games=max_games))

        per_game_stats = []
        for game in games:
            stats = compute_single_game_stats(game, username)
            if stats is not None:
                per_game_stats.append(stats)

        if len(per_game_stats) < min_games:
            reason = f"only {len(per_game_stats)} usable games (min {min_games})"
            print(f"  {username}: {reason}, skipping")
            save_skipped(conn, username, reason)
            continue

        row = aggregate_player_stats(username, per_game_stats)
        save_player_row(conn, row)
        print(f"  {username}: saved ({len(per_game_stats)} games used)")

    conn.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--usernames", required=True, help="Text file, one username per line")
    parser.add_argument("--db", required=True, help="SQLite output path")
    parser.add_argument("--max-games", type=int, default=50)
    parser.add_argument("--min-games", type=int, default=20)
    args = parser.parse_args()

    with open(args.usernames) as f:
        usernames = [line.strip() for line in f if line.strip()]

    build_player_dataset(usernames, args.db, max_games=args.max_games, min_games=args.min_games)
    print("Done.")


if __name__ == "__main__":
    main()
