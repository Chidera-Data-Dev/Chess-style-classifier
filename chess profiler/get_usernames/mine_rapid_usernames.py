"""
Rapid-player username mining from a monthly archive
======================================================
Streams a Lichess monthly .pgn.zst archive directly from its URL,
filters to rated rapid games, and collects distinct usernames as it goes
— stopping once the target count is reached. Avoids the daily-snapshot
limit of the tournament-results approach: a single month's archive
naturally contains hundreds of thousands of distinct rapid players
across every rating band.

Install dependencies first:
    pip install python-chess requests zstandard

Usage:
    python mine_rapid_usernames.py \
        --url https://database.lichess.org/standard/lichess_db_standard_rated_2023-06.pgn.zst \
        --out usernames.txt \
        --target 10000
"""

import argparse
import io
import sys

import chess.pgn
import requests
import zstandard as zstd

RAPID_MIN_SECONDS = 600    # 10 min
RAPID_MAX_SECONDS = 1799   # just under 30 min


def is_rapid_time_control(time_control):
    """TimeControl header looks like '600+5' (base+increment) or '-'
    for correspondence/unlimited. Returns True if base seconds fall in
    the rapid range."""
    if not time_control or time_control == "-":
        return False
    base = time_control.split("+")[0]
    try:
        base_seconds = int(base)
    except ValueError:
        return False
    return RAPID_MIN_SECONDS <= base_seconds <= RAPID_MAX_SECONDS


def stream_games(url):
    """Streams and decompresses the archive on the fly — same pattern as
    lichess_filter_extract.py. Nothing large ever touches disk."""
    resp = requests.get(url, stream=True)
    resp.raise_for_status()
    dctx = zstd.ZstdDecompressor()
    with dctx.stream_reader(resp.raw) as reader:
        text_stream = io.TextIOWrapper(reader, encoding="utf-8", errors="replace")
        while True:
            game = chess.pgn.read_game(text_stream)
            if game is None:
                break
            yield game


def mine_usernames(url, target=10000):
    seen = set()
    scanned = 0

    for game in stream_games(url):
        scanned += 1
        headers = game.headers

        if "Rated" not in headers.get("Event", ""):
            continue
        if not is_rapid_time_control(headers.get("TimeControl", "")):
            continue
        if headers.get("WhiteTitle") == "BOT" or headers.get("BlackTitle") == "BOT":
            continue

        white = headers.get("White")
        black = headers.get("Black")
        if white:
            seen.add(white)
        if black:
            seen.add(black)

        if scanned % 20000 == 0:
            print(f"scanned={scanned} unique_usernames={len(seen)}", file=sys.stderr)

        if len(seen) >= target:
            break

    return seen, scanned


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True, help="URL of a lichess_db_standard_rated_*.pgn.zst file")
    parser.add_argument("--out", required=True, help="Output usernames.txt path")
    parser.add_argument("--target", type=int, default=10000, help="Stop once this many unique usernames are collected")
    args = parser.parse_args()

    usernames, scanned = mine_usernames(args.url, target=args.target)

    with open(args.out, "w") as f:
        for name in sorted(usernames):
            f.write(name + "\n")

    print(f"Done. Scanned {scanned} games, collected {len(usernames)} unique rapid usernames -> {args.out}")


if __name__ == "__main__":
    main()
