#!/usr/bin/env python3
"""Pull bracket prices from Kalshi into the CSV that bracket_edge.py reads.

Public market data needs no key. A game event holds one market per team; a
title event holds one market per team to win the tournament. Prices are the
YES ask (what you pay to buy), as the note asks.

    python3 fetch_kalshi.py series KXNBAGAME
    python3 fetch_kalshi.py snapshot --tournament ncaa \
        --title-event KXNCAA-26 --game KXNCAA-26R1A --game KXNCAA-26R1B
    python3 bracket_edge.py --csv snapshots.csv

List --game in bracket order: the first n/2 rows are the first round, and each
game is one market per team. Title rows are written for the teams in those
games, so a futures event with extra teams still works.
"""
import argparse
import csv
import json
import os
import sys
import urllib.parse
import urllib.request

BASE = os.environ.get("KALSHI_API_BASE", "https://api.elections.kalshi.com/trade-api/v2")
UA = "bracket-edge/1.0"


def get(path, **params):
    url = BASE + path
    if params:
        url += "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.load(r)


def markets(event):
    """Every market in one event, following the cursor."""
    out, cursor = [], ""
    while True:
        d = get("/markets", event_ticker=event, limit=200, **({"cursor": cursor} if cursor else {}))
        out += d.get("markets", [])
        cursor = d.get("cursor") or ""
        if not cursor:
            return out


def team(m):
    name = (m.get("yes_sub_title") or "").strip()
    if not name:
        raise ValueError(f"{m.get('ticker')}: no yes_sub_title to name the team")
    return name


def ask(m):
    """YES ask in dollars, the price paid to buy. Raises if nobody offers."""
    v = m.get("yes_ask_dollars")
    if not v or float(v) <= 0:
        raise ValueError(f"{m.get('ticker')}: no YES ask, so buying YES is not possible")
    return float(v)


def build(tournament, title_markets, game_markets, result=None):
    """CSV rows from raw Kalshi market lists. Raises if prices are missing."""
    title = {}
    for m in title_markets:
        name = team(m)
        if name in title:
            raise ValueError(f"two title markets both named {name!r}; rename one")
        title[name] = ask(m)

    games = []
    for ms in game_markets:
        if len(ms) != 2:
            raise ValueError(f"game event has {len(ms)} markets, expected 2: "
                             + ", ".join(m.get("ticker", "?") for m in ms))
        a, b = ms
        games.append((team(a), team(b), ask(a), ask(b)))

    order = []
    for a, b, _, _ in games:
        for x in (a, b):
            if x not in order:
                order.append(x)
    missing = [x for x in order if x not in title]
    if missing:
        raise ValueError("title event has no market for: " + ", ".join(missing))

    rows = [(tournament, "game", a, b, pa, pb) for a, b, pa, pb in games]
    rows += [(tournament, "title", x, "", title[x], "") for x in order]
    if result:
        if result not in title:
            raise ValueError(f"result {result!r} is not in the title event")
        rows.append((tournament, "winner", result, "", "", ""))
    return rows


def write(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tournament", "type", "a", "b", "pa", "pb"])
        w.writerows(rows)


def cmd_series(a):
    cursor = ""
    while True:
        d = get("/events", series_ticker=a.series, status=a.status, limit=200,
                **({"cursor": cursor} if cursor else {}))
        for e in d.get("events", []):
            print(f"{e['event_ticker']:<28} {e.get('sub_title', ''):<24} {e.get('title', '')}")
        cursor = d.get("cursor") or ""
        if not cursor:
            return


def cmd_snapshot(a):
    title = markets(a.title_event)
    games = [markets(e) for e in a.game]
    rows = build(a.tournament, title, games, a.result)
    if a.out == "-":
        w = csv.writer(sys.stdout)
        w.writerow(["tournament", "type", "a", "b", "pa", "pb"])
        w.writerows(rows)
    else:
        write(a.out, rows)
        print(f"{len(rows)} rows -> {a.out}", file=sys.stderr)


def cmd_markets(a):
    """One event's markets as title rows, for viewing rather than modelling."""
    ms = [m for m in markets(a.event) if m.get("status") == "active"]
    if not ms:
        raise SystemExit(f"{a.event}: no active markets")
    rows, seen = [], set()
    for m in ms:
        name = team(m)
        if name in seen:
            raise SystemExit(f"{a.event}: two markets both named {name!r}")
        seen.add(name)
        try:
            price = ask(m)
        except ValueError as e:
            print(f"skip {e}", file=sys.stderr)
            continue
        rows.append((a.tournament or a.event, "title", name, "", price, ""))
    if a.out == "-":
        w = csv.writer(sys.stdout)
        w.writerow(["tournament", "type", "a", "b", "pa", "pb"])
        w.writerows(rows)
    else:
        write(a.out, rows)
        print(f"{len(rows)} markets -> {a.out}", file=sys.stderr)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("series", help="list open events in a series")
    s.add_argument("series")
    s.add_argument("--status", default="open")
    s.set_defaults(func=cmd_series)

    n = sub.add_parser("snapshot", help="write one tournament as a price CSV")
    n.add_argument("--tournament", default="kalshi")
    n.add_argument("--title-event", required=True, help="event with one market per team")
    n.add_argument("--game", action="append", required=True,
                   help="game event, repeat in bracket order")
    n.add_argument("--result", help="team that won, for backtesting")
    n.add_argument("--out", default="snapshots.csv")
    n.set_defaults(func=cmd_snapshot)

    k = sub.add_parser("markets", help="write one event's markets as title rows")
    k.add_argument("--event", required=True)
    k.add_argument("--tournament", help="section name, defaults to the event ticker")
    k.add_argument("--out", default="markets.csv")
    k.set_defaults(func=cmd_markets)

    a = p.parse_args(argv)
    a.func(a)


if __name__ == "__main__":
    main()
