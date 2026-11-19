"""Build a real pre-tournament NCAA replay from Kalshi historical prices.

Example:
    python3 replay.py --season 2026
    python3 bracket_edge.py --csv replay.csv
    python3 backtest.py --csv replay.csv

The cutoff is after the First Four and before the Round of 64, so no
tournament result leaks into the prices. The prior is Elo from regular-season
results only, also cut off before the tournament.
"""
import argparse, csv, datetime, json, time, urllib.error, urllib.request
from pathlib import Path

BASE = "https://api.elections.kalshi.com/trade-api/v2"
HERE = Path(__file__).resolve().parent


def dt(s):
    return datetime.datetime.fromisoformat(s.replace("Z", "+00:00"))


def get(url, tries=5):
    for i in range(tries):
        try:
            return json.load(urllib.request.urlopen(url, timeout=60))
        except urllib.error.HTTPError as e:
            if i == tries - 1:
                raise
            time.sleep(5 * (i + 1) if e.code == 429 else 1)
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(1)


def fetch_series(series, cache, refresh=False):
    path = cache / f"{series}.json"
    if path.exists() and not refresh:
        return json.load(open(path))
    out, cursor = [], ""
    while True:
        url = f"{BASE}/historical/markets?series_ticker={series}&limit=200"
        if cursor:
            url += f"&cursor={cursor}"
        d = get(url)
        out += d.get("markets", [])
        cursor = d.get("cursor", "")
        if not cursor:
            break
    json.dump(out, open(path, "w"))
    return out


def fetch_event(event, cache, refresh=False):
    path = cache / f"{event}.json"
    if path.exists() and not refresh:
        return json.load(open(path))
    out, cursor = [], ""
    while True:
        url = f"{BASE}/historical/markets?event_ticker={event}&limit=200"
        if cursor:
            url += f"&cursor={cursor}"
        d = get(url)
        out += d.get("markets", [])
        cursor = d.get("cursor", "")
        if not cursor:
            break
    json.dump(out, open(path, "w"))
    return out


def games_from(markets):
    """Event ticker -> {teams, winner, close, tickers} for two-team games."""
    ev = {}
    for m in markets:
        ev.setdefault(m["event_ticker"], []).append(m)
    out = {}
    for e, arr in ev.items():
        if len(arr) != 2:
            continue
        w = [m for m in arr if m["result"] == "yes"]
        l = [m for m in arr if m["result"] == "no"]
        if len(w) != 1 or len(l) != 1:
            continue
        out[e] = {
            "teams": [m["yes_sub_title"] for m in arr],
            "winner": w[0]["yes_sub_title"],
            "close": dt(arr[0]["close_time"]),
            "tickers": {m["yes_sub_title"]: m["ticker"] for m in arr},
        }
    return out


def build_tree(games, finalists, champion, start):
    by = {}
    for g in games.values():
        if g["close"] < start:
            continue
        for t in g["teams"]:
            by.setdefault(t, []).append(g)

    def team_node(team, before):
        won = [g for g in by.get(team, []) if g["close"] < before and g["winner"] == team]
        if not won:
            return {"team": team, "children": []}
        return game_node(max(won, key=lambda x: x["close"]))

    def game_node(g):
        return {"game": g, "children": [team_node(t, g["close"]) for t in g["teams"]]}

    root_close = max(g["close"] for g in games.values()) + datetime.timedelta(hours=1)
    root = {"game": {"teams": finalists, "winner": champion, "close": root_close},
            "children": [team_node(t, root_close) for t in finalists]}
    first = []

    def walk(n):
        if "game" in n and all("team" in c for c in n["children"]):
            first.append(n["game"])
        for c in n["children"]:
            walk(c)

    walk(root)
    return root, first


def elo(games, cutoff):
    seq = sorted((g for g in games.values() if g["close"] < cutoff),
                 key=lambda g: g["close"])
    r = {}
    for g in seq:
        w, l = g["winner"], [t for t in g["teams"] if t != g["winner"]][0]
        rw, rl = r.get(w, 1500.0), r.get(l, 1500.0)
        ew = 1 / (1 + 10 ** ((rl - rw) / 400))
        r[w], r[l] = rw + 20 * (1 - ew), rl - 20 * ew
    return r


def candle(ticker, cutoff, cache, refresh=False):
    key = cache / "candles" / cutoff.strftime("%Y%m%dT%H%M%SZ") / f"{ticker}.json"
    if key.exists() and not refresh:
        return json.load(open(key))
    key.parent.mkdir(parents=True, exist_ok=True)
    lo = int((cutoff - datetime.timedelta(days=5)).timestamp())
    hi = int((cutoff + datetime.timedelta(hours=1)).timestamp())
    d = get(f"{BASE}/historical/markets/{ticker}/candlesticks"
            f"?start_ts={lo}&end_ts={hi}&period_interval=60")
    cs = [c for c in d.get("candlesticks", []) if c["end_period_ts"] <= cutoff.timestamp()]
    val = None
    if cs:
        c = cs[-1]
        ask = c["yes_ask"].get("close")
        bid = c["yes_bid"].get("close")
        if ask is not None and bid is not None:
            af, bf = float(ask), float(bid)
            if 0 <= af <= 1 and 0 <= bf <= 1 and np.isfinite(af) and np.isfinite(bf):
                val = {"ask": af, "bid": bf, "ts": c["end_period_ts"]}
    json.dump(val, open(key, "w"))
    time.sleep(0.3)
    return val


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--season", default="2026")
    p.add_argument("--games-series", default="KXNCAAMBGAME")
    p.add_argument("--label", default=None, help="tournament name in the CSV")
    p.add_argument("--cutoff", default="2026-03-19T12:00:00+00:00",
                   help="after First Four, before Round of 64")
    p.add_argument("--prior-cutoff", default="2026-03-16T12:00:00+00:00",
                   help="regular-season results only")
    p.add_argument("--title-event", default=None)
    p.add_argument("--max-title-spread", type=float, default=0.10,
                   help="reject a tournament when title quotes are this wide (stale 50/50)")
    p.add_argument("--max-game-spread", type=float, default=0.20,
                   help="reject a game quote when ask-bid exceeds this (stale)")
    p.add_argument("--out", default="replay.csv")
    p.add_argument("--cache", default=str(HERE / "cache"))
    p.add_argument("--refresh", action="store_true")
    a = p.parse_args(argv)

    cache = Path(a.cache)
    cache.mkdir(parents=True, exist_ok=True)
    cutoff, pcut = dt(a.cutoff), dt(a.prior_cutoff)
    title_event = a.title_event or f"KXMARMAD-{a.season[-2:]}"
    label = a.label or f"ncaa{a.season}"

    games = games_from(fetch_series(a.games_series, cache, a.refresh))
    tms = fetch_event(title_event, cache, a.refresh)
    title_ticker = {m["yes_sub_title"]: m["ticker"] for m in tms}
    winner = [m["yes_sub_title"] for m in tms if m["result"] == "yes"]
    if len(winner) != 1:
        raise SystemExit(f"{title_event}: expected one winning title market")
    champion = winner[0]

    # finalists are the two title markets that closed last
    finalists = [m["yes_sub_title"] for m in
                 sorted(tms, key=lambda m: m["close_time"])[-2:]]
    if champion not in finalists:
        raise SystemExit(f"{title_event}: champion not in finalists {finalists}")

    root, first = build_tree(games, finalists, champion, cutoff)
    teams = [t for g in first for t in g["teams"]]
    if len(teams) != len(set(teams)) or len(first) * 2 != len(teams):
        raise SystemExit(f"bad bracket: {len(first)} games, {len(teams)} teams")

    elo_r = elo(games, pcut)
    prior = {t: (elo_r.get(t, 1500.0) - 1500.0) / 173.7 for t in teams}
    missing = [t for t in teams if t not in title_ticker]
    if missing:
        raise SystemExit(f"no title market for {', '.join(missing)}")

    rows = [(label, "prior", t, "", prior[t], "") for t in teams]
    for g in first:
        a_, b_ = g["teams"]
        ca = candle(g["tickers"][a_], cutoff, cache, a.refresh)
        cb = candle(g["tickers"][b_], cutoff, cache, a.refresh)
        if not ca or not cb:
            raise SystemExit(f"no candle for {g['event']}")
        if ca["ask"] - ca["bid"] > a.max_game_spread:
            raise SystemExit(f"{a_}: stale game quote bid {ca['bid']:.2f} "
                             f"ask {ca['ask']:.2f} (spread > {a.max_game_spread:.2f})")
        if cb["ask"] - cb["bid"] > a.max_game_spread:
            raise SystemExit(f"{b_}: stale game quote bid {cb['bid']:.2f} "
                             f"ask {cb['ask']:.2f} (spread > {a.max_game_spread:.2f})")
        rows.append((label, "game", a_, b_,
                     (ca["ask"] + ca["bid"]) / 2, (cb["ask"] + cb["bid"]) / 2))
    for t in teams:
        c = candle(title_ticker[t], cutoff, cache, a.refresh)
        if not c:
            raise SystemExit(f"no title candle for {t}")
        if c["ask"] - c["bid"] > a.max_title_spread:
            raise SystemExit(f"{t}: stale title quote bid {c['bid']:.2f} "
                             f"ask {c['ask']:.2f} (spread > {a.max_title_spread:.2f})")
        rows.append((label, "title", t, "", (c["ask"] + c["bid"]) / 2, ""))
    rows.append((label, "winner", champion, "", "", ""))

    with open(a.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tournament", "type", "a", "b", "pa", "pb"])
        for r in rows:
            w.writerow(r)
    print(f"{a.out}: {len(first)} first-round games, {len(teams)} teams, champion {champion}")


if __name__ == "__main__":
    main()
