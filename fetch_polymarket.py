"""Build a replay CSV using Polymarket title prices and the Kalshi bracket.

Polymarket's 2026 first-round game markets are thin: several have no trade
history at the cutoff. The title market is liquid, so this script swaps only
the title prices. Game prices and priors stay from the Kalshi replay, which
isolates the title-market comparison.
"""
import csv
import json
import urllib.request
from pathlib import Path

CUTOFF = 1773921600  # 2026-03-19T12:00:00Z
TITLE_SLUG = "2026-ncaa-tournament-winner"
HERE = Path(__file__).resolve().parent
SRC = HERE / "replays/m_ncaa_2026.csv"
OUT = HERE / "cross/p_ncaa_2026.csv"
UA = "Mozilla/5.0 (X11; Linux x86_64) bracket-edge/1.0"

TITLE = {
    "California Baptist": "Cal Baptist",
    "Hawai'i": "Hawaii",
    "Iowa St.": "Iowa State",
    "Kennesaw St.": "Kennesaw State",
    "Michigan St.": "Michigan State",
    "North Dakota St.": "North Dakota State",
    "Northern Iowa": "Norhtern Iowa",
    "Ohio St.": "Ohio State",
    "Queens University": "Queens",
    "Saint Mary's": "St. Mary's",
    "St. John's": "St John's",
    "Tennessee St.": "Tennessee State",
    "UConn": "Connecticut",
    "Utah St.": "Utah State",
    "Wright St.": "Wright State",
}


def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def price(token):
    url = (
        "https://clob.polymarket.com/prices-history?"
        f"market={token}&startTs={CUTOFF - 14 * 86400}&endTs={CUTOFF}&fidelity=60"
    )
    hist = get(url).get("history", [])
    return float(hist[-1]["p"]) if hist else None


def main():
    rows = list(csv.DictReader(SRC.open()))
    event = next(e for e in get(
        "https://gamma-api.polymarket.com/events?slug=" + TITLE_SLUG) if e.get("slug") == TITLE_SLUG)
    markets = {m["groupItemTitle"]: m for m in event["markets"] if m.get("groupItemTitle")}

    out, title_sum = [], 0.0
    for r in rows:
        r = dict(r)
        r["tournament"] = "p_ncaa_2026"
        if r["type"] == "title":
            name = TITLE.get(r["a"], r["a"])
            m = markets.get(name)
            if m is None:
                raise SystemExit(f"no Polymarket title market for {r['a']} ({name})")
            p = price(json.loads(m["clobTokenIds"])[0])
            if p is None:
                raise SystemExit(f"no Polymarket price for {r['a']} ({name})")
            r["pa"] = f"{p:.6f}"
            title_sum += p
        out.append(r)

    OUT.parent.mkdir(exist_ok=True)
    with OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["tournament", "type", "a", "b", "pa", "pb"])
        w.writeheader()
        w.writerows(out)
    print(f"wrote {OUT} ({len(out)} rows); title sum {title_sum:.3f}")


if __name__ == "__main__":
    main()
