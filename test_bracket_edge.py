#!/usr/bin/env python3
"""Self-checks for bracket_edge.py, backtest.py, fetch_kalshi.py and visual.py."""
import os
import tempfile
import types

import numpy as np

import backtest
import fetch_kalshi
import visual
from backtest import profit
from bracket_edge import (
    Tournament, analyse, check, devig, load_csv, rows, sample, simulate,
)

CSV = """tournament,type,a,b,pa,pb
t1,game,A,B,0.70,0.32
t1,game,C,D,0.55,0.47
t1,title,A,,0.34,
t1,title,B,,0.22,
t1,title,C,,0.26,
t1,title,D,,0.18,
t1,prior,A,,1.0,
t1,winner,A,,,
"""


def test_devig():
    a, b = devig(0.70, 0.32)
    assert abs(a + b - 1.0) < 1e-12
    assert abs(devig(0.55, 0.47)[0] - 0.55 / 1.02) < 1e-12


def test_csv_roundtrip():
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as f:
        f.write(CSV)
        path = f.name
    try:
        t, = load_csv(path)
        n, games, title, prior = check(t)
        assert n == 4
        assert games[0][:2] == (0, 1)
        assert abs(title[0] - 0.34) < 1e-12
        assert prior[0] == 1.0 and prior[3] == 0.0
        assert t.winner == 0
    finally:
        os.unlink(path)


def test_bad_bracket():
    t = sample()
    t.games = t.games[:1]
    try:
        check(t)
    except ValueError as e:
        assert "first-round games" in str(e)
    else:
        raise AssertionError("incomplete first round was accepted")


def test_simulate_two_teams():
    out = simulate([(0, 1)], np.array([[0.7, 0.3]]), np.zeros((1, 2)), 2, 1)
    assert np.allclose(out, [[0.7, 0.3]])


def test_sample_output():
    t = sample()
    res = analyse(t, draws=300, rng=np.random.default_rng(0))
    assert abs(res["mean"].sum() - 1.0) < 1e-9
    assert np.all((res["mean"] >= 0) & (res["mean"] <= 1))
    assert np.all(res["sd"] >= 0)
    assert np.all(res["lo"] <= res["mean"]) and np.all(res["mean"] <= res["hi"])


def test_row_formulas():
    t = sample()
    res = analyse(t, draws=300, rng=np.random.default_rng(1))
    fee = 0.02
    for r in rows(t, res, fee=fee, min_edge=0.03, min_sure=0.9):
        yes = r["model"] - r["market"] - fee
        no = r["market"] - r["model"] - fee
        assert abs(r["edge"] - max(yes, no)) < 1e-12
        assert r["side"] == ("YES" if yes >= no else "NO")
        assert 0.0 <= r["sure"] <= 1.0
        want = r["side"] if r["edge"] >= 0.03 and r["sure"] >= 0.9 else "-"
        assert r["action"] == want


def test_profit_sides():
    t = sample()
    t.winner = 0
    t.title[0] = 0.34
    assert abs(profit(t, 0, "YES", 0.02) - (1 - 0.34 - 0.02)) < 1e-12
    assert abs(profit(t, 1, "NO", 0.02) - (0.22 - 0 - 0.02)) < 1e-12


def market(ticker, name, ask):
    return {"ticker": ticker, "yes_sub_title": name,
            "yes_ask_dollars": f"{ask:.2f}", "yes_bid_dollars": f"{ask - 0.02:.2f}"}


def test_fetch_build():
    title = [market("T-A", "A", 0.30), market("T-B", "B", 0.20),
             market("T-C", "C", 0.25), market("T-D", "D", 0.15),
             market("T-E", "E", 0.10)]
    games = [
        [market("G-AB-A", "A", 0.60), market("G-AB-B", "B", 0.42)],
        [market("G-CD-C", "C", 0.55), market("G-CD-D", "D", 0.47)],
    ]
    out = fetch_kalshi.build("t", title, games, "A")
    assert out[0] == ("t", "game", "A", "B", 0.60, 0.42)
    assert ("t", "title", "D", "", 0.15, "") in out
    assert out[-1] == ("t", "winner", "A", "", "", "")
    assert not any(r[2] == "E" for r in out)          # unused futures team dropped
    assert fetch_kalshi.ask(market("x", "X", 0.4)) == 0.4


def test_fetch_rejects_missing_price():
    games = [[market("G-A", "A", 0.6), market("G-B", "B", 0.4)]]
    try:
        fetch_kalshi.build("t", [market("T-A", "A", 0.3)], games)
    except ValueError as e:
        assert "no market for" in str(e)
    else:
        raise AssertionError("missing title market was accepted")


def test_fetch_rejects_zero_ask():
    try:
        fetch_kalshi.ask(market("G-A", "A", 0.0))
    except ValueError as e:
        assert "no YES ask" in str(e)
    else:
        raise AssertionError("zero ask was accepted")


def test_fetch_rejects_duplicate_title():
    title = [market("T-A1", "A", 0.3), market("T-A2", "A", 0.2)]
    try:
        fetch_kalshi.build("t", title, [])
    except ValueError as e:
        assert "both named" in str(e)
    else:
        raise AssertionError("duplicate title names were accepted")


def test_backtest_needs_winner():
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as f:
        f.write(CSV.replace("t1,winner,A,,,\n", ""))
        path = f.name
    try:
        try:
            backtest.main(["--csv", path, "--draws", "20"])
        except SystemExit as e:
            assert "winner" in str(e)
        else:
            raise AssertionError("backtest without a winner was accepted")
    finally:
        os.unlink(path)


def test_visual_model():
    t = sample()
    args = types.SimpleNamespace(draws=100, prior_sd=0.6, model_game_noise=0.25,
                                 fee=0.02, min_edge=0.03, min_sure=0.9)
    mr, note = visual.model_rows(t, args, np.random.default_rng(0))
    assert note == "" and len(mr) == 4
    page = visual.section(t, mr, True, "", 100.0)
    assert "YES" in page and "+13.0c" in page and "model mean" in page
    flagged = [dict(r) for r in mr]
    flagged[0]["action"] = "YES"
    page = visual.section(t, flagged, True, "", 100.0)
    assert "data-edge=" in page and 'class="money"' in page
    svg = visual.scatter([{**r, "tournament": "sample"} for r in mr])
    assert "<svg" in svg and "sample / A:" in svg


def test_visual_market_only():
    t = Tournament("poll")
    for name, price in (("A", 0.1), ("B", 0.4), ("C", 0.2)):
        t.title[t.index(name)] = price
    mr = visual.market_rows(t)
    assert [r["team"] for r in mr] == ["B", "C", "A"]
    page = visual.section(t, mr, False, "prices only", 100.0)
    assert "YES ask" in page and "model mean" not in page


def test_visual_money():
    yes = {"action": "YES", "edge": 0.10, "market": 0.50, "side": "YES"}
    no = {"action": "NO", "edge": 0.10, "market": 0.80, "side": "NO"}
    assert round(visual.expected_profit([yes], 100), 2) == 20.0
    assert round(visual.expected_profit([no], 100), 2) == 50.0
    assert visual.expected_profit([], 100) == 0
    assert visual.usd(12.7) == "$12.70"


if __name__ == "__main__":
    tests = [(k, v) for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    for name, fn in tests:
        fn()
        print(f"ok  {name}")
    print(f"{len(tests)} checks passed")
