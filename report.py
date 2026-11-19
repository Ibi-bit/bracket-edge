#!/usr/bin/env python3
"""Build one self-contained HTML report from the real replay CSVs.

    python3 report.py
    python3 report.py --min-edge 0.03 --min-sure 0.60 --stake 100

Writes report.html plus a CSV of every team's model and market probability.
No external assets, so the HTML can be emailed or opened offline.
"""
import argparse
import base64
import csv
import html
import io
from pathlib import Path

import numpy as np

from backtest import profit
from bracket_edge import (
    FEE, GAME_NOISE, PRIOR_SD,
    analyse, check, load_csv, rows,
)
from sweep import mean_se, score

FILTERS = [
    (0.03, 0.90),
    (0.03, 0.85),
    (0.03, 0.60),
    (0.01, 0.60),
    (0.01, 0.00),
]

CSS = """
:root{
  --bg:#070b14;--panel:#0f172a;--panel2:#111c33;--line:#1e2b45;
  --text:#e5edf9;--muted:#8fa1bd;--teal:#5eead4;--amber:#fbbf24;
  --green:#34d399;--red:#f87171;--track:#1b2940;
}
*{box-sizing:border-box}
body{margin:0;background:radial-gradient(1200px 700px at 15% -10%,#13213d 0%,var(--bg) 55%);
  color:var(--text);font:14px/1.55 ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
  -webkit-font-smoothing:antialiased}
.wrap{max-width:1080px;margin:0 auto;padding:38px 20px 64px}
.hero h1{font-size:clamp(27px,4.2vw,42px);line-height:1.08;margin:6px 0 12px;letter-spacing:-.025em}
.eyebrow{font-size:11px;letter-spacing:.18em;text-transform:uppercase;color:var(--teal);font-weight:700}
.sub{max-width:820px;color:var(--muted);margin:0 0 22px;font-size:15px}
.alert{display:flex;gap:12px;align-items:flex-start;border:1px solid rgba(251,191,36,.38);
  background:rgba(251,191,36,.08);border-radius:14px;padding:14px 16px;margin:0 0 22px;color:#fde68a}
.alert b{color:var(--amber)}
.alert .icon{font-size:18px;line-height:1.2}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin:0 0 22px}
.card{background:linear-gradient(180deg,var(--panel2),var(--panel));border:1px solid var(--line);
  border-radius:14px;padding:14px 16px}
.card .k{font-size:10px;letter-spacing:.11em;text-transform:uppercase;color:var(--muted);font-weight:700}
.card .v{font-size:25px;font-weight:800;margin-top:3px;font-variant-numeric:tabular-nums;letter-spacing:-.02em}
.card .s{font-size:10px;color:var(--muted);margin-top:2px;font-weight:500}
.panel{background:linear-gradient(180deg,rgba(17,28,51,.92),rgba(15,23,42,.92));
  border:1px solid var(--line);border-radius:18px;padding:20px;margin:0 0 18px;
  box-shadow:0 20px 50px rgba(0,0,0,.25)}
.panel h2{font-size:18px;margin:0 0 4px}
.panel h3{font-size:13px;margin:18px 0 8px;color:var(--text)}
.note{color:var(--muted);font-size:12px;margin:0 0 14px}
.note b{color:var(--text);font-weight:600}
.chart{margin-top:12px}
.chart-head{display:flex;justify-content:space-between;color:var(--muted);font-size:10px;
  letter-spacing:.06em;text-transform:uppercase;margin:0 0 4px 180px}
.brow{display:grid;grid-template-columns:180px minmax(220px,1fr) 130px;gap:14px;align-items:center;
  padding:5px 0}
.blabel{font-size:12px;font-weight:600;line-height:1.35}
.blabel small{display:block;color:var(--muted);font-size:10px;font-weight:500}
.btrack{position:relative;height:20px;background:rgba(27,41,64,.55);border-radius:5px}
.bzero{position:absolute;top:0;bottom:0;width:1px;background:rgba(229,237,249,.35)}
.bbar{position:absolute;top:6px;height:8px;border-radius:4px;min-width:2px}
.bbar.pos{background:linear-gradient(90deg,rgba(52,211,153,.55),var(--green))}
.bbar.neg{background:linear-gradient(90deg,var(--red),rgba(248,113,113,.55))}
.bbar.amber{background:linear-gradient(90deg,rgba(251,191,36,.55),var(--amber))}
.bwhisk{position:absolute;top:9px;height:2px;background:rgba(229,237,249,.65)}
.bcap{position:absolute;top:5px;height:10px;width:1px;background:rgba(229,237,249,.65)}
.bval{text-align:right;font-variant-numeric:tabular-nums;font-weight:700;font-size:12px}
.bval small{display:block;color:var(--muted);font-weight:500;font-size:10px}
.legend{display:flex;flex-wrap:wrap;gap:16px;margin:12px 0 0;color:var(--muted);font-size:11px}
.legend span{display:flex;align-items:center;gap:6px}
.sw{width:18px;height:7px;border-radius:4px}
.sw.green{background:var(--green)}.sw.red{background:var(--red)}.sw.amber{background:var(--amber)}
.sw.whisk{height:2px;background:rgba(229,237,249,.65)}
.table-wrap{overflow-x:auto;margin-top:10px;border:1px solid var(--line);border-radius:12px}
table{width:100%;border-collapse:collapse;font-size:12px;min-width:720px}
th{position:sticky;top:0;background:#111c33;color:var(--muted);text-align:left;font-weight:700;
  font-size:10px;letter-spacing:.08em;text-transform:uppercase;padding:10px 10px;border-bottom:1px solid var(--line)}
td{padding:9px 10px;border-bottom:1px solid rgba(30,43,69,.55);white-space:nowrap}
tr:last-child td{border-bottom:0}
tr.won td:first-child{box-shadow:inset 3px 0 0 var(--green)}
tr.lost td:first-child{box-shadow:inset 3px 0 0 var(--red)}
td.num,th.num{text-align:right;font-variant-numeric:tabular-nums}
.pos{color:var(--green)}.neg{color:var(--red)}.mut{color:var(--muted)}
.pill{display:inline-block;font-size:10px;font-weight:800;letter-spacing:.05em;padding:2px 7px;
  border-radius:999px;border:1px solid transparent}
.pill.yes{color:var(--green);background:rgba(52,211,153,.12);border-color:rgba(52,211,153,.35)}
.pill.no{color:var(--red);background:rgba(248,113,113,.12);border-color:rgba(248,113,113,.35)}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:18px}
.box{background:rgba(7,11,20,.35);border:1px solid var(--line);border-radius:12px;padding:14px 16px;
  overflow:hidden}
.grid2 .brow{grid-template-columns:96px minmax(0,1fr) 82px;gap:8px}
.grid2 .chart-head{margin-left:104px}
.grid2 .blabel,.grid2 .bval{font-size:11px}
.box h3{margin:0 0 6px;font-size:13px}
.box p,.box li{color:var(--muted);font-size:12px;margin:0 0 8px}
.box ul{margin:0;padding-left:18px}
.box li{margin-bottom:5px}
.btn{display:inline-block;text-decoration:none;color:#06211c;background:var(--teal);
  font-weight:800;font-size:12px;padding:9px 13px;border-radius:9px;margin:4px 8px 4px 0}
.btn.ghost{color:var(--text);background:transparent;border:1px solid var(--line)}
details{margin-top:10px}
summary{cursor:pointer;color:var(--teal);font-weight:700;font-size:12px}
.scroll{max-height:420px;overflow:auto;border:1px solid var(--line);border-radius:12px;margin-top:10px}
footer{color:var(--muted);font-size:11px;margin-top:28px;line-height:1.7}
code{background:rgba(255,255,255,.05);border:1px solid var(--line);padding:1px 5px;border-radius:5px;
  font-size:11px;color:#cbd5e1}
@media (max-width:720px){
  .wrap{padding:26px 14px 48px}
  .brow{grid-template-columns:120px minmax(120px,1fr) 92px;gap:8px}
  .blabel{font-size:11px}.bval{font-size:11px}
  .grid2{grid-template-columns:1fr}
  .chart-head{margin-left:128px}
}
"""


NAMES = {"m_crown_2026": "2026 Crown", "m_ncaa_2026": "2026 NCAA men"}


def pretty(name):
    return NAMES.get(name, name)


def esc(x):
    return html.escape(str(x))


def pct(x):
    return f"{x * 100:.1f}%"


def cents(x):
    return f"{x * 100:+.1f}c"


def usd(x):
    return f"${x:,.2f}"


def signed_usd(x):
    return f"{'+' if x >= 0 else '-'}${abs(x):,.2f}"


def pos(v, lo, hi):
    return (v - lo) / (hi - lo) * 100.0


def bar(label, sub, value, lo, hi, colour, value_text):
    z, x = pos(0.0, lo, hi), pos(value, lo, hi)
    left, width = min(z, x), abs(x - z)
    return (f'<div class="brow"><div class="blabel">{label}<small>{sub}</small></div>'
            f'<div class="btrack"><div class="bzero" style="left:{z:.2f}%"></div>'
            f'<div class="bbar {colour}" style="left:{left:.2f}%;width:{width:.2f}%"></div></div>'
            f'<div class="bval">{value_text}</div></div>')


def whisker(label, sub, d, lo, hi):
    if d["n"] == 0:
        return (f'<div class="brow"><div class="blabel">{label}<small>{sub}</small></div>'
                f'<div class="btrack"><div class="bzero" style="left:{pos(0, lo, hi):.2f}%"></div></div>'
                f'<div class="bval mut">no trades</div></div>')
    z, m = pos(0.0, lo, hi), pos(d["m"], lo, hi)
    a, b = pos(d["m"] - d["se"], lo, hi), pos(d["m"] + d["se"], lo, hi)
    left, width = min(z, m), abs(m - z)
    colour = "pos" if d["m"] >= 0 else "neg"
    sub = f"{d['n']} trades · {d['hit']:.0%} hit"
    value = f"{cents(d['m'])}<small>±{d['se'] * 100:.1f}c</small>"
    return (f'<div class="brow"><div class="blabel">{label}<small>{sub}</small></div>'
            f'<div class="btrack"><div class="bzero" style="left:{z:.2f}%"></div>'
            f'<div class="bbar {colour}" style="left:{left:.2f}%;width:{width:.2f}%"></div>'
            f'<div class="bwhisk" style="left:{min(a, b):.2f}%;width:{abs(b - a):.2f}%"></div>'
            f'<div class="bcap" style="left:{a:.2f}%"></div>'
            f'<div class="bcap" style="left:{b:.2f}%"></div></div>'
            f'<div class="bval">{value}</div></div>')


def filter_stats(all_rows, me, ms):
    xs = [r for r in all_rows if r["edge"] >= me and r["sure"] >= ms]
    if not xs:
        return {"n": 0, "m": 0.0, "se": 0.0, "hit": 0.0, "pred": 0.0}
    vals = np.array([r["profit"] for r in xs])
    m = float(vals.mean())
    se = float(vals.std(ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0
    return {"n": len(xs), "m": m, "se": se,
            "hit": float(np.mean(vals > 0)),
            "pred": float(np.mean([r["edge"] for r in xs]))}


def collect(tours, a):
    rng = np.random.default_rng(a.seed)
    all_rows, summaries = [], []
    for t in tours:
        res = analyse(t, a.draws, a.prior_sd, a.model_game_noise, rng)
        base = rows(t, res, a.fee, 0.0, 0.0)
        brier, logs = score(t, res)
        title_sum = sum(t.title.values())
        trades = []
        for r in base:
            i = t.index(r["team"])
            pr = profit(t, i, r["side"], a.fee)
            cost = r["market"] if r["side"] == "YES" else 1.0 - r["market"]
            d = {
                **r,
                "tournament": t.name,
                "index": i,
                "profit": pr,
                "cost": cost,
                "cash": a.stake / cost * pr if cost > 0 else 0.0,
                "team_result": "champion" if t.winner == i else "out",
                "trade_result": "WON" if pr > 0 else "LOST",
                "action": (r["side"] if r["edge"] >= a.min_edge and r["sure"] >= a.min_sure
                           else "-"),
            }
            all_rows.append(d)
            if d["action"] != "-":
                trades.append(d)

        vals = [r["profit"] for r in trades]
        m, se = mean_se(vals)
        summaries.append({
            "name": t.name,
            "teams": len(t.teams),
            "winner": t.teams[t.winner],
            "n": len(trades),
            "mean": m,
            "se": se,
            "hit": float(np.mean([v > 0 for v in vals])) if vals else 0.0,
            "cash": sum(r["cash"] for r in trades),
            "cash_no_wv": sum(r["cash"] for r in trades if r["team"] != "West Virginia"),
            "top_model": max(r["model"] for r in base),
            "top_market": max(r["market"] for r in base) / title_sum,
            "brier": brier,
            "logs": logs,
        })
    return all_rows, summaries


def data_csv(all_rows):
    out = io.StringIO()
    w = csv.writer(out)
    w.writerow(["tournament", "team", "model", "sd", "lo90", "hi90", "market",
                "edge", "sure", "side", "action", "team_result", "trade_result",
                "profit_per_contract", "cash_on_100"])
    for r in sorted(all_rows, key=lambda x: (x["tournament"], -x["edge"])):
        w.writerow([r["tournament"], r["team"],
                    f"{r['model']:.4f}", f"{r['sd']:.4f}",
                    f"{r['lo']:.4f}", f"{r['hi']:.4f}", f"{r['market']:.4f}",
                    f"{r['edge']:.4f}", f"{r['sure']:.4f}", r["side"], r["action"],
                    r["team_result"], r["trade_result"],
                    f"{r['profit']:.4f}", f"{r['cash']:.2f}"])
    return out.getvalue()


def trade_table(trades):
    body = ""
    for r in sorted(trades, key=lambda x: (-x["profit"], x["tournament"])):
        cls = "won" if r["profit"] > 0 else "lost"
        side = "yes" if r["side"] == "YES" else "no"
        body += (f'<tr class="{cls}"><td>{esc(pretty(r["tournament"]))}</td>'
                 f'<td><b>{esc(r["team"])}</b></td>'
                 f'<td><span class="pill {side}">{r["side"]}</span></td>'
                 f'<td class="num">{pct(r["market"])}</td>'
                 f'<td class="num">{pct(r["model"])}</td>'
                 f'<td class="num {"pos" if r["edge"] >= 0 else "neg"}">{cents(r["edge"])}</td>'
                 f'<td class="num">{r["sure"]:.0%}</td>'
                 f'<td>{r["team_result"]}</td>'
                 f'<td class="{"pos" if r["profit"] > 0 else "neg"}">{r["trade_result"]}</td>'
                 f'<td class="num {"pos" if r["profit"] > 0 else "neg"}">{cents(r["profit"])}</td>'
                 f'<td class="num {"pos" if r["cash"] > 0 else "neg"}">{signed_usd(r["cash"])}</td></tr>')
    return (f'<div class="table-wrap"><table><thead><tr>'
            f'<th>Tournament</th><th>Team</th><th>Side</th><th class="num">Price</th>'
            f'<th class="num">Model</th><th class="num">Edge</th><th class="num">Sure</th>'
            f'<th>Team result</th><th>Trade</th><th class="num">Per $1</th>'
            f'<th class="num">On $100</th></tr></thead><tbody>{body}</tbody></table></div>')


def all_table(all_rows):
    body = ""
    for r in sorted(all_rows, key=lambda x: (x["tournament"], -x["edge"])):
        act = r["action"]
        pill = (f'<span class="pill {"yes" if act == "YES" else "no"}">{act}</span>'
                if act != "-" else '<span class="mut">—</span>')
        body += (f'<tr><td>{esc(pretty(r["tournament"]))}</td><td>{esc(r["team"])}</td>'
                 f'<td class="num">{pct(r["model"])}</td>'
                 f'<td class="num">{pct(r["lo"])}</td><td class="num">{pct(r["hi"])}</td>'
                 f'<td class="num">{pct(r["market"])}</td>'
                 f'<td class="num {"pos" if r["edge"] >= 0 else "neg"}">{cents(r["edge"])}</td>'
                 f'<td class="num">{r["sure"]:.0%}</td><td>{pill}</td>'
                 f'<td>{r["team_result"]}</td>'
                 f'<td class="num {"pos" if r["profit"] > 0 else "neg"}">{cents(r["profit"])}</td></tr>')
    return (f'<div class="scroll"><table><thead><tr><th>Tournament</th><th>Team</th>'
            f'<th class="num">Model</th><th class="num">5%</th><th class="num">95%</th>'
            f'<th class="num">Market</th><th class="num">Edge</th><th class="num">Sure</th>'
            f'<th>Action</th><th>Team result</th><th class="num">Per $1</th>'
            f'</tr></thead><tbody>{body}</tbody></table></div>')


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dir", default="replays")
    p.add_argument("--out", default="report.html")
    p.add_argument("--data", default="report_data.csv")
    p.add_argument("--fee", type=float, default=FEE)
    p.add_argument("--min-edge", type=float, default=0.03)
    p.add_argument("--min-sure", type=float, default=0.60)
    p.add_argument("--stake", type=float, default=100.0,
                   help="cash staked per paper trade")
    p.add_argument("--draws", type=int, default=1000)
    p.add_argument("--prior-sd", type=float, default=PRIOR_SD)
    p.add_argument("--model-game-noise", type=float, default=GAME_NOISE)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)

    tours = []
    for path in sorted(Path(a.dir).glob("*.csv")):
        tours += load_csv(path)
    if not tours:
        raise SystemExit(f"no CSVs in {a.dir}")
    for t in tours:
        check(t)
        if t.winner is None:
            raise SystemExit(f"{t.name}: no winner row, so P&L cannot be settled")

    all_rows, summaries = collect(tours, a)
    trades = [r for r in all_rows if r["action"] != "-"]
    total_cash = sum(r["cash"] for r in trades)
    total_no_wv = sum(r["cash"] for r in trades if r["team"] != "West Virginia")
    total_vals = [r["profit"] for r in trades]
    total_mean, total_se = mean_se(total_vals)
    total_hit = float(np.mean([v > 0 for v in total_vals])) if total_vals else 0.0

    csv_text = data_csv(all_rows)
    Path(a.data).write_text(csv_text)
    b64 = base64.b64encode(csv_text.encode()).decode()
    download = f"data:text/csv;base64,{b64}"

    # Cash charts. Separate scales: the no-WV side is all negative, so it would
    # vanish beside the +$1,093 Crown bar on a shared axis.
    all_vals = [s["cash"] for s in summaries] + [total_cash]
    all_lo = min(0.0, min(all_vals))
    all_hi = max(0.0, max(all_vals)) * 1.08 or 1.0
    no_vals = [s["cash_no_wv"] for s in summaries] + [total_no_wv]
    no_lo = min(0.0, min(no_vals)) * 1.08
    no_hi = max(0.0, max(no_vals)) or 1.0

    cash_all_rows = ""
    cash_no_rows = ""
    for s in summaries:
        cash_all_rows += bar(esc(pretty(s["name"])), f"all {s['n']} trades", s["cash"],
                             all_lo, all_hi, "pos" if s["cash"] >= 0 else "neg",
                             signed_usd(s["cash"]))
        cash_no_rows += bar(esc(pretty(s["name"])), "same trades", s["cash_no_wv"],
                            no_lo, no_hi, "neg", signed_usd(s["cash_no_wv"]))
    cash_all_rows += bar("Total", f"all {len(trades)} trades", total_cash,
                         all_lo, all_hi, "pos" if total_cash >= 0 else "neg",
                         signed_usd(total_cash))
    cash_no_rows += bar("Total", "same trades", total_no_wv,
                        no_lo, no_hi, "neg", signed_usd(total_no_wv))

    # Filter chart.
    fs = []
    for me, ms in FILTERS:
        d = filter_stats(all_rows, me, ms)
        fs.append((me, ms, d))
    valid = [d for _, _, d in fs if d["n"] > 0]
    flo = min(0.0, min(d["m"] - d["se"] for d in valid))
    fhi = max(0.0, max(d["m"] + d["se"] for d in valid)) * 1.12 or 1.0
    filter_rows = ""
    for me, ms, d in fs:
        label = f"{me * 100:.0f}c / {ms * 100:.0f}%"
        filter_rows += whisker(label, "", d, flo, fhi)

    # Forecast table.
    forecast_rows = ""
    for s in summaries:
        b, l = s["brier"], s["logs"]
        model_wins = b[0] < b[1]
        forecast_rows += (f'<tr><td>{esc(pretty(s["name"]))}</td><td>{s["teams"]}</td>'
                          f'<td class="num">{b[0] * 100:.3f}</td><td class="num">{b[1] * 100:.3f}</td>'
                          f'<td class="num">{l[0]:.3f}</td><td class="num">{l[1]:.3f}</td>'
                          f'<td class="{"pos" if model_wins else "neg"}">'
                          f'{"model" if model_wins else "market"}</td></tr>')

    cards = [
        ("Tournaments", len(summaries), "real, settled"),
        ("Teams", sum(s["teams"] for s in summaries), "NCAA + Crown"),
        ("Paper trades", len(trades), f"{a.min_edge * 100:.0f}c edge / {a.min_sure * 100:.0f}% sure"),
        ("Paper P&L", signed_usd(total_cash), f"on {usd(a.stake * len(trades))} staked"),
        ("Without WV", signed_usd(total_no_wv), "the fragile part"),
    ]
    card_html = "".join(
        f'<div class="card"><div class="k">{k}</div>'
        f'<div class="v {"pos" if isinstance(v, str) and v.startswith("+") else "neg" if isinstance(v, str) and v.startswith("-") else ""}">{v}</div>'
        f'<div class="s">{s}</div></div>' for k, v, s in cards)

    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Bracket model — real 2026 replay</title><style>{CSS}</style></head>
<body><div class="wrap">
<header class="hero">
  <div class="eyebrow">Real prices · settled results</div>
  <h1>The bracket model, tested for real</h1>
  <p class="sub">The model prices every team’s title chance from the bracket. When its number differs from the market by more than the fee, that gap is the “edge”. This is what happened when we ran it on Kalshi prices from before the tournament, then settled every trade.</p>
</header>

<div class="alert"><div class="icon">!</div><div><b>Fragile result.</b> One 8c West Virginia longshot is almost the entire profit. Remove that single trade and the eight paper trades lose <b>{usd(abs(total_no_wv))}</b>. The 64-team NCAA tournament, which is the meaningful sample, lost money.</div></div>

<section class="cards">{card_html}</section>

<section class="panel">
  <h2>The whole result in one picture</h2>
  <p class="note">Cash P&amp;L if you staked <b>{usd(a.stake)} per trade</b> at the midpoint price. The Crown is an 8-team tournament; the NCAA is 64 teams. The right panel removes the single West Virginia winner.</p>
  <div class="grid2">
    <div class="box">
      <h3>All trades</h3>
      <div class="chart"><div class="chart-head"><span>loss</span><span>profit</span></div>{cash_all_rows}</div>
      <div class="legend"><span><i class="sw green"></i>profit</span><span><i class="sw red"></i>loss</span></div>
    </div>
    <div class="box">
      <h3>Without West Virginia</h3>
      <div class="chart"><div class="chart-head"><span>loss</span><span>zero</span></div>{cash_no_rows}</div>
      <div class="legend"><span><i class="sw red"></i>loss</span></div>
    </div>
  </div>
</section>

<section class="panel">
  <h2>Filter trade-off</h2>
  <p class="note">Every team is scored once, then the same draws are filtered at different thresholds. The bar is the average realised profit per <b>$1 contract</b>; the whisker is ±1 standard error. A tight filter gives no trades. A loose filter gives a big number with a huge error bar.</p>
  <div class="chart">
    <div class="chart-head"><span>loss</span><span>profit</span></div>
    {filter_rows}
  </div>
  <div class="legend"><span><i class="sw green"></i>average</span><span><i class="sw whisk"></i>±1 SE</span></div>
  <p class="note" style="margin-top:12px">The 90% filter has <b>zero trades</b>. The 60% filter is the one used on this page: {len(trades)} trades, {cents(total_mean)} per contract, {total_hit:.0%} hit rate. That is not statistically distinguishable from zero.</p>
</section>

<section class="panel">
  <h2>Forecast scorecard</h2>
  <p class="note">Brier and log loss are proper scoring rules: lower is better. Brier is ×100 so the digits are readable. The market won the big tournament; the model won the small one.</p>
  <div class="table-wrap"><table><thead><tr><th>Tournament</th><th class="num">Teams</th>
    <th class="num">Model Brier</th><th class="num">Market Brier</th>
    <th class="num">Model log loss</th><th class="num">Market log loss</th><th>Better</th>
  </tr></thead><tbody>{forecast_rows}</tbody></table></div>
</section>

<section class="panel">
  <h2>The trades</h2>
  <p class="note">Eight paper trades at {a.min_edge * 100:.0f}c edge and {a.min_sure * 100:.0f}% sure. “Per $1” is profit per $1 payout contract after the {a.fee * 100:.1f}c fee. “On $100” is cash staked, so contracts = stake ÷ cost.</p>
  {trade_table(trades)}
</section>

<section class="panel">
  <h2>All 72 team probabilities</h2>
  <p class="note">Every team in the two tournaments, sorted by edge. The 90% range is the 5th–95th percentile over 1,000 jittered runs.</p>
  <a class="btn" download="bracket_report_data.csv" href="{download}">Download the data (CSV)</a>
  <a class="btn ghost" href="#method">Method &amp; caveats</a>
  <details><summary>Show the full table</summary>{all_table(all_rows)}</details>
</section>

<section class="panel" id="method">
  <h2>Method and caveats</h2>
  <div class="grid2">
    <div class="box">
      <h3>What was tested</h3>
      <ul>
        <li><b>2026 NCAA men:</b> 64 teams, cutoff 19 Mar 2026 12:00 UTC, after the First Four, before the Round of 64. Champion: Michigan.</li>
        <li><b>2026 Crown:</b> 8 teams, cutoff 1 Apr 2026 12:00 UTC. Champion: West Virginia.</li>
        <li>Prices: Kalshi historical 1-minute candles, midpoint of YES bid/ask.</li>
        <li>Ratings: Elo from regular-season results before the tournament; no future games.</li>
        <li>Model: exact bracket simulation, 1,000 jittered runs, {a.fee * 100:.1f}c fee.</li>
      </ul>
    </div>
    <div class="box">
      <h3>What it does not prove</h3>
      <ul>
        <li>Two tournaments is a tiny sample. One champion is close to a coin flip.</li>
        <li>The +{total_mean * 100:.1f}c average is driven by one 8c winner; the error bar is ±{total_se * 100:.1f}c.</li>
        <li>Midpoints are not fills. An 8c contract may not absorb {usd(a.stake)} of cash.</li>
        <li>No slippage, no liquidity limit, no taxes, no correlation adjustment.</li>
        <li>The model is a gap finder, not a better forecaster: the market beat it on the 64-team test.</li>
      </ul>
    </div>
  </div>
  <p class="note" style="margin-top:14px">Reproduce with <code>python3 replay.py --season 2026</code>, then <code>python3 report.py</code>. Historical data from Kalshi’s public API. This is a research note, not financial advice.</p>
</section>

<footer>Generated from {', '.join(esc(pretty(t.name)) for t in tours)}. Brier and log loss use normalized market title prices. Paper P&amp;L assumes every trade fills at the midpoint. Past results do not guarantee future results.</footer>
</div></body></html>
"""
    Path(a.out).write_text(page)
    print(f"wrote {a.out} ({len(tours)} tournaments, {len(trades)} trades)")
    print(f"wrote {a.data} ({len(all_rows)} team rows)")


if __name__ == "__main__":
    main()
