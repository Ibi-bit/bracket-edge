#!/usr/bin/env python3
"""Render model output, or plain market prices, as one self-contained HTML page.

A complete knockout bracket gets the model overlay: market tick, model mean and
its 90% range. Anything else, such as a political field, shows market prices
only. No web server, no external assets; one inline script scales the stake.
"""
import argparse
import html

import numpy as np

from bracket_edge import (
    DRAWS, FEE, GAME_NOISE, MIN_EDGE, MIN_SURE, PRIOR_SD,
    analyse, check, load_csv, rows,
)

CSS = """
:root{
  --bg:#070b14;--panel:#0f172a;--panel2:#111c33;--line:#1e2b45;
  --text:#e5edf9;--muted:#8fa1bd;--teal:#5eead4;--amber:#fbbf24;
  --green:#34d399;--red:#f87171;--track:#1b2940;
}
*{box-sizing:border-box}
body{margin:0;background:radial-gradient(1200px 600px at 15% -10%,#13213d 0%,var(--bg) 55%);
  color:var(--text);font:14px/1.5 ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
  -webkit-font-smoothing:antialiased}
.wrap{max-width:1120px;margin:0 auto;padding:34px 20px 64px}
.hero h1{font-size:clamp(26px,4vw,38px);line-height:1.1;margin:6px 0 10px;letter-spacing:-.02em}
.eyebrow{font-size:11px;letter-spacing:.18em;text-transform:uppercase;color:var(--teal);font-weight:700}
.sub{max-width:760px;color:var(--muted);margin:0 0 24px}
.stake-row{display:flex;align-items:center;gap:12px;flex-wrap:wrap;margin:0 0 22px;
  padding:12px 14px;border:1px solid var(--line);border-radius:12px;
  background:rgba(17,28,51,.6);max-width:760px}
.stake-row label{font-size:11px;letter-spacing:.1em;text-transform:uppercase;
  color:var(--muted);font-weight:700}
.money-input{display:flex;align-items:center;gap:2px;background:#0b1322;
  border:1px solid var(--line);border-radius:8px;padding:4px 8px}
.money-input span{color:var(--muted)}
.money-input input{width:90px;background:none;border:0;color:var(--text);
  font:inherit;font-weight:700;outline:none}
.stake-row .hint{color:var(--muted);font-size:11px}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin:0 0 22px}
.card{background:linear-gradient(180deg,var(--panel2),var(--panel));border:1px solid var(--line);
  border-radius:14px;padding:14px 16px}
.card .k{font-size:11px;letter-spacing:.1em;text-transform:uppercase;color:var(--muted)}
.card .v{font-size:24px;font-weight:700;margin-top:4px;font-variant-numeric:tabular-nums}
.card .s{font-size:10px;color:var(--muted);margin-top:2px;font-weight:500}
.panel{background:linear-gradient(180deg,rgba(17,28,51,.9),rgba(15,23,42,.9));
  border:1px solid var(--line);border-radius:18px;padding:20px;margin:0 0 18px;
  box-shadow:0 20px 50px rgba(0,0,0,.25)}
.tour-head{display:flex;align-items:baseline;justify-content:space-between;gap:14px;
  flex-wrap:wrap;margin-bottom:8px}
.tour-head h2{font-size:18px;margin:0}
.tour-head span{color:var(--muted);font-size:12px}
.note{color:var(--muted);font-size:12px;margin:0 0 12px}
.axis,.row{display:grid;grid-template-columns:minmax(96px,1.1fr) minmax(220px,4fr) 64px 58px;
  gap:14px;align-items:center}
.axis{padding:0 0 6px}
.axis-track{display:flex;justify-content:space-between;font-size:10px;color:var(--muted);
  font-variant-numeric:tabular-nums}
.rows{border-top:1px solid var(--line)}
.row{padding:8px 0;border-bottom:1px solid rgba(30,43,69,.55)}
.row:hover{background:rgba(94,234,212,.035)}
.name{font-weight:600;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.track{position:relative;height:24px}
.track:before{content:"";position:absolute;left:0;right:0;top:11px;height:2px;
  background:var(--track);border-radius:2px}
.range{position:absolute;top:9px;height:6px;border-radius:4px;min-width:2px;
  background:rgba(94,234,212,.16);border:1px solid rgba(94,234,212,.38)}
.model{position:absolute;top:6px;width:12px;height:12px;border-radius:50%;background:var(--teal);
  transform:translateX(-50%);box-shadow:0 0 0 2px #0f172a,0 0 12px rgba(94,234,212,.55)}
.market{position:absolute;top:3px;width:2px;height:18px;border-radius:2px;background:var(--amber);
  transform:translateX(-50%)}
.bar{position:absolute;left:0;top:7px;height:10px;border-radius:5px;min-width:2px;
  background:linear-gradient(90deg,rgba(251,191,36,.55),var(--amber))}
.edge{text-align:right;font-variant-numeric:tabular-nums;font-weight:700;font-size:13px}
.edge.pos{color:var(--green)}.edge.neg{color:var(--red)}
.edge.mut{color:var(--muted)}.edge.amber{color:var(--amber)}
.money{display:block;font-size:10px;font-weight:600;color:var(--muted);margin-top:1px}
.act{display:flex;justify-content:flex-end}
.pill{font-size:10px;font-weight:800;letter-spacing:.06em;padding:3px 7px;border-radius:999px;
  border:1px solid transparent;min-width:42px;text-align:center}
.pill.yes{color:var(--green);background:rgba(52,211,153,.12);border-color:rgba(52,211,153,.35)}
.pill.no{color:var(--red);background:rgba(248,113,113,.12);border-color:rgba(248,113,113,.35)}
.pill.none{color:var(--muted);border-color:var(--line)}
.legend{display:flex;flex-wrap:wrap;gap:16px;margin-top:12px;color:var(--muted);font-size:11px}
.legend span{display:flex;align-items:center;gap:6px}
.lg-dot{width:10px;height:10px;border-radius:50%;background:var(--teal)}
.lg-tick{width:2px;height:12px;background:var(--amber)}
.lg-range{width:22px;height:6px;border-radius:3px;
  background:rgba(94,234,212,.2);border:1px solid rgba(94,234,212,.4)}
.lg-bar{width:22px;height:8px;border-radius:4px;background:var(--amber)}
.scatter svg{width:100%;height:auto;display:block}
footer{color:var(--muted);font-size:11px;margin-top:28px;line-height:1.7}
@media (max-width:640px){
  .wrap{padding:24px 14px 48px}
  .axis,.row{grid-template-columns:minmax(76px,1fr) minmax(120px,3fr) 54px 46px;gap:8px}
  .panel{padding:15px;border-radius:14px}
  .name{font-size:13px}.edge{font-size:12px}
  .pill{min-width:36px;padding:2px 5px;font-size:9px}
}
"""


def esc(x):
    return html.escape(str(x))


def pct(x):
    return f"{x * 100:.1f}%"


def cents(x):
    return f"{x * 100:+.1f}c"


def usd(x):
    return f"${x:,.2f}"


def stake_profit(r, stake):
    """Dollars of edge for `stake` dollars of cost on one flagged row."""
    cost = r["market"] if r["side"] == "YES" else 1 - r["market"]
    return stake * r["edge"] / cost if cost > 0 else 0.0


def expected_profit(rows_, stake):
    """Sum of `stake_profit` over the flagged rows."""
    return sum(stake_profit(r, stake) for r in rows_ if r["action"] != "-")


def clamp(x):
    return max(0.0, min(1.0, float(x)))


def model_rows(t, a, rng):
    """Return (rows, note). Note is set when the bracket is not usable."""
    if not t.games:
        return None, "No bracket games; showing market prices only."
    try:
        check(t)
        res = analyse(t, a.draws, a.prior_sd, a.model_game_noise, rng)
    except ValueError as e:
        return None, f"No complete bracket ({esc(e)}); showing market prices only."
    return rows(t, res, a.fee, a.min_edge, a.min_sure), ""


def market_rows(t):
    out = [{"team": t.teams[i], "market": p} for i, p in t.title.items()]
    out.sort(key=lambda r: -r["market"])
    return out


def row_html(r, model, stake):
    name = esc(r["team"])
    if not model:
        price = clamp(r["market"])
        return (f'<div class="row" title="{name}: market {pct(price)}">'
                f'<div class="name">{name}</div>'
                f'<div class="track"><div class="bar" style="width:{price * 100:.2f}%"></div></div>'
                f'<div class="edge amber">{pct(price)}</div>'
                f'<div class="act"></div></div>')

    lo, hi = clamp(r["lo"]), clamp(r["hi"])
    mean, price = clamp(r["model"]), clamp(r["market"])
    edge, sure, action = r["edge"], r["sure"], r["action"]
    side = r["side"]
    cls = "pos" if edge >= 0 else "neg"
    pill = "yes" if action == "YES" else "no" if action == "NO" else "none"
    label = action if action != "-" else "—"
    cost = price if side == "YES" else 1 - price
    roi = edge / cost if cost > 0 else 0.0
    money = (f'<span class="money" data-edge="{edge:.6f}" data-cost="{cost:.6f}">'
             f'{usd(stake_profit(r, stake))}</span>'
             if action != "-" and cost > 0 else "")
    tip = (f"{name}: model {pct(mean)} (90% {pct(lo)}–{pct(hi)}), market {pct(price)}, "
           f"edge {cents(edge)} on {side} ({roi * 100:+.1f}% on cost), "
           f"sure {sure * 100:.0f}%")
    return (f'<div class="row" title="{esc(tip)}">'
            f'<div class="name">{name}</div>'
            f'<div class="track">'
            f'<div class="range" style="left:{lo * 100:.2f}%;width:{(hi - lo) * 100:.2f}%"></div>'
            f'<div class="model" style="left:{mean * 100:.2f}%"></div>'
            f'<div class="market" style="left:{price * 100:.2f}%"></div>'
            f'</div>'
            f'<div class="edge {cls}">{cents(edge)}{money}</div>'
            f'<div class="act"><span class="pill {pill}">{label}</span></div>'
            f'</div>')


def section(t, mr, model, note, stake):
    flagged = sum(1 for r in mr if r.get("action", "-") != "-")
    meta = f"{len(mr)} markets"
    if model:
        meta += f" · {flagged} flagged"
    if t.winner is not None and 0 <= t.winner < len(t.teams):
        meta += f" · winner {esc(t.teams[t.winner])}"

    axis = ('<div class="axis"><div></div><div class="axis-track">'
            '<span>0%</span><span>25%</span><span>50%</span><span>75%</span><span>100%</span>'
            '</div><div></div><div></div></div>')
    body = "".join(row_html(r, model, stake) for r in mr)
    if model:
        legend = ('<div class="legend">'
                  '<span><i class="lg-dot"></i>model mean</span>'
                  '<span><i class="lg-tick"></i>market YES ask</span>'
                  '<span><i class="lg-range"></i>90% range</span>'
                  '</div>')
    else:
        legend = ('<div class="legend"><span><i class="lg-bar"></i>'
                  'YES ask, sorted by price</span></div>')
    note_html = f'<p class="note">{note}</p>' if note else ""
    return (f'<section class="panel tour">'
            f'<div class="tour-head"><h2>{esc(t.name)}</h2><span>{meta}</span></div>'
            f'{note_html}{axis}<div class="rows">{body}</div>{legend}</section>')


def scatter(points):
    """Edge against sure: the two flag criteria, one point per team."""
    if not points:
        return ""
    w, h = 760, 360
    left, right, top, bottom = 58, 18, 18, 44
    emin = min(0.0, min(p["edge"] for p in points))
    emax = max(0.05, max(p["edge"] for p in points))
    pad = (emax - emin) * 0.08 or 0.01
    emin, emax = emin - pad, emax + pad

    def x(v):
        return left + v * (w - left - right)

    def y(v):
        return h - bottom - (v - emin) / (emax - emin) * (h - top - bottom)

    parts = [f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="edge against sure">']
    parts.append(f'<rect x="{left}" y="{top}" width="{w - left - right}" '
                 f'height="{h - top - bottom}" rx="8" fill="rgba(255,255,255,.02)"/>')

    for v in (0, 0.25, 0.5, 0.75, 1):
        parts.append(f'<line x1="{x(v):.1f}" y1="{top}" x2="{x(v):.1f}" y2="{h - bottom}" '
                     f'stroke="#1e2b45" stroke-width="1"/>')
        parts.append(f'<text x="{x(v):.1f}" y="{h - bottom + 16}" fill="#8fa1bd" '
                     f'font-size="10" text-anchor="middle">{v * 100:.0f}%</text>')
    for i in range(5):
        v = emin + (emax - emin) * i / 4
        parts.append(f'<line x1="{left}" y1="{y(v):.1f}" x2="{w - right}" y2="{y(v):.1f}" '
                     f'stroke="#1e2b45" stroke-width="1"/>')
        parts.append(f'<text x="{left - 8}" y="{y(v) + 3:.1f}" fill="#8fa1bd" '
                     f'font-size="10" text-anchor="end">{v * 100:.0f}c</text>')

    parts.append(f'<line x1="{x(0.9):.1f}" y1="{top}" x2="{x(0.9):.1f}" y2="{h - bottom}" '
                 f'stroke="#fbbf24" stroke-dasharray="4 4" opacity=".65"/>')
    parts.append(f'<text x="{x(0.9) - 6:.1f}" y="{top + 12}" fill="#fbbf24" '
                 f'font-size="10" text-anchor="end">sure 90%</text>')
    parts.append(f'<line x1="{left}" y1="{y(0.03):.1f}" x2="{w - right}" y2="{y(0.03):.1f}" '
                 f'stroke="#fbbf24" stroke-dasharray="4 4" opacity=".65"/>')
    parts.append(f'<text x="{left + 6}" y="{y(0.03) - 6:.1f}" fill="#fbbf24" '
                 f'font-size="10">edge 3c</text>')

    flagged = sorted((p for p in points if p["action"] != "-"),
                     key=lambda p: -p["edge"])[:24]
    for p in points:
        colour = ("#34d399" if p["action"] == "YES"
                  else "#f87171" if p["action"] == "NO" else "#64748b")
        tip = (f'{p["tournament"]} / {p["team"]}: edge {cents(p["edge"])}, '
               f'sure {p["sure"] * 100:.0f}%')
        parts.append(f'<circle cx="{x(clamp(p["sure"])):.1f}" cy="{y(p["edge"]):.1f}" r="4.5" '
                     f'fill="{colour}" stroke="#0f172a" stroke-width="1.5">'
                     f'<title>{esc(tip)}</title></circle>')
    for p in flagged:
        px, py = x(clamp(p["sure"])), y(p["edge"])
        if px > w - right - 72:
            parts.append(f'<text x="{px - 7:.1f}" y="{py + 4:.1f}" fill="#cbd5e1" '
                         f'font-size="10" text-anchor="end">{esc(p["team"])}</text>')
        else:
            parts.append(f'<text x="{px + 7:.1f}" y="{py + 4:.1f}" fill="#cbd5e1" '
                         f'font-size="10">{esc(p["team"])}</text>')

    parts.append(f'<text x="{(left + w - right) / 2:.1f}" y="{h - 8}" fill="#8fa1bd" '
                 f'font-size="11" text-anchor="middle">sure — share of runs the trade still profits</text>')
    cy = (top + h - bottom) / 2
    parts.append(f'<text x="14" y="{cy:.1f}" fill="#8fa1bd" font-size="11" text-anchor="middle" '
                 f'transform="rotate(-90 14 {cy:.1f})">edge per $1 contract (c)</text>')
    parts.append("</svg>")
    return ('<section class="panel scatter"><h2>Where the flags are</h2>'
            '<p class="note">Top right of the dashed lines is a trade: edge at least 3c and '
            'sure at least 90%. Hover a point for exact numbers.</p>'
            + "".join(parts) + "</section>")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", required=True)
    p.add_argument("--out", default="visual.html")
    p.add_argument("--fee", type=float, default=FEE)
    p.add_argument("--min-edge", type=float, default=MIN_EDGE)
    p.add_argument("--min-sure", type=float, default=MIN_SURE)
    p.add_argument("--draws", type=int, default=DRAWS)
    p.add_argument("--prior-sd", type=float, default=PRIOR_SD)
    p.add_argument("--model-game-noise", type=float, default=GAME_NOISE)
    p.add_argument("--stake", type=float, default=100.0,
                   help="dollars per flagged trade for the profit estimate")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)

    tours = load_csv(a.csv)
    rng = np.random.default_rng(a.seed)
    sections, points, market_prices = [], [], []
    total, flagged, best = 0, 0, None

    for t in tours:
        mr, note = model_rows(t, a, rng)
        model = mr is not None
        if not model:
            mr = market_rows(t)
            market_prices += [r["market"] for r in mr]
            if not mr:
                note = "No title prices in this group."
        else:
            for r in mr:
                points.append({**r, "tournament": t.name})
                if r["action"] != "-":
                    flagged += 1
                    best = r["edge"] if best is None else max(best, r["edge"])
        total += len(mr)
        sections.append(section(t, mr, model, note, a.stake))

    flagged_edges = [r["edge"] for r in points if r["action"] != "-"]
    total_profit = expected_profit(points, a.stake)

    if points:
        cards = [("Tournaments", len(tours)), ("Markets", total),
                 ("Flagged", flagged), ("Best edge", cents(best) if best is not None else "—")]
        if flagged:
            cards.append(("Expected profit",
                          f'<span id="total">{usd(total_profit)}</span>'
                          f'<div class="s" id="total-label">at {usd(a.stake)} staked per flagged trade</div>'))
    else:
        cards = [("Tournaments", len(tours)), ("Markets", total),
                 ("Top price", pct(max(market_prices, default=0))),
                 ("Median", pct(float(np.median(market_prices)) if market_prices else 0))]
    card_html = "".join(f'<div class="card"><div class="k">{k}</div>'
                        f'<div class="v">{v}</div></div>' for k, v in cards)
    sub = (f"Model vs market for knockout brackets. Any other set of markets, such as a "
           f"political field, shows prices only. Fee {a.fee * 100:.1f}c · edge per $1 "
           f"contract · flag at ≥{a.min_edge * 100:.1f}c and sure ≥{a.min_sure * 100:.0f}%.")

    stake_html = ""
    script = ""
    if flagged:
        stake_html = (f'<div class="stake-row"><label for="stake">Stake per flagged trade</label>'
                      f'<div class="money-input"><span>$</span>'
                      f'<input id="stake" type="number" min="0" step="10" value="{a.stake:g}"></div>'
                      f'<span class="hint">Theoretical: stake ÷ cost × edge, before liquidity and slippage.</span>'
                      f'</div>')
        script = """<script>
(() => {
  const stakeEl = document.getElementById('stake');
  const totalEl = document.getElementById('total');
  const labelEl = document.getElementById('total-label');
  if (!stakeEl || !totalEl || !labelEl) return;
  const fmt = n => '$' + n.toLocaleString('en-US', {minimumFractionDigits: 2, maximumFractionDigits: 2});
  const update = () => {
    const stake = Math.max(0, parseFloat(stakeEl.value) || 0);
    let total = 0;
    document.querySelectorAll('.money[data-edge]').forEach(el => {
      const cost = parseFloat(el.dataset.cost);
      const p = cost > 0 ? stake * parseFloat(el.dataset.edge) / cost : 0;
      total += p;
      el.textContent = fmt(p);
    });
    totalEl.textContent = fmt(total);
    labelEl.textContent = `at ${fmt(stake)} per flagged trade`;
  };
  stakeEl.addEventListener('input', update);
  update();
})();
</script>"""

    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Prediction market viewer</title><style>{CSS}</style></head>
<body><div class="wrap">
<header class="hero"><div class="eyebrow">Prediction market viewer</div>
<h1>Model vs market</h1><p class="sub">{esc(sub)}</p></header>
{stake_html}
<section class="cards">{card_html}</section>
{''.join(sections)}
{scatter(points)}
{script}
<footer>Generated from {esc(a.csv)}. Prices are YES asks. Expected profit assumes every
flagged trade pays the model edge; it ignores liquidity, slippage and being wrong. This is
a simulation, not financial advice.</footer>
</div></body></html>
"""
    with open(a.out, "w") as f:
        f.write(page)
    print(f"{len(tours)} sections -> {a.out}")


if __name__ == "__main__":
    main()
