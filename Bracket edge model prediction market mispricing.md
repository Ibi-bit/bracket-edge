# Bracket edge model: prediction market mispricing

Nov 19, 2026 · @Leon

## Summary

The model turns linked prediction market prices into its own win chance for each team, with a confidence level, and flags trades where the market price disagrees. On simulated data it earned +6.9c per $1 contract when the title market was mispriced and lost 4.7c when it was not, so it does not invent profit.

The synthetic results below are the controlled test, and predicted edges ran about two and a half times larger than realised ones. A real 2026 NCAA replay is included: the model was worse than the market on Brier and log loss, and no trade passed the 90% filter. One tournament is a sample, not a track record.

## The idea

A trade is only a hedge when one outcome is required for another, and the cleanest case is a knockout bracket. A team must win its next game to win the title, so its title price can never exceed its next-game price.

- **Hard violation:** title price above next-game price. This is the only gap that is inconsistent whatever you believe about the teams.
- **Model gap:** the model's win chance differs from the market's title price. This is a bet on the model, not a hedge.
- **Not an inconsistency:** best-of series, group stages and other non-elimination games, where losing one game does not end the run.

## How the model works

The model builds its own title probability for each team from the next-game markets, then simulates the whole bracket.

1. **Devig.** Each game's two prices are normalised to sum to 1, which removes the platform's margin.
2. **Rate teams.** A rating per team is fitted by logistic loss so rating gaps match the devigged game probabilities, pulled toward a prior rating such as Elo, seeds or past results.
3. **Build matchup odds.** A pair with a direct market uses its price. Every other pair uses the logistic of the rating gap.
4. **Simulate the bracket.** For each team, multiply its chance of beating each possible opponent, round by round. The result is a title probability, and the probabilities sum to 1 across teams.
5. **Measure uncertainty.** Repeat steps 2 to 4 100 times, jittering the game prices (0.25 in logit units) and the prior ratings (0.6). The spread of the results is the model's uncertainty.

Round-1 prices alone say nothing about strength across different halves of the bracket, so the prior rating is what links them. Without a prior, the model centres every pair on zero.

## Reading the output

Each team gets a mean win chance, a 90% range, and `sure`, the share of runs in which the trade still profits.

| Column | Meaning |
| --- | --- |
| model | Mean title probability across the 100 runs |
| sd | Standard deviation across the runs |
| 90% range | 5th to 95th percentile of the runs |
| market | Title price paid to buy YES |
| edge | Expected profit per $1 contract after the 0.02 fee, on the better side (YES or NO) |
| sure | Share of runs where that trade still profits after fees |
| action | The side to trade, shown only when edge is at least 0.03 and sure is at least 0.9, otherwise a dash |

Buying YES earns model minus price minus fee. Buying NO earns price minus model minus fee, treating the NO price as 1 minus the YES price.

In the sample data, team A shows a 49% model chance against a 34% market price. The edge is 13.0c, but `sure` is 0.89, just under the 0.90 filter, so no trade is flagged. Team C shows a 4.9c edge and `sure` 0.77, also filtered. The sample is there to show the filter rejecting a high-edge, borderline-confidence trade.

## Backtest results

On simulated data the flagged trades earned money only when the title market was mispriced. The setup is 8 teams, game prices with 0.2 logit noise and a 3% overround, a title market with a 5% overround, prior ratings with 0.6 noise, and a 0.02 fee.

| Scenario | Tournaments | Flagged trades | Mean profit per trade |
| --- | --- | --- | --- |
| Efficient title market (no noise) | 600 | 395 | -4.7c (± 2.1c) |
| Mispriced title market (0.5 logit noise) | 1,200 | 1,460 | +6.9c (± 1.0c) |

The confidence level ranks trades. In the mispriced run, among candidate trades with a predicted edge of at least 3%:

| sure | Trades | Mean profit per trade |
| --- | --- | --- |
| 0.5 to 0.8 | 1,172 | +0.9c (± 1.2c) |
| 0.8 to 0.9 | 817 | +1.9c (± 1.4c) |
| 0.9 to 1.0 | 1,460 | +6.9c (± 1.0c) |

- **Overstated edge:** flagged trades predicted a 17.1c edge but realised 6.9c, so shrink predicted edges by about 60%.
- **Weaker forecast, selective profit:** the model scored worse than the market on Brier (0.7134 against 0.6888) and log loss (1.5275 against 1.4718), and true probabilities scored 0.6726 and 1.4193. The market is scored after removing its overround. The profit comes from trading only large gaps, not from a better overall forecast.

## Real replay: 2026 NCAA men

`replay.py` rebuilds the actual 64-team bracket from Kalshi's historical game markets and takes the last one-minute candle before the cutoff.

- Cutoff: 2026-03-19 12:00 UTC, after the First Four and before the Round of 64. First-round and title prices are both from that moment.
- 32 first-round games, 64 teams; Michigan beat UConn in the final.
- Prior: Elo from 5,167 regular-season results before 2026-03-16. No tournament games are used.
- Prices are midpoints. The title market is normalised before scoring.

With 1,000 draws the model's top teams were Michigan 8.8%, Duke 8.0% and Arizona 7.8%. The normalised title market had Duke 16.0%, Michigan 14.3% and Arizona 14.3%.

| Metric | Model | Market |
| --- | --- | --- |
| Brier (x100, lower is better) | 87.0 | 79.6 |
| Log loss (lower is better) | 2.435 | 1.942 |

No team passed edge ≥ 3c and sure ≥ 90%, so the strategy made no trades. Among every team with edge ≥ 3c, the four NO trades were Duke +19c, Arizona +17c, Florida +10c and Michigan -83c, a mean of -9.3c per $1 contract. The loser is the eventual champion. The 90% filter avoided it only because Michigan's sure was 0.83; that is luck, not a validated rule.

At 100 draws the flag count is seed-sensitive (0 to 3), so use at least 1,000 draws for a stable `sure`.

## Limits and risks

The backtest shows the code does not invent profit, not that real markets are mispriced. The real replay is one tournament, so it cannot separate a bad model from a bad year.

- **Synthetic data:** the amount of market noise is my assumption, not a measurement.
- **Not a bracket:** politics, awards and other fields have no elimination structure, so the model overlay does not apply; the viewer can still show their prices.
- **Prior dependence:** edge quality tracks the quality of the prior rating, because round-1 prices do not link the two halves of a bracket.
- **Winner's curse:** trades are picked because the model disagrees most with the market, so the predicted edge is biased upward.
- **Fees and lockup:** a title market can tie up money until the tournament ends, so a small edge gives a poor annualised return.
- **Leg risk and liquidity:** a model gap is a bet, not a hedge, and thin markets may not fill at the shown price. Use the ask when buying and the bid when selling.
- **Resolution wording:** linked markets can resolve under slightly different rules, which can break a hedge.

This is general information, not financial advice.

## Running it

Four Python files — `bracket_edge.py`, `backtest.py`, `fetch_kalshi.py` and `visual.py` — need only numpy and the standard library. The first three do the modelling and data; `visual.py` turns the same CSV into a self-contained HTML page.

```
python3 bracket_edge.py
python3 backtest.py
python3 backtest.py --title-noise 0 --n 600
python3 backtest.py --csv snapshots.csv
python3 visual.py --csv snapshots.csv --out visual.html
```

For real data, `fetch_kalshi.py` writes the CSV straight from Kalshi's public API, which needs no key. Find the event tickers first, then pass the title event and each first-round game in bracket order:

```
python3 fetch_kalshi.py series KXNBAGAME
python3 fetch_kalshi.py snapshot --tournament ncaa \
    --title-event KXNCAA-26 --game KXNCAA-26R1A --game KXNCAA-26R1B
python3 bracket_edge.py --csv snapshots.csv
```

Prices are the YES ask, the price paid to buy. Title rows are written only for teams in the games, so a futures event with extra teams still loads. Add `--result TEAM` once the winner is known, which lets `backtest.py --csv` score the run.

The viewer is not bracket-only. `markets` dumps any Kalshi event as title rows, so a political field or an award market shows as a sorted price ladder:

```
python3 fetch_kalshi.py markets --event KXPRESNOMD-28 --out politics.csv
python3 visual.py --csv politics.csv --out politics.html
```

The model overlay appears only when the CSV has a complete bracket; otherwise the page shows market prices only. The two are separate questions: a bracket links games to a title, while a political field is one market per candidate and has no elimination structure to simulate.

`--csv` reads one row per price, rating or result. List the `game` rows in bracket order, one per first-round matchup. The number of teams must be a power of 2, and `title` rows must cover every team. A `prior` is a logit-scale rating, not a raw Elo number: a 400-point Elo gap is about 2.3 logit units, so use `(elo - 1500) / 173.7`.

```csv
tournament,type,a,b,pa,pb
t1,game,A,B,0.70,0.32
t1,game,C,D,0.55,0.47
t1,title,A,,0.34,
t1,prior,A,,1.0,
t1,winner,A,,,
```

- **`--fee`:** cost per contract, default 0.02.
- **`--min-edge`:** smallest edge to flag, default 0.03.
- **`--min-sure`:** smallest confidence to flag, default 0.9.
- **`--draws`:** runs used to measure uncertainty, default 100.
- **`--prior-sd` and `--model-game-noise`:** jitter on ratings and game prices, defaults 0.6 and 0.25.

## Next steps

The next step is real data, because every result above rests on simulated markets.

- [ ] Log game and title prices (fillable prices, not last traded) with `fetch_kalshi.py snapshot` for a few weeks, plus the results, in the CSV format
- [ ] Add prior ratings from a real source such as Elo or seeds
- [ ] Run the backtest on the logged data and check whether flagged gaps close in your favour
- [ ] For linked non-bracket markets, such as nominee then president, model the actual conditional graph; the bracket simulation does not apply
- [ ] Raise `--min-edge` to allow for the overstated edge, then paper trade before risking money
