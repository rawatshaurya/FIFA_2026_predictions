# Model Card

## Intended Use

FIFA Predict estimates regulation-time home-win, draw, and away-win
probabilities for senior international matches. For knockout fixtures it also
derives eventual-winner probabilities. It is intended for education, portfolio
work, and fan analysis.

It is not designed for betting, financial decisions, player evaluation, or
live in-match forecasting.

## Model Selection

The training pipeline compares:

- A class-frequency baseline.
- A dynamic-Elo probability baseline.
- Multinomial logistic regression.
- Histogram gradient boosting.

Selection uses chronological rolling-origin multiclass log loss. Brier score,
calibration, and accuracy are reported as supporting diagnostics. The selected
model must beat the naive baseline; a simpler model is retained when added
complexity does not improve rolling log loss.

Calibration is trained from time-safe out-of-fold predictions. The selected
model is then refitted on all completed matches available before the prediction
cutoff.

## Features

All model features are available strictly before kickoff:

- Dynamic Elo and expected result.
- Rolling 5- and 10-match results.
- Rolling goals scored and conceded.
- Rolling win, draw, and loss rates.
- Rest days.
- Neutral venue, home advantage, and host status.
- Competition importance.

Unseen teams receive documented default state values instead of causing a
prediction failure.

## Player Adjustment

Player quality, expected-XI quality, squad depth, and unavailable quality form
a separate conservative probability adjustment. This overlay is intentionally
not represented as a trained historical feature because the base results data
does not provide reliable point-in-time lineups and injury reports.

If either team lacks sufficient player data, the validated match-model
probability is returned unchanged.

## Knockout Matches

The regulation-time draw probability is allocated between the two teams using
their relative calibrated strength, shrunk toward 50/50 to reflect extra-time
and penalty uncertainty. Eventual-winner probabilities are therefore derived
quantities, not a separately trained shootout model.

## Limitations

- National-team strength can change rapidly between international windows.
- Game ratings are imperfect proxies for real player quality.
- Availability and projected-lineup files depend on timely, accurate reporting.
- The player overlay has not been validated as a causal injury or lineup model.
- Tactical matchups, travel conditions, manager changes, and live events are
  not modeled directly.
- Historical performance does not guarantee calibration in a new tournament.

Generated validation metrics are written to `artifacts/model_metrics.json` and
are intentionally not committed because they depend on the current data cutoff.
