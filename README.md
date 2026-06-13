# FIFA 2026 Match Prediction

[![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB.svg)](https://www.python.org/)
[![uv](https://img.shields.io/badge/package%20manager-uv-DE5FE9.svg)](https://docs.astral.sh/uv/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A reproducible ML project for forecasting FIFA World Cup matches. It produces
regulation-time home-win, draw, and away-win probabilities, plus
eventual-winner probabilities for knockout fixtures.

The project combines a leakage-safe match model with an optional, transparent
player-context adjustment. It is a portfolio and fan-analysis project, not a
betting system.

## Highlights

- Dynamic Elo, recent form, goals, rest, venue, host, and competition features.
- Chronological rolling-origin validation and out-of-fold calibration.
- Comparison of Elo, multinomial logistic regression, and histogram gradient
  boosting against a naive class-frequency baseline.
- Interactive team selection, registered-squad review, and editable starting
  XIs before a prediction is shown.
- Timestamped player ratings, injuries/availability, and projected or confirmed
  lineups.
- Predictions for upcoming and completed tournament matches without leaking a
  completed match's result into its own features.
- Optional football-data.org refresh with cached and public-data fallbacks.

## Quick Start

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and run:

```powershell
git clone <your-repository-url>
cd FIFA_Predict
uv sync --locked --dev
uv run fifa-predict audit
uv run fifa-predict players
uv run fifa-predict train
```

The API token is optional. In PowerShell:

```powershell
$env:FOOTBALL_DATA_API_TOKEN="your-token"
```

Set `FOOTBALL_DATA_API_TOKEN` in the process environment to prefer the
[football-data.org](https://www.football-data.org/) World Cup feed. Without a
token, the project uses its cache and then the public
[openfootball/worldcup](https://github.com/openfootball/worldcup) snapshot.
`.env.example` documents the variable for users whose shell or development
environment supports dotenv files; the application does not load `.env`
automatically.

If the global uv cache is unavailable on Windows:

```powershell
$env:UV_CACHE_DIR="$PWD\.uv-cache"
$env:UV_PYTHON_INSTALL_DIR="$PWD\.uv-python"
uv sync --locked --dev
```

## Interactive Prediction

On Windows:

```powershell
.\predict.cmd
```

On any supported platform:

```bash
uv run fifa-predict match
```

Enter two team names, review each numbered squad and proposed XI, accept the
suggestion or enter 11 squad numbers, and receive the probabilities. Known
projected or confirmed lineups are proposed first; otherwise a balanced XI is
built from the available rated squad.

Use `--offline` to prevent refresh attempts:

```bash
uv run fifa-predict match --offline
```

## Other Commands

```bash
uv run fifa-predict audit
uv run fifa-predict players
uv run fifa-predict train
uv run fifa-predict predict
uv run fifa-predict tournament
uv run fifa-predict predict --offline
```

`predict` retrains through the current cutoff and writes dated CSV and JSON
files under `artifacts/`.

`tournament` predicts every tournament fixture, including completed matches.
It advances state chronologically, so each prediction uses only information
available before that kickoff. Completed rows also contain the actual score,
actual result, and whether the predicted class was correct.

## Notebooks

Run Jupyter with:

```bash
uv run jupyter lab
```

Then execute:

1. `notebooks/01_data_audit.ipynb`
2. `notebooks/02_model_training.ipynb`
3. `notebooks/03_current_predictions.ipynb`

Reusable logic lives in `src/fifa_predict`; the notebooks are thin,
reproducible analysis interfaces rather than the only implementation.

## Player Context

`uv run fifa-predict players` creates a local registered-squad snapshot and
joins players to the CC0
[EA Sports FC 26 dataset](https://www.kaggle.com/datasets/justdhia/ea-sports-fc-26-player-ratings).
The game rating is an approximate quality signal, not an objective scouting
grade. Unmatched registered players are retained and explicitly imputed rather
than silently dropped.

Optional local inputs can be created from the committed examples:

```powershell
Copy-Item data\player_availability.example.csv data\player_availability.csv
Copy-Item data\starting_lineups.example.csv data\starting_lineups.csv
```

- `player_availability.csv` records available, doubtful, injured, suspended,
  or unavailable players. `reported_at` must precede kickoff.
- `starting_lineups.csv` uses `projected_starter` or `starter`.
  `announced_at` must precede kickoff.
- Eleven timely `starter` rows mark a lineup as confirmed.

When both teams have player data, the application adjusts the decisive win
probabilities using expected-XI quality, top-18 depth, and quality lost through
unavailability. The draw probability is retained. This is a conservative,
explicit contextual overlay rather than a historically trained feature,
because the match-results source does not contain point-in-time injuries and
lineups.

## Model

Every feature is computed strictly before kickoff:

- Dynamic Elo and Elo expectation.
- Rolling 5- and 10-match form, scoring, conceding, and result rates.
- Rest days and neutral-site context.
- Host-country and home-advantage indicators.
- Competition importance.

Models are evaluated with rolling-origin multiclass log loss. Brier score,
calibration, and accuracy are supporting metrics. Calibration is fitted from
time-safe out-of-fold probabilities, and the final model is refitted through
the prediction cutoff. The more complex model is retained only when it improves
on simpler alternatives.

See [MODEL_CARD.md](MODEL_CARD.md) and [DATA_CARD.md](DATA_CARD.md) for the
intended use, validation design, provenance, and limitations.

## Development

```bash
uv sync --locked --dev
uv run --frozen pytest --cov=fifa_predict
uv run --frozen fifa-predict --help
```

Tests use local fixtures and do not require network access. Downloaded data,
caches, model binaries, prediction exports, secrets, and virtual environments
are excluded from Git.

Contributions are welcome through [CONTRIBUTING.md](CONTRIBUTING.md). This
project is released under the [MIT License](LICENSE).
