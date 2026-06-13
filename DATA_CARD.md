# Data Card

## Sources

| Dataset | Purpose | Local path | Redistribution |
| --- | --- | --- | --- |
| [martj42/international_results](https://github.com/martj42/international_results) | Senior international training results from 2000 onward | `data/raw/international_results.csv` | CC0 |
| [football-data.org](https://www.football-data.org/documentation/quickstart) | Optional current fixture/result refresh | `data/cache/` | API terms apply |
| [openfootball/worldcup](https://github.com/openfootball/worldcup) | Public fallback fixture/result snapshot | `data/cache/` | Public domain project |
| [EA Sports FC 26 ratings](https://www.kaggle.com/datasets/justdhia/ea-sports-fc-26-player-ratings) | Approximate individual player quality | `data/player_ratings.csv` | CC0 dataset |
| Published 2026 squad tables | Registered-squad snapshot and roster membership | `data/world_cup_rosters.csv` | Generated locally; source metadata retained |

Third-party bulk data is downloaded locally and excluded from Git. Committed
example CSVs document optional player-input schemas without redistributing the
full sources.

## Provenance

Normalized match rows retain source, retrieval time, stage, neutral-site
status, and score provenance. Team aliases pass through one canonical mapping
before feature construction or player joins.

The registered-squad importer retains its retrieval timestamp and source URL.
FC 26 ratings are joined to those registered players; players missing from the
game dataset remain in the roster with an explicit team-based imputation.

## Time Safety

Features are updated after a match is predicted, never before it. Availability
reports require `reported_at < kickoff`, and lineup records require
`announced_at < kickoff`. Same-day or future match results cannot enter the
current row's feature state.

## Local Files

Generated and private files are ignored:

- `data/raw/`, `data/cache/`, and `data/processed/`
- `data/player_ratings.csv`
- `data/world_cup_rosters.csv`
- `data/player_availability.csv`
- `data/starting_lineups.csv`
- `artifacts/`
- `.env`

The `.gitkeep` files preserve the expected empty directory structure.

## Known Limitations

- Public fallback feeds can lag official fixture changes.
- Team-name normalization requires maintenance when sources introduce aliases.
- Squad announcements and replacements can change after a snapshot is fetched.
- Historical results do not contain complete point-in-time player context.
- Availability labels may be uncertain or source-dependent.
