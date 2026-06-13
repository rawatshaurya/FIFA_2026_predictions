from __future__ import annotations

import argparse
import json

from fifa_predict.pipeline import (
    audit_sources,
    predict_remaining,
    predict_tournament,
    train_from_sources,
)
from fifa_predict.interactive import run_interactive
from fifa_predict.players import (
    PlayerContextStore,
    refresh_player_ratings,
    refresh_world_cup_rosters,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="FIFA 2026 prediction pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in (
        "audit",
        "train",
        "predict",
        "tournament",
        "players",
        "match",
    ):
        command = subparsers.add_parser(name)
        command.add_argument(
            "--offline",
            action="store_true",
            help="Use local caches without network requests.",
        )
        command.add_argument(
            "--no-refresh",
            action="store_true",
            help="Prefer existing caches.",
        )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    refresh = not args.no_refresh
    if args.command == "match":
        run_interactive(offline=args.offline)
    elif args.command == "players":
        if args.offline:
            store = PlayerContextStore.from_csv()
            if not store.available:
                raise SystemExit(
                    "No local player ratings found; rerun without --offline."
                )
            if not store.rosters_available:
                raise SystemExit(
                    "No local registered rosters found; rerun without --offline."
                )
            ratings = store.ratings
            rosters = store.rosters
        else:
            ratings = refresh_player_ratings()
            rosters = refresh_world_cup_rosters(ratings)
        print(
            json.dumps(
                {
                    "players": len(ratings),
                    "national_teams": int(ratings["national_team"].nunique()),
                    "rating_date": str(ratings["rating_date"].max()),
                    "output": "data/player_ratings.csv",
                    "registered_squads": int(rosters["team"].nunique()),
                    "registered_players": len(rosters),
                    "roster_output": "data/world_cup_rosters.csv",
                },
                indent=2,
            )
        )
    elif args.command == "audit":
        audit = audit_sources(refresh=refresh, offline=args.offline)
        print(json.dumps(audit.summary(), indent=2))
    elif args.command == "train":
        bundle, audit = train_from_sources(
            refresh=refresh, offline=args.offline, save=True
        )
        print(
            json.dumps(
                {
                    "model": bundle.model_version,
                    "metrics": bundle.metrics,
                    "data": audit.summary(),
                },
                indent=2,
            )
        )
    elif args.command == "predict":
        predictions, paths, bundle, audit = predict_remaining(
            refresh=refresh, offline=args.offline
        )
        print(
            json.dumps(
                {
                    "model": bundle.model_version,
                    "predictions": len(predictions),
                    "csv": str(paths[0]),
                    "json": str(paths[1]),
                    "data": audit.summary(),
                },
                indent=2,
            )
        )
    else:
        predictions, paths, bundle, audit = predict_tournament(
            refresh=refresh, offline=args.offline
        )
        completed = [item for item in predictions if item.actual_result is not None]
        correct = sum(item.prediction_correct is True for item in completed)
        adjusted = sum(item.player_context_applied for item in predictions)
        confirmed_lineups = sum(
            item.home_lineup_confirmed is True for item in predictions
        ) + sum(item.away_lineup_confirmed is True for item in predictions)
        print(
            json.dumps(
                {
                    "model": bundle.model_version,
                    "fixtures": len(predictions),
                    "completed": len(completed),
                    "completed_accuracy": (
                        correct / len(completed) if completed else None
                    ),
                    "player_adjusted_fixtures": adjusted,
                    "confirmed_team_lineups": confirmed_lineups,
                    "csv": str(paths[0]),
                    "json": str(paths[1]),
                    "data": audit.summary(),
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
