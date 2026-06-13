from __future__ import annotations

import hashlib
import io
import difflib
import re
import unicodedata
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

from fifa_predict import config
from fifa_predict.teams import canonical_team

PLAYER_RATINGS_PATH = config.DATA_DIR / "player_ratings.csv"
PLAYER_AVAILABILITY_PATH = config.DATA_DIR / "player_availability.csv"
STARTING_LINEUPS_PATH = config.DATA_DIR / "starting_lineups.csv"
WORLD_CUP_ROSTERS_PATH = config.DATA_DIR / "world_cup_rosters.csv"

STATUS_DEFAULTS = {
    "available": 1.0,
    "doubtful": 0.5,
    "injured": 0.0,
    "suspended": 0.0,
    "unavailable": 0.0,
}
PLAYER_RATINGS_DATE = pd.Timestamp("2026-03-18", tz="UTC")


@dataclass(frozen=True)
class TeamPlayerContext:
    expected_xi_quality: float
    squad_depth_quality: float
    unavailable_quality: float
    availability_rate: float
    lineup_confirmed: bool
    player_data_available: bool
    expected_starters: int


@dataclass(frozen=True)
class MatchPlayerContext:
    home: TeamPlayerContext
    away: TeamPlayerContext

    @property
    def quality_difference(self) -> float:
        return self.home.expected_xi_quality - self.away.expected_xi_quality

    @property
    def depth_difference(self) -> float:
        return self.home.squad_depth_quality - self.away.squad_depth_quality

    @property
    def unavailable_quality_difference(self) -> float:
        return self.home.unavailable_quality - self.away.unavailable_quality


@dataclass(frozen=True)
class PlayerSelection:
    player_id: str
    player_name: str
    position: str
    overall_rating: int
    availability_probability: float


def _empty_context() -> TeamPlayerContext:
    return TeamPlayerContext(
        expected_xi_quality=0.0,
        squad_depth_quality=0.0,
        unavailable_quality=0.0,
        availability_rate=1.0,
        lineup_confirmed=False,
        player_data_available=False,
        expected_starters=0,
    )


class PlayerContextStore:
    """Point-in-time player quality, availability, and lineup information."""

    def __init__(
        self,
        ratings: pd.DataFrame | None = None,
        availability: pd.DataFrame | None = None,
        lineups: pd.DataFrame | None = None,
        rosters: pd.DataFrame | None = None,
    ) -> None:
        self.ratings = self._normalize_ratings(ratings)
        self.availability = self._normalize_availability(availability)
        self.lineups = self._normalize_lineups(lineups)
        self.rosters = self._normalize_rosters(rosters)

    @classmethod
    def from_csv(
        cls,
        ratings_path: Path = PLAYER_RATINGS_PATH,
        availability_path: Path = PLAYER_AVAILABILITY_PATH,
        lineups_path: Path = STARTING_LINEUPS_PATH,
        rosters_path: Path = WORLD_CUP_ROSTERS_PATH,
    ) -> "PlayerContextStore":
        def read(path: Path) -> pd.DataFrame | None:
            return pd.read_csv(path) if path.exists() else None

        return cls(
            read(ratings_path),
            read(availability_path),
            read(lineups_path),
            read(rosters_path),
        )

    @property
    def available(self) -> bool:
        return not self.ratings.empty

    @property
    def rosters_available(self) -> bool:
        return not self.rosters.empty

    @staticmethod
    def _normalize_ratings(frame: pd.DataFrame | None) -> pd.DataFrame:
        if frame is None or frame.empty:
            return pd.DataFrame(
                columns=[
                    "player_id",
                    "player_name",
                    "national_team",
                    "overall_rating",
                    "position",
                    "rating_date",
                    "source",
                ]
            )
        required = {
            "player_id",
            "player_name",
            "national_team",
            "overall_rating",
            "rating_date",
        }
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"Player ratings missing columns: {sorted(missing)}")
        output = frame.copy()
        output["national_team"] = output["national_team"].map(canonical_team)
        output["overall_rating"] = pd.to_numeric(
            output["overall_rating"], errors="raise"
        )
        output["rating_date"] = pd.to_datetime(output["rating_date"], utc=True)
        output = output.loc[output["overall_rating"].between(1, 100)]
        return output.sort_values("rating_date").drop_duplicates(
            ["player_id", "rating_date"], keep="last"
        )

    @staticmethod
    def _normalize_availability(frame: pd.DataFrame | None) -> pd.DataFrame:
        if frame is None or frame.empty:
            return pd.DataFrame(
                columns=[
                    "fixture_id",
                    "match_date",
                    "team",
                    "player_id",
                    "status",
                    "availability_probability",
                    "reported_at",
                    "source",
                ]
            )
        required = {"match_date", "team", "player_id", "status", "reported_at"}
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"Availability data missing columns: {sorted(missing)}")
        output = frame.copy()
        output["team"] = output["team"].map(canonical_team)
        output["match_date"] = pd.to_datetime(output["match_date"], utc=True)
        output["reported_at"] = pd.to_datetime(output["reported_at"], utc=True)
        status_probability = output["status"].str.casefold().map(STATUS_DEFAULTS)
        supplied = (
            pd.to_numeric(output["availability_probability"], errors="coerce")
            if "availability_probability" in output
            else pd.Series(np.nan, index=output.index)
        )
        output["availability_probability"] = supplied.fillna(status_probability)
        if output["availability_probability"].isna().any():
            unknown = sorted(
                output.loc[
                    output["availability_probability"].isna(), "status"
                ].unique()
            )
            raise ValueError(f"Unknown availability status: {unknown}")
        output["availability_probability"] = output[
            "availability_probability"
        ].clip(0, 1)
        return output

    @staticmethod
    def _normalize_lineups(frame: pd.DataFrame | None) -> pd.DataFrame:
        if frame is None or frame.empty:
            return pd.DataFrame(
                columns=[
                    "fixture_id",
                    "match_date",
                    "team",
                    "player_id",
                    "lineup_status",
                    "announced_at",
                    "source",
                ]
            )
        required = {
            "match_date",
            "team",
            "player_id",
            "lineup_status",
            "announced_at",
        }
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"Lineup data missing columns: {sorted(missing)}")
        output = frame.copy()
        output["team"] = output["team"].map(canonical_team)
        output["match_date"] = pd.to_datetime(output["match_date"], utc=True)
        output["announced_at"] = pd.to_datetime(output["announced_at"], utc=True)
        output["lineup_status"] = output["lineup_status"].str.casefold()
        return output

    @staticmethod
    def _normalize_rosters(frame: pd.DataFrame | None) -> pd.DataFrame:
        columns = [
            "team",
            "squad_number",
            "player_id",
            "player_name",
            "position",
            "overall_rating",
            "rating_imputed",
            "source",
            "retrieved_at",
        ]
        if frame is None or frame.empty:
            return pd.DataFrame(columns=columns)
        required = set(columns).difference({"retrieved_at"})
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"Roster data missing columns: {sorted(missing)}")
        output = frame.copy()
        output["team"] = output["team"].map(canonical_team)
        output["squad_number"] = pd.to_numeric(
            output["squad_number"], errors="raise"
        ).astype(int)
        output["overall_rating"] = pd.to_numeric(
            output["overall_rating"], errors="raise"
        ).astype(float)
        output["rating_imputed"] = output["rating_imputed"].astype(str).str.casefold().isin(
            {"true", "1", "yes"}
        )
        if "retrieved_at" not in output:
            output["retrieved_at"] = pd.NA
        return output[columns].drop_duplicates(
            ["team", "player_id"], keep="last"
        )

    def _ratings_at(self, team: str, cutoff: pd.Timestamp) -> pd.DataFrame:
        candidates = self.ratings.loc[
            (self.ratings["national_team"] == canonical_team(team))
            & (self.ratings["rating_date"] <= cutoff)
        ].copy()
        return candidates.sort_values("rating_date").drop_duplicates(
            "player_id", keep="last"
        )

    def squad_for_match(
        self,
        team: str,
        match_date: pd.Timestamp | str,
        *,
        fixture_id: str | None = None,
        as_of: pd.Timestamp | str | None = None,
    ) -> pd.DataFrame:
        """Return the top-26 rated squad with point-in-time availability."""
        kickoff = pd.to_datetime(match_date, utc=True)
        cutoff = kickoff if as_of is None else min(pd.to_datetime(as_of, utc=True), kickoff)
        registered = self.rosters.loc[
            self.rosters["team"] == canonical_team(team)
        ].copy()
        if not registered.empty:
            ratings = registered.rename(columns={"team": "national_team"})
            ratings["rating_date"] = cutoff
            ratings["roster_verified"] = True
        else:
            candidates = self._ratings_at(team, cutoff)
            if candidates.empty:
                return candidates.assign(
                    availability_probability=pd.Series(dtype=float)
                )
            ratings = candidates.nlargest(26, "overall_rating")
            ratings["rating_imputed"] = False
            ratings["squad_number"] = pd.NA
            ratings["roster_verified"] = False
        if ratings.empty:
            return ratings.assign(availability_probability=pd.Series(dtype=float))

        availability = self.availability.loc[
            (self.availability["team"] == canonical_team(team))
            & (self.availability["match_date"] == kickoff)
            & (self.availability["reported_at"] <= cutoff)
        ].sort_values("reported_at")
        if fixture_id is not None and "fixture_id" in availability:
            exact = availability["fixture_id"].astype(str) == str(fixture_id)
            availability = availability.loc[
                exact | availability["fixture_id"].isna()
            ]
        availability = availability.drop_duplicates("player_id", keep="last")
        squad = ratings.merge(
            availability[
                ["player_id", "availability_probability", "status"]
            ],
            on="player_id",
            how="left",
        )
        squad["availability_reported"] = squad[
            "availability_probability"
        ].notna()
        squad["availability_probability"] = squad[
            "availability_probability"
        ].fillna(1.0).astype(float)
        lineups = self.lineups.loc[
            (self.lineups["team"] == canonical_team(team))
            & (self.lineups["match_date"] == kickoff)
            & (self.lineups["announced_at"] <= cutoff)
        ].sort_values("announced_at")
        if fixture_id is not None and "fixture_id" in lineups:
            exact = lineups["fixture_id"].astype(str) == str(fixture_id)
            lineups = lineups.loc[exact | lineups["fixture_id"].isna()]
        lineups = lineups.drop_duplicates("player_id", keep="last")
        squad = squad.merge(
            lineups[["player_id", "lineup_status"]],
            on="player_id",
            how="left",
        )
        return squad.sort_values(
            ["squad_number", "availability_probability", "overall_rating"],
            ascending=[True, False, False],
            na_position="last",
        ).reset_index(drop=True)

    @staticmethod
    def propose_lineup(squad: pd.DataFrame) -> list[str]:
        """Choose a balanced 4-3-3 from available players, then fill gaps."""
        available = squad.loc[squad["availability_probability"] > 0].copy()
        if len(available) < 11:
            raise ValueError("Fewer than 11 available players in the squad")
        if "lineup_status" in available:
            known = available.loc[
                available["lineup_status"].isin({"starter", "projected_starter"})
            ]
            if len(known) >= 11:
                return known.head(11)["player_id"].astype(str).tolist()

        def group(position: object) -> str:
            value = str(position).upper()
            if value == "GK":
                return "GK"
            if value in {"DF", "CB", "LB", "RB", "LWB", "RWB"}:
                return "DEF"
            if value in {"MF", "CDM", "CM", "CAM", "LM", "RM"}:
                return "MID"
            return "FWD"

        available["_group"] = available["position"].map(group)
        selected: list[str] = []
        for group_name, count in (("GK", 1), ("DEF", 4), ("MID", 3), ("FWD", 3)):
            candidates = available.loc[
                (available["_group"] == group_name)
                & (~available["player_id"].isin(selected))
            ].nlargest(count, "overall_rating")
            selected.extend(candidates["player_id"].astype(str).tolist())
        if len(selected) < 11:
            fill = available.loc[
                ~available["player_id"].isin(selected)
            ].nlargest(11 - len(selected), "overall_rating")
            selected.extend(fill["player_id"].astype(str).tolist())
        return selected[:11]

    def with_confirmed_lineups(
        self,
        home_team: str,
        away_team: str,
        match_date: pd.Timestamp | str,
        home_player_ids: list[str],
        away_player_ids: list[str],
        *,
        fixture_id: str | None = None,
        announced_at: pd.Timestamp | str | None = None,
    ) -> "PlayerContextStore":
        """Return a copy containing user-confirmed starting elevens."""
        kickoff = pd.to_datetime(match_date, utc=True)
        announced = (
            kickoff - pd.Timedelta(minutes=1)
            if announced_at is None
            else min(pd.to_datetime(announced_at, utc=True), kickoff)
        )
        rows = []
        for team, player_ids in (
            (home_team, home_player_ids),
            (away_team, away_player_ids),
        ):
            if len(player_ids) != 11 or len(set(player_ids)) != 11:
                raise ValueError(f"{team} must have exactly 11 unique starters")
            rows.extend(
                {
                    "fixture_id": fixture_id,
                    "match_date": kickoff,
                    "team": canonical_team(team),
                    "player_id": player_id,
                    "lineup_status": "starter",
                    "announced_at": announced,
                    "source": "interactive user selection",
                }
                for player_id in player_ids
            )
        lineups = pd.concat([self.lineups, pd.DataFrame(rows)], ignore_index=True)
        return PlayerContextStore(
            ratings=self.ratings,
            availability=self.availability,
            lineups=lineups,
            rosters=self.rosters,
        )

    def team_context(
        self,
        team: str,
        match_date: pd.Timestamp | str,
        *,
        fixture_id: str | None = None,
        as_of: pd.Timestamp | str | None = None,
    ) -> TeamPlayerContext:
        kickoff = pd.to_datetime(match_date, utc=True)
        cutoff = kickoff if as_of is None else min(pd.to_datetime(as_of, utc=True), kickoff)
        merged = self.squad_for_match(
            team, kickoff, fixture_id=fixture_id, as_of=cutoff
        )
        if merged.empty:
            return _empty_context()

        lineups = self.lineups.loc[
            (self.lineups["team"] == canonical_team(team))
            & (self.lineups["match_date"] == kickoff)
            & (self.lineups["announced_at"] <= cutoff)
        ]
        if fixture_id is not None and "fixture_id" in lineups:
            exact = lineups["fixture_id"].astype(str) == str(fixture_id)
            lineups = lineups.loc[exact | lineups["fixture_id"].isna()]
        starters = lineups.loc[
            lineups["lineup_status"].isin({"starter", "projected_starter"})
        ].drop_duplicates("player_id", keep="last")
        confirmed_starters = starters.loc[
            starters["lineup_status"] == "starter"
        ]
        lineup_confirmed = len(confirmed_starters) >= 11

        if len(starters) >= 11:
            expected = merged.loc[
                merged["player_id"].isin(starters["player_id"])
            ]
        else:
            weighted = merged.assign(
                expected_value=(
                    merged["overall_rating"]
                    * merged["availability_probability"]
                )
            )
            expected = weighted.nlargest(11, "expected_value")

        expected_quality = (
            float(expected["overall_rating"].mean()) if not expected.empty else 0.0
        )
        available_players = merged.loc[merged["availability_probability"] > 0]
        depth = available_players.nlargest(18, "overall_rating")
        squad_depth = float(depth["overall_rating"].mean()) if not depth.empty else 0.0
        unavailable_quality = float(
            (
                merged["overall_rating"]
                * (1.0 - merged["availability_probability"])
            ).sum()
            / 11.0
        )
        return TeamPlayerContext(
            expected_xi_quality=expected_quality,
            squad_depth_quality=squad_depth,
            unavailable_quality=unavailable_quality,
            availability_rate=float(merged["availability_probability"].mean()),
            lineup_confirmed=lineup_confirmed,
            player_data_available=True,
            expected_starters=int(len(expected)),
        )

    def match_context(
        self,
        home_team: str,
        away_team: str,
        match_date: pd.Timestamp | str,
        *,
        fixture_id: str | None = None,
        as_of: pd.Timestamp | str | None = None,
    ) -> MatchPlayerContext:
        return MatchPlayerContext(
            home=self.team_context(
                home_team, match_date, fixture_id=fixture_id, as_of=as_of
            ),
            away=self.team_context(
                away_team, match_date, fixture_id=fixture_id, as_of=as_of
            ),
        )


def convert_fc25_ratings(source: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    fc26 = {"overallRating", "nationality", "position"}.issubset(source.columns)
    fc25 = {"Name", "Nationality", "Overall", "Position"}.issubset(source.columns)
    if not fc26 and not fc25:
        raise ValueError("Unsupported EA player-ratings schema")

    for row in source.to_dict("records"):
        if fc26:
            common_name = row.get("commonName")
            if pd.notna(common_name) and str(common_name).strip():
                name = str(common_name).strip()
            else:
                name = " ".join(
                    part
                    for part in (
                        str(row.get("firstName") or "").strip(),
                        str(row.get("lastName") or "").strip(),
                    )
                    if part
                )
            nationality = canonical_team(str(row["nationality"]))
            club = str(row.get("team") or "").strip()
            overall = int(row["overallRating"])
            position = str(row["position"])
            raw_id = row.get("id")
            player_id = (
                f"ea_{int(raw_id)}"
                if pd.notna(raw_id)
                else hashlib.sha1(
                    f"{name}|{nationality}|{club}".encode("utf-8")
                ).hexdigest()[:16]
            )
        else:
            name = str(row["Name"]).strip()
            nationality = canonical_team(str(row["Nationality"]))
            club = str(row.get("Club") or "").strip()
            overall = int(row["Overall"])
            position = str(row["Position"])
            identity = f"{name}|{nationality}|{club}".encode("utf-8")
            player_id = hashlib.sha1(identity).hexdigest()[:16]
        rows.append(
            {
                "player_id": player_id,
                "player_name": name,
                "national_team": nationality,
                "overall_rating": overall,
                "position": position,
                "rating_date": PLAYER_RATINGS_DATE.isoformat(),
                "source": config.PLAYER_RATINGS_SOURCE,
            }
        )
    return pd.DataFrame(rows).drop_duplicates("player_id").sort_values(
        ["national_team", "overall_rating"], ascending=[True, False]
    )


def refresh_player_ratings(
    *,
    output_path: Path = PLAYER_RATINGS_PATH,
    session: requests.Session | None = None,
) -> pd.DataFrame:
    """Download and convert the CC0 FC 25 player ratings dataset."""
    client = session or requests.Session()
    response = client.get(config.PLAYER_RATINGS_DATASET_URL, timeout=60)
    response.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        csv_files = [name for name in archive.namelist() if name.endswith(".csv")]
        if not csv_files:
            raise ValueError("Player ratings archive contains no CSV file")
        preferred = next(
            (
                name
                for name in csv_files
                if name.casefold().endswith("ea_fc26_players.csv")
            ),
            csv_files[0],
        )
        with archive.open(preferred) as source_file:
            source = pd.read_csv(source_file)
    ratings = convert_fc25_ratings(source)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ratings.to_csv(output_path, index=False)
    return ratings


def _name_key(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    ascii_name = normalized.encode("ascii", "ignore").decode("ascii")
    ascii_name = re.sub(r"\([^)]*\)", " ", ascii_name)
    return re.sub(r"[^a-z0-9]+", " ", ascii_name.casefold()).strip()


def _broad_position(value: str) -> str:
    match = re.search(r"\b(GK|DF|MF|FW)\b", value.upper())
    if not match:
        raise ValueError(f"Unknown roster position: {value}")
    return match.group(1)


def parse_world_cup_rosters_html(
    html: str,
    ratings: pd.DataFrame,
    *,
    retrieved_at: str,
) -> pd.DataFrame:
    """Parse registered squads and join player ratings by team and name."""
    soup = BeautifulSoup(html, "html.parser")
    rows: list[dict[str, object]] = []
    for heading in soup.find_all("h3"):
        team_text = heading.get_text(" ", strip=True)
        table = heading.find_next("table")
        if table is None:
            continue
        table_rows = table.select("tr")
        if not table_rows:
            continue
        headers = [
            cell.get_text(" ", strip=True)
            for cell in table_rows[0].find_all(["th", "td"])
        ]
        if "Player" not in headers or "Pos." not in headers:
            continue
        team = canonical_team(team_text)
        team_ratings = ratings.loc[ratings["national_team"] == team].copy()
        team_ratings["_name_key"] = team_ratings["player_name"].map(_name_key)
        by_name = {
            key: group.iloc[0]
            for key, group in team_ratings.groupby("_name_key")
        }
        available_keys = list(by_name)
        team_rows: list[dict[str, object]] = []
        for table_row in table_rows[1:]:
            cells = [
                cell.get_text(" ", strip=True)
                for cell in table_row.find_all(["th", "td"])
            ]
            if len(cells) < 3 or not cells[0].isdigit():
                continue
            player_name = re.sub(
                r"\s*\(\s*captain\s*\)\s*", "", cells[2], flags=re.IGNORECASE
            ).strip()
            key = _name_key(player_name)
            matched = by_name.get(key)
            if matched is None and available_keys:
                close = difflib.get_close_matches(
                    key, available_keys, n=1, cutoff=0.9
                )
                if close:
                    matched = by_name[close[0]]
            position = _broad_position(cells[1])
            if matched is not None:
                player_id = str(matched["player_id"])
                overall = float(matched["overall_rating"])
                imputed = False
            else:
                player_id = "roster_" + hashlib.sha1(
                    f"{team}|{player_name}".encode("utf-8")
                ).hexdigest()[:16]
                overall = np.nan
                imputed = True
            team_rows.append(
                {
                    "team": team,
                    "squad_number": int(cells[0]),
                    "player_id": player_id,
                    "player_name": player_name,
                    "position": position,
                    "overall_rating": overall,
                    "rating_imputed": imputed,
                    "source": config.WORLD_CUP_SQUADS_URL,
                    "retrieved_at": retrieved_at,
                }
            )
        if not team_rows:
            continue
        matched_values = pd.Series(
            [
                row["overall_rating"]
                for row in team_rows
                if not pd.isna(row["overall_rating"])
            ],
            dtype=float,
        )
        team_default = (
            float(matched_values.median())
            if not matched_values.empty
            else float(team_ratings.nlargest(26, "overall_rating")["overall_rating"].median())
            if not team_ratings.empty
            else 65.0
        )
        for row in team_rows:
            if not pd.isna(row["overall_rating"]):
                continue
            positional = [
                float(candidate["overall_rating"])
                for candidate in team_rows
                if candidate["position"] == row["position"]
                and not pd.isna(candidate["overall_rating"])
            ]
            row["overall_rating"] = (
                float(np.median(positional)) if positional else team_default
            )
        rows.extend(team_rows)
    output = pd.DataFrame(rows)
    if output.empty:
        raise ValueError("No registered World Cup squads found in source")
    return output.sort_values(["team", "squad_number"]).reset_index(drop=True)


def refresh_world_cup_rosters(
    ratings: pd.DataFrame,
    *,
    output_path: Path = WORLD_CUP_ROSTERS_PATH,
    session: requests.Session | None = None,
) -> pd.DataFrame:
    client = session or requests.Session()
    response = client.get(
        config.WORLD_CUP_SQUADS_URL,
        headers={
            "User-Agent": (
                "FIFA-Predict/0.1 "
                "(local portfolio project; registered squad importer)"
            )
        },
        timeout=60,
    )
    response.raise_for_status()
    retrieved_at = pd.Timestamp.now(tz="UTC").isoformat()
    rosters = parse_world_cup_rosters_html(
        response.text, ratings, retrieved_at=retrieved_at
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rosters.to_csv(output_path, index=False)
    return rosters


def adjust_probabilities(
    probabilities: np.ndarray,
    context: MatchPlayerContext,
) -> np.ndarray:
    """Apply a conservative player-strength adjustment to decisive outcomes."""
    if not context.home.player_data_available or not context.away.player_data_available:
        return probabilities.copy()

    # Translate a one-point expected-XI quality edge to 12 Elo points, with
    # smaller contributions from depth and unavailable-player quality.
    player_elo = (
        12.0 * context.quality_difference
        + 4.0 * context.depth_difference
        - 6.0 * context.unavailable_quality_difference
    )
    player_elo = float(np.clip(player_elo, -120.0, 120.0))
    draw = float(probabilities[1])
    decisive_total = max(1.0 - draw, 1e-9)
    baseline_home_share = float(probabilities[0] / decisive_total)
    baseline_logit = np.log(
        np.clip(baseline_home_share, 1e-6, 1 - 1e-6)
        / np.clip(1 - baseline_home_share, 1e-6, 1 - 1e-6)
    )
    adjusted_home_share = 1.0 / (
        1.0 + np.exp(-(baseline_logit + player_elo * np.log(10) / 400.0))
    )
    adjusted = np.array(
        [
            decisive_total * adjusted_home_share,
            draw,
            decisive_total * (1.0 - adjusted_home_share),
        ]
    )
    return adjusted / adjusted.sum()
