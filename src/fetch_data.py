"""Download Premier League match results from football-data.co.uk and reshape
them into one row per team per match (two rows per match).

Usage: python -m src.fetch_data
"""
import io
import urllib.request

import pandas as pd

from src.model import MATCH_DATA_PATH

SEASONS = ["2122", "2223", "2324", "2425", "2526"]
URL = "https://www.football-data.co.uk/mmz4281/{season}/E0.csv"


def fetch_season(season):
    req = urllib.request.Request(URL.format(season=season), headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        raw = resp.read().decode("utf-8-sig", errors="replace")
    df = pd.read_csv(io.StringIO(raw))
    df = df.dropna(subset=["HomeTeam", "FTHG", "FTAG", "HS", "HST"]).copy()
    df["SEASON"] = f"20{season[:2]}/{season[2:]}"
    return df


def to_team_rows(df):
    home = pd.DataFrame({
        "SEASON": df["SEASON"], "DATE": df["Date"],
        "TEAM": df["HomeTeam"], "OPPONENT": df["AwayTeam"], "HOME": 1,
        "SHOTS": df["HS"], "SHOTS ON TARGET": df["HST"], "CORNERS": df["HC"],
        "GOALS": df["FTHG"],
    })
    away = pd.DataFrame({
        "SEASON": df["SEASON"], "DATE": df["Date"],
        "TEAM": df["AwayTeam"], "OPPONENT": df["HomeTeam"], "HOME": 0,
        "SHOTS": df["AS"], "SHOTS ON TARGET": df["AST"], "CORNERS": df["AC"],
        "GOALS": df["FTAG"],
    })
    return pd.concat([home, away], ignore_index=True)


def main():
    frames = []
    for season in SEASONS:
        df = fetch_season(season)
        print(f"{season}: {len(df)} matches")
        frames.append(df)
    rows = to_team_rows(pd.concat(frames, ignore_index=True))
    rows = rows.dropna().astype({"GOALS": int, "SHOTS": int, "SHOTS ON TARGET": int, "CORNERS": int})
    MATCH_DATA_PATH.parent.mkdir(exist_ok=True)
    rows.to_csv(MATCH_DATA_PATH, index=False)
    print(f"Wrote {len(rows)} team-match rows to {MATCH_DATA_PATH}")


if __name__ == "__main__":
    main()
