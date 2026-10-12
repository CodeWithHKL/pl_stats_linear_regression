# Goal prediction with Poisson regression

Predicts a team's goals in a Premier League match with scikit-learn.

## Setup
    python -m venv .venv
    .venv\Scripts\pip install -r requirements.txt
    .venv\Scripts\python -m src.fetch_data

`fetch_data` downloads 5 seasons (2021/22–2025/26) from football-data.co.uk into `data/pl_matches.csv`
(3,800 team-match rows: team, opponent, home/away, shots, shots on target, corners, goals).

## Models (`src/model.py`)
- **fixture** (pre-match): team + opponent + home/away, one-hot encoded, L2-regularised Poisson. Learns attack/defence strength per team.
- **stats** (post-match): shots, shots on target, corners, home/away. Explains goals; not usable before kickoff.

## Run
    python goal_predictor.py        # cross-validated MSE vs a predict-the-mean baseline
    python -m src.predict fixture --team Arsenal --opponent Chelsea --home
    python -m src.predict stats --shots 15 --on-target 6 --corners 5 --home

Only teams that appeared in the data can be used in `fixture` mode.
