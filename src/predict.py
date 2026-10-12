"""Predict goals.

python -m src.predict fixture --team Arsenal --opponent Chelsea --home
python -m src.predict stats --shots 15 --on-target 6 --corners 5 --home
"""
import argparse

import pandas as pd

from src.model import features_for, fit, load_data


def predict(model, kind, **row):
    X = pd.DataFrame([row])[features_for(kind)]
    return float(model.predict(X)[0])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = parser.add_subparsers(dest="kind", required=True)

    fixture = sub.add_parser("fixture", help="pre-match: expected goals for TEAM against OPPONENT")
    fixture.add_argument("--team", required=True)
    fixture.add_argument("--opponent", required=True)
    fixture.add_argument("--home", action="store_true", help="TEAM plays at home")

    stats = sub.add_parser("stats", help="expected goals from in-match stats")
    stats.add_argument("--shots", type=int, required=True)
    stats.add_argument("--on-target", type=int, required=True)
    stats.add_argument("--corners", type=int, required=True)
    stats.add_argument("--home", action="store_true")

    args = parser.parse_args()
    data = load_data()
    home = int(args.home)

    if args.kind == "fixture":
        known = set(data["TEAM"])
        for name in (args.team, args.opponent):
            if name not in known:
                parser.error(f"unknown team {name!r}; choose from: {', '.join(sorted(known))}")
        row = {"TEAM": args.team, "OPPONENT": args.opponent, "HOME": home}
    else:
        row = {"SHOTS": args.shots, "SHOTS ON TARGET": args.on_target, "CORNERS": args.corners, "HOME": home}

    print(f"Expected goals: {predict(fit(data, args.kind), args.kind, **row):.2f}")


if __name__ == "__main__":
    main()
