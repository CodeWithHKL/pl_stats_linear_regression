"""Evaluate the goal-prediction models on data/pl_matches.csv (run `python -m src.fetch_data` first)."""
from src.model import cross_validate, load_data


def main():
    data = load_data()
    print(f"{len(data)} team-match rows, {data['SEASON'].nunique()} seasons")
    for kind in ("stats", "fixture"):
        cv = cross_validate(data, kind)
        print(f"[{kind}] 5-fold CV MSE {cv['model']:.3f}  (mean baseline {cv['baseline']:.3f})")


if __name__ == "__main__":
    main()
