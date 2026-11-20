"""Recalculate Table 7 screening areas using a nonspatial score distribution."""
import csv
from pathlib import Path

SAMPLE = Path(__file__).resolve().parents[1] / "sample_data" / "screening_score_bins.csv"


def main():
    totals = {name: [0, 0.0] for name in ("Excluded", "High", "Suitable", "Moderate")}
    with SAMPLE.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            score = int(row["weighted_sum"])
            if score == -1:
                grade = "Excluded"
            elif score / 69 <= 1.75:
                grade = "High"
            elif score / 69 <= 2.75:
                grade = "Suitable"
            else:
                grade = "Moderate"
            totals[grade][0] += int(row["cell_count"])
            totals[grade][1] += float(row["area_km2"])

    def combine(names):
        return tuple(sum(totals[name][index] for name in names) for index in (0, 1))

    summary = {"Stage 1 candidates": combine(totals), **totals,
               "Retained": combine(("High", "Suitable", "Moderate"))}
    print(f"{'Screening group':<22} {'Cells':>10} {'Area (km2)':>12}")
    for name, (count, area) in summary.items():
        print(f"{name:<22} {count:>10,d} {area:>12.2f}")


if __name__ == "__main__":
    main()