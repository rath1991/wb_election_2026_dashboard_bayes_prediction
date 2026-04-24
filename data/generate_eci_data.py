"""
Generates approximate ECI historical data for 294 WB constituencies.
Based on known 2021/2016/2011 WB assembly election results.
Run once: python data/generate_eci_data.py
"""
import csv
import random
import os

random.seed(2026)

# Regional vote share distributions per election
# (tmc_mean, bjp_mean, left_mean) by region
REGIONAL_PROFILES = {
    2021: {
        "north_bengal":       (0.42, 0.46, 0.06),
        "jangalmahal":        (0.40, 0.50, 0.05),
        "medinipur":          (0.48, 0.40, 0.06),
        "urban_kolkata":      (0.54, 0.34, 0.06),
        "south_bengal_rural": (0.54, 0.30, 0.10),
    },
    2016: {
        "north_bengal":       (0.38, 0.10, 0.22),
        "jangalmahal":        (0.35, 0.12, 0.28),
        "medinipur":          (0.42, 0.10, 0.25),
        "urban_kolkata":      (0.47, 0.12, 0.22),
        "south_bengal_rural": (0.48, 0.08, 0.24),
    },
    2011: {
        "north_bengal":       (0.36, 0.06, 0.32),
        "jangalmahal":        (0.28, 0.05, 0.48),
        "medinipur":          (0.38, 0.05, 0.36),
        "urban_kolkata":      (0.44, 0.06, 0.28),
        "south_bengal_rural": (0.44, 0.04, 0.34),
    },
}

def generate_constituency_data(constituency_id, region, year):
    tmc_m, bjp_m, left_m = REGIONAL_PROFILES[year][region]
    noise = 0.07
    tmc = max(0.15, min(0.85, random.gauss(tmc_m, noise)))
    bjp = max(0.02, min(0.75, random.gauss(bjp_m, noise * 0.8)))
    left = max(0.01, min(0.40, random.gauss(left_m, noise * 0.6)))
    total = tmc + bjp + left
    others = max(0.01, 1.0 - total)
    # Renormalize
    grand_total = tmc + bjp + left + others
    tmc /= grand_total
    bjp /= grand_total
    left /= grand_total
    others /= grand_total

    # Determine winner
    scores = {"tmc": tmc, "bjp": bjp, "left": left, "other": others}
    winner = max(scores, key=scores.get)

    return {
        "constituency_id": constituency_id,
        "tmc_voteshare": round(tmc, 4),
        "bjp_voteshare": round(bjp, 4),
        "left_voteshare": round(left, 4),
        "others_voteshare": round(others, 4),
        "winner": winner,
    }


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    constituencies_path = os.path.join(script_dir, "wb_constituencies.csv")

    with open(constituencies_path) as f:
        reader = csv.DictReader(f)
        constituencies = list(reader)

    for year in [2021, 2016, 2011]:
        out_path = os.path.join(script_dir, f"ecidata_{year}.csv")
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "constituency_id", "tmc_voteshare", "bjp_voteshare",
                "left_voteshare", "others_voteshare", "winner"
            ])
            writer.writeheader()
            for c in constituencies:
                row = generate_constituency_data(
                    int(c["constituency_id"]), c["region"], year
                )
                writer.writerow(row)
        print(f"Written {out_path}")

    print("Done. Validate against actual ECI results before using in production.")


if __name__ == "__main__":
    main()
