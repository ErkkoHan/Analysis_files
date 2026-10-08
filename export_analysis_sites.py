import os

import pandas as pd

from features import build_features

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")


# --------------------------------------------------------------------------- #
# Establishments in the analytical sample (input for study_site_map.R)
# --------------------------------------------------------------------------- #
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df, tiers = build_features()
    num = [f for f in tiers["Enhanced"] if f not in ("region", "business", "building")]
    common = df.dropna(subset=num)
    cids = sorted(common["cid"].unique())
    pd.DataFrame({"cid": cids}).to_csv(os.path.join(OUT_DIR, "analysis_sites.csv"), index=False)
    print(f"{len(cids)} establishments, {len(common):,} transitions")


if __name__ == "__main__":
    main()
