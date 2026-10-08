import os

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Data source
# --------------------------------------------------------------------------- #
HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "dataset")
RAW_CSV = os.path.join(DATA_DIR, "full_data.csv")
ESM_XLSX = os.path.join(DATA_DIR, "ESM.xlsx")
DATA = RAW_CSV if os.path.exists(RAW_CSV) else ESM_XLSX
CSV_ENCODINGS = ("utf-8-sig", "utf-8", "big5hkscs", "cp950")

FACILITY_ITEMS = [f"facilities{i}" for i in range(1, 9)]
SANITATION_ITEMS = [f"sanitation{i}" for i in range(1, 10)]

REGION_CODES = {"Taipei": "臺北市", "Taichung": "臺中市", "New Taipei": "新北市",
                "Nantou": "南投縣", "Kaohsiung": "高雄市", "Hsinchu City": "新竹市",
                "Taoyuan": "桃園市", "Tainan": "臺南市", "Other city": "Other city"}
BUSINESS_CODES = {"Conveyor-belt sushi": "迴轉壽司店", "Bakery": "烘焙坊", "Hot-pot": "火鍋店",
                  "Western-style": "西餐", "Café": "咖啡店", "Ramen": "拉麵店", "Other": "其他",
                  "Japanese": "日式", "Chinese": "中式"}
BUILDING_CODES = {"In-mall / department store": "y", "Standalone / street-level": "n"}

_REPORTED = []


def _read_csv(path):
    for enc in CSV_ENCODINGS:
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("utf-8", b"", 0, 1, f"cannot decode {path}")


def _read_esm(path):
    s1 = pd.read_excel(path, sheet_name="Table S1", header=1)
    df = pd.DataFrame({
        "cid": s1["site_id"].str[1:].astype(int),
        "yearmonth": s1["service_period"].str.replace("-", "").astype(int),
        "put_date": pd.to_datetime(s1["deploy_date"]),
        "take_date": pd.to_datetime(s1["collect_date"]),
        "period": s1["trapping_days"],
        "all_monitor": s1["stations_deployed"],
        "catch_monitor": s1["stations_positive"],
        "number": s1["cockroaches_captured"],
        "monitor_rate": s1["positive_rate"],
        "rate": s1["capture_rate"],
        "city": s1["region"].map(REGION_CODES),
        "type": s1["business_type"].map(BUSINESS_CODES),
        "is_department_store": s1["building_type"].map(BUILDING_CODES),
        "period_avg_temp": s1["mean_temp_c"],
        "period_avg_rh": s1["mean_rh_pct"],
        "distance_km": 0.0,
    })
    for c in FACILITY_ITEMS + SANITATION_ITEMS:
        df[c] = s1[c]
    if df[["city", "type", "is_department_store"]].isna().any().any():
        raise ValueError("Unrecognised category label in Table S1")
    return df


def read_data(path=None):
    src = path or DATA
    if not os.path.exists(src):
        raise FileNotFoundError(
            f"Data file not found: {src}\n"
            "Place ESM.xlsx (Online Resource) in the dataset/ folder.")
    df = _read_esm(src) if src.lower().endswith(".xlsx") else _read_csv(src)
    if not _REPORTED:
        _REPORTED.append(src)
        print(f"[data] {src} ({len(df):,} rows)")
    return df


# --------------------------------------------------------------------------- #
# Train/test split
# --------------------------------------------------------------------------- #
SPLIT_YEARMONTH = 202604
APP_ERA_START = 202605
MIN_N = 100


def train_mask(df):
    return (df["yearmonth"] < SPLIT_YEARMONTH).values


def time_split(df):
    m = train_mask(df)
    return df[m].copy(), df[~m].copy()


# --------------------------------------------------------------------------- #
# Feature definitions
# --------------------------------------------------------------------------- #
MONTH_COLS = ["month_sin", "month_cos"]
LAG_COLS = ["density_lag1", "posrate_lag1", "density_lag2", "posrate_lag2"]

ITEM_LABELS = {
    "facilities1": "Holes in wall/floor/ceiling",
    "facilities2": "Peeling wall/ceiling",
    "facilities3": "Eroded floor / standing water",
    "facilities4": "Long-unused items piled up",
    "facilities5": "Poor drainage flow",
    "facilities6": "Organic residue in drains",
    "facilities7": "Excessive door gaps",
    "facilities8": "Equipment obstructs control/inspection",
    "sanitation1": "Food/waste on worktops",
    "sanitation2": "Food/waste on kitchen floor",
    "sanitation3": "Food/waste inside machines",
    "sanitation4": "Food/waste on dining-area floor",
    "sanitation5": "Poorly stored storeroom items",
    "sanitation6": "Dropped food/waste in receiving area",
    "sanitation7": "Dropped food/waste in staff areas",
    "sanitation8": "Food/waste at waste-disposal site",
    "sanitation9": "Refuse dumped around premises",
}


def feature_groups(feats):
    names, cols, done_month = [], [], False
    mcols = [c for c in feats if c in MONTH_COLS]
    for f in feats:
        if f in mcols:
            if not done_month:
                names.append("month")
                cols.append(list(mcols))
                done_month = True
            continue
        names.append(f)
        cols.append([f])
    return names, cols


def add_sequence(df):
    df["put"] = pd.to_datetime(df["put_date"], errors="coerce")
    df["take"] = pd.to_datetime(df["take_date"], errors="coerce")
    df = df.sort_values(["cid", "put"]).reset_index(drop=True)
    df["slot"] = df.groupby(["cid", "yearmonth"])["take"].rank(method="first").astype(int)
    if (df["slot"] > 2).any():
        raise ValueError("More than two visits in one service period")
    df["seq"] = ((df["yearmonth"] // 100) * 12 + (df["yearmonth"] % 100)) * 2 + (df["slot"] - 1)
    df = df.sort_values(["cid", "seq"]).reset_index(drop=True)
    df["month"] = df["yearmonth"] % 100
    ang = 2 * np.pi * df["month"] / 12
    df["month_sin"] = np.sin(ang)
    df["month_cos"] = np.cos(ang)
    df = df[df["distance_km"] <= 100].copy()
    reg_cnt = df["region"].value_counts()
    small = reg_cnt[reg_cnt < MIN_N].index.tolist()
    df["region"] = df["region"].where(~df["region"].isin(small), "Other city")
    df["building"] = df["is_department_store"]
    return df


def add_lags(df, g):
    for L in (1, 2):
        ok = (df["seq"] - g["seq"].shift(L)) == L
        df[f"density_lag{L}"] = g["density"].shift(L).where(ok)
        df[f"posrate_lag{L}"] = g["distribution"].shift(L).where(ok)
    return df


def load_monitoring():
    df = read_data().rename(columns={
        "rate": "density", "monitor_rate": "distribution",
        "city": "region", "type": "business",
        "period_avg_temp": "temp", "period_avg_rh": "humidity",
    })
    return add_sequence(df)


def build_features():
    df = load_monitoring()
    g = df.groupby("cid", group_keys=False)
    df["step_next"] = g["seq"].shift(-1) - df["seq"]
    df["y_density"] = g["density"].shift(-1)
    df["y_posrate"] = g["distribution"].shift(-1)
    df["y_number"] = g["number"].shift(-1)
    df["y_catch_n"] = g["catch_monitor"].shift(-1)
    df["y_all_n"] = g["all_monitor"].shift(-1)
    df = add_lags(df, g)
    df = df[df["step_next"] == 1].copy()

    base = (["density", "distribution"] + MONTH_COLS
            + ["region", "business", "building", "temp", "humidity"])
    tiers = {"Baseline": base, "Enhanced": base + LAG_COLS}
    return df, tiers


if __name__ == "__main__":
    df, tiers = build_features()
    complete = df.dropna(subset=[f for f in tiers["Enhanced"]
                                 if f not in ("region", "business", "building")])
    tr, te = time_split(complete)
    print(f"transitions: {len(df):,} ({df['cid'].nunique()} sites); "
          f"complete Enhanced: {len(complete):,} ({complete['cid'].nunique()} sites)")
    print(f"train {len(tr):,} / test {len(te):,}")
    for name, feats in tiers.items():
        print(f"  {name}: {len(feature_groups(feats)[0])} predictors / {len(feats)} columns")
