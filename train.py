import os
import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import IsolationForest
from sklearn.covariance import MinCovDet
from sklearn.preprocessing import RobustScaler


# =========================
# Config
# =========================
BASE_DIR = r"C:\Users\78598\Documents\anom"
DATA_PATH = os.path.join(BASE_DIR,"data1.xlsx")

MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

os.makedirs(os.path.join(MODELS_DIR, "point_models"), exist_ok=True)
os.makedirs(os.path.join(MODELS_DIR, "distribution_models"), exist_ok=True)
os.makedirs(os.path.join(MODELS_DIR, "temporal_models"), exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

RANDOM_STATE = 42

# thresholds for temporal scoring availability
MIN_POINTS_FOR_TEMPORAL = 8  # if a meter has >= this count, we compute temporal features
CONTAMINATION = 0.01         # expected fraction of anomalies (tune later)


# =========================
# Robust datetime parser
# =========================
def parse_meter_datetime(series: pd.Series) -> pd.Series:
    """
    Parse datetime strings like:
    'Feb 01, 2026, 23:45:00:000000'
    and other close variants.

    Returns datetime64[ns], invalid -> NaT
    """
    s = series.astype(str).str.strip()

    # Fix the last ":" before microseconds -> "."
    # 23:45:00:000000 -> 23:45:00.000000
    s = s.str.replace(r"(?<=\d{2}:\d{2}:\d{2}):(?=\d{6}$)", ".", regex=True)

    # Fast strict parse
    out = pd.to_datetime(s, format="%b %d, %Y, %H:%M:%S.%f", errors="coerce")

    # Fallback: no microseconds
    mask = out.isna()
    if mask.any():
        out2 = pd.to_datetime(s[mask], format="%b %d, %Y, %H:%M:%S", errors="coerce")
        out.loc[mask] = out2

    # Last resort (slow) for unexpected strings
    mask = out.isna()
    if mask.any():
        out3 = pd.to_datetime(s[mask], errors="coerce")
        out.loc[mask] = out3

    return out


# =========================
# Feature Engineering
# =========================
def safe_div(a, b, eps=1e-6):
    return a / (b + eps)

def compute_point_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Point-based features per row.
    Works with single reading.
    """
    V = df[["V1", "V2", "V3"]].astype(float)
    A = df[["A1", "A2", "A3"]].astype(float)

    X = pd.DataFrame(index=df.index)
    X["V_mean"] = V.mean(axis=1)
    X["V_std"] = V.std(axis=1)
    X["V_min"] = V.min(axis=1)
    X["V_max"] = V.max(axis=1)
    X["V_imb"] = safe_div((X["V_max"] - X["V_min"]), X["V_mean"])

    X["A_mean"] = A.mean(axis=1)
    X["A_std"] = A.std(axis=1)
    X["A_min"] = A.min(axis=1)
    X["A_max"] = A.max(axis=1)
    X["A_imb"] = safe_div((X["A_max"] - X["A_min"]), X["A_mean"])

    # phase dominance ratio
    X["A_phase_ratio_max_min"] = safe_div(X["A_max"], X["A_min"])

    # approximate apparent power proxy (no PF)
    X["S_proxy"] = (V["V1"] * A["A1"]) + (V["V2"] * A["A2"]) + (V["V3"] * A["A3"])

    # ratios between phases (robust to scale)
    X["V1_V3"] = safe_div(V["V1"], V["V3"])
    X["A1_A3"] = safe_div(A["A1"], A["A3"])

    # zero flags (helpful for CT / phase issues)
    X["zero_A1"] = (A["A1"] == 0).astype(int)
    X["zero_A2"] = (A["A2"] == 0).astype(int)
    X["zero_A3"] = (A["A3"] == 0).astype(int)
    X["zero_V1"] = (V["V1"] == 0).astype(int)
    X["zero_V2"] = (V["V2"] == 0).astype(int)
    X["zero_V3"] = (V["V3"] == 0).astype(int)

    # Replace inf/NaN
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return X


def compute_distribution_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Distribution features per meter (ignores order).
    Works with any count >= 1.
    Output indexed by Meter Number.
    """
    rows = []
    for meter, g in df.groupby("Meter Number", sort=False):
        V = g[["V1", "V2", "V3"]].astype(float)
        A = g[["A1", "A2", "A3"]].astype(float)

        V_mean_row = V.mean(axis=1)
        A_mean_row = A.mean(axis=1)

        V_imb_row = safe_div((V.max(axis=1) - V.min(axis=1)), V_mean_row)
        A_imb_row = safe_div((A.max(axis=1) - A.min(axis=1)), A_mean_row)

        row = {
            "Meter Number": meter,
            "n_points": len(g),

            "V_mean": V_mean_row.mean(),
            "V_std": V_mean_row.std(ddof=0),
            "V_min": V.min().min(),
            "V_max": V.max().max(),
            "V_imb_mean": V_imb_row.mean(),
            "V_imb_std": V_imb_row.std(ddof=0),

            "A_mean": A_mean_row.mean(),
            "A_std": A_mean_row.std(ddof=0),
            "A_min": A.min().min(),
            "A_max": A.max().max(),
            "A_imb_mean": A_imb_row.mean(),
            "A_imb_std": A_imb_row.std(ddof=0),

            "zeroA_ratio": (A == 0).mean().mean(),
            "zeroV_ratio": (V == 0).mean().mean(),
        }
        rows.append(row)

    Xd = pd.DataFrame(rows).set_index("Meter Number")
    Xd = Xd.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return Xd


def compute_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Temporal features per meter using ordered time series.
    Only meaningful when meter has multiple points.
    Output indexed by Meter Number.
    """
    rows = []
    for meter, g in df.groupby("Meter Number", sort=False):
        if len(g) < MIN_POINTS_FOR_TEMPORAL:
            continue

        g = g.sort_values("Meter Datetime")
        V = g[["V1", "V2", "V3"]].astype(float)
        A = g[["A1", "A2", "A3"]].astype(float)

        # Use mean across phases as main series
        V_series = V.mean(axis=1).values
        A_series = A.mean(axis=1).values

        # First differences
        dV = np.diff(V_series)
        dA = np.diff(A_series)

        # Spike counts using robust thresholds
        # (median absolute deviation approach)
        def mad(x):
            med = np.median(x)
            return np.median(np.abs(x - med)) + 1e-6

        dA_mad = mad(dA) if len(dA) > 0 else 1.0
        dV_mad = mad(dV) if len(dV) > 0 else 1.0

        spike_A = int(np.sum(np.abs(dA) > 6.0 * dA_mad))
        spike_V = int(np.sum(np.abs(dV) > 6.0 * dV_mad))

        # Trend (slope) via simple linear fit on index
        t = np.arange(len(A_series))
        if len(t) >= 2:
            slope_A = float(np.polyfit(t, A_series, 1)[0])
            slope_V = float(np.polyfit(t, V_series, 1)[0])
        else:
            slope_A = 0.0
            slope_V = 0.0

        row = {
            "Meter Number": meter,
            "n_points": len(g),

            "dA_std": float(np.std(dA)) if len(dA) else 0.0,
            "dA_max": float(np.max(np.abs(dA))) if len(dA) else 0.0,
            "dV_std": float(np.std(dV)) if len(dV) else 0.0,
            "dV_max": float(np.max(np.abs(dV))) if len(dV) else 0.0,

            "spike_A": spike_A,
            "spike_V": spike_V,

            "slope_A": slope_A,
            "slope_V": slope_V,
        }
        rows.append(row)

    if not rows:
        # no meters have enough points
        return pd.DataFrame()

    Xt = pd.DataFrame(rows).set_index("Meter Number")
    Xt = Xt.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return Xt


# =========================
# Training & Scoring Helpers
# =========================
def fit_and_score_isoforest(X: np.ndarray, contamination=CONTAMINATION, random_state=RANDOM_STATE):
    model = IsolationForest(
        n_estimators=300,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1
    )
    model.fit(X)
    # Higher score = more anomalous
    score = -model.decision_function(X)
    return model, score

def fit_and_score_mcd(X: np.ndarray):
    # Robust covariance / Mahalanobis distance (higher = more outlier)
    mcd = MinCovDet().fit(X)
    score = mcd.mahalanobis(X)
    return mcd, score

def normalize_scores(s: pd.Series) -> pd.Series:
    # Robust normalization to 0..100 using percentiles (stable with outliers)
    lo = np.nanpercentile(s, 5)
    hi = np.nanpercentile(s, 95)
    if hi - lo < 1e-9:
        return pd.Series(np.zeros(len(s)), index=s.index, dtype=float)
    out = (s - lo) / (hi - lo)
    out = np.clip(out, 0, 1) * 100.0
    return out


# =========================
# Main
# =========================
def main():
    # Load
    df = pd.read_excel(DATA_PATH)
    required_cols = ["Meter Number", "Meter Datetime", "Office", "V1", "V2", "V3", "A1", "A2", "A3"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Parse datetime automatically
    df["Meter Datetime"] = parse_meter_datetime(df["Meter Datetime"])
    bad = df["Meter Datetime"].isna().sum()
    if bad > 0:
        print(f"⚠️ Warning: {bad} rows have invalid datetime and will be dropped.")
        df = df.dropna(subset=["Meter Datetime"]).copy()

    # Sort after parsing (IMPORTANT)
    df = df.sort_values(["Meter Number", "Meter Datetime"]).reset_index(drop=True)
    df = df.drop_duplicates(subset=["Meter Number", "Meter Datetime"], keep="last")

    # Ensure numeric columns
    for c in ["V1", "V2", "V3", "A1", "A2", "A3"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["V1", "V2", "V3", "A1", "A2", "A3"]).copy()

    print(f"✅ Loaded rows: {len(df):,}")
    print(f"✅ Unique meters: {df['Meter Number'].nunique():,}")

    # -------------------------
    # POINT FEATURES (per row)
    # -------------------------
    Xp = compute_point_features(df)

    scaler_p = RobustScaler()
    Xp_scaled = scaler_p.fit_transform(Xp)

    joblib.dump(scaler_p, os.path.join(MODELS_DIR, "point_models", "scaler.pkl"))

    # Train point models
    iso_point, s_if_point = fit_and_score_isoforest(Xp_scaled)
    joblib.dump(iso_point, os.path.join(MODELS_DIR, "point_models", "isolation_forest.pkl"))

    mcd_point, s_mcd_point = fit_and_score_mcd(Xp_scaled)
    joblib.dump(mcd_point, os.path.join(MODELS_DIR, "point_models", "robust_cov_mcd.pkl"))

    df["score_point_iforest"] = s_if_point
    df["score_point_mcd"] = s_mcd_point

    # -------------------------
    # DISTRIBUTION FEATURES (per meter)
    # -------------------------
    Xd = compute_distribution_features(df)

    scaler_d = RobustScaler()
    Xd_scaled = scaler_d.fit_transform(Xd)

    joblib.dump(scaler_d, os.path.join(MODELS_DIR, "distribution_models", "scaler.pkl"))

    iso_dist, s_if_dist = fit_and_score_isoforest(Xd_scaled)
    joblib.dump(iso_dist, os.path.join(MODELS_DIR, "distribution_models", "isolation_forest.pkl"))

    Xd["score_dist_iforest"] = s_if_dist

    # Merge dist score back to rows
    df = df.merge(
        Xd[["score_dist_iforest", "n_points"]],
        left_on="Meter Number",
        right_index=True,
        how="left"
    )

    # -------------------------
    # TEMPORAL FEATURES (per meter, optional)
    # -------------------------
    Xt = compute_temporal_features(df)
    if len(Xt) > 0:
        scaler_t = RobustScaler()
        Xt_scaled = scaler_t.fit_transform(Xt)
        joblib.dump(scaler_t, os.path.join(MODELS_DIR, "temporal_models", "scaler.pkl"))

        iso_temp, s_if_temp = fit_and_score_isoforest(Xt_scaled)
        joblib.dump(iso_temp, os.path.join(MODELS_DIR, "temporal_models", "isolation_forest.pkl"))

        Xt["score_temp_iforest"] = s_if_temp

        # Merge temporal score to rows
        df = df.merge(
            Xt[["score_temp_iforest"]],
            left_on="Meter Number",
            right_index=True,
            how="left"
        )
    else:
        df["score_temp_iforest"] = np.nan

    # -------------------------
    # SCORE NORMALIZATION + DYNAMIC WEIGHTS
    # -------------------------
    # Normalize to 0..100
    df["n_score_point_iforest"] = normalize_scores(df["score_point_iforest"])
    df["n_score_point_mcd"] = normalize_scores(df["score_point_mcd"])
    df["n_score_dist_iforest"] = normalize_scores(df["score_dist_iforest"])
    df["n_score_temp_iforest"] = normalize_scores(df["score_temp_iforest"].fillna(df["score_temp_iforest"].median()))

    # Dynamic weights:
    # - Always use point scores (works even with single row)
    # - Use dist always (works even with 1 row per meter, but becomes more useful with more)
    # - Use temporal only when available
    has_temp = df["score_temp_iforest"].notna().astype(float)

    # Base weights
    w_point = 0.55
    w_dist = 0.30
    w_temp = 0.15

    # If temporal not available, distribute its weight into point+dist
    w_temp_eff = w_temp * has_temp
    w_missing = w_temp * (1.0 - has_temp)

    w_point_eff = w_point + 0.6 * w_missing
    w_dist_eff = w_dist + 0.4 * w_missing

    # final score per row
    df["final_score"] = (
        w_point_eff * (0.7 * df["n_score_point_iforest"] + 0.3 * df["n_score_point_mcd"])
        + w_dist_eff * df["n_score_dist_iforest"]
        + w_temp_eff * df["n_score_temp_iforest"]
    )

    # -------------------------
    # SAVE RESULTS
    # -------------------------
    # Row-level results (keeps single readings)
    out_rows = df.copy()
    out_rows.sort_values("final_score", ascending=False).to_csv(
        os.path.join(RESULTS_DIR, "scores_rows.csv"), index=False, encoding="utf-8-sig"
    )

    # Meter-level summary (use max/mean score)
    meter_summary = df.groupby("Meter Number", as_index=False).agg(
        Office=("Office", "first"),
        n_points=("Meter Datetime", "count"),
        start_time=("Meter Datetime", "min"),
        end_time=("Meter Datetime", "max"),
        final_score_max=("final_score", "max"),
        final_score_mean=("final_score", "mean"),
        point_iforest_max=("n_score_point_iforest", "max"),
        dist_iforest=("n_score_dist_iforest", "max"),
        temp_iforest=("n_score_temp_iforest", "max"),
    )
    meter_summary = meter_summary.sort_values("final_score_max", ascending=False)
    meter_summary.to_csv(
        os.path.join(RESULTS_DIR, "scores_meters.csv"), index=False, encoding="utf-8-sig"
    )

    # Metrics report (unsupervised)
    top_1pct = int(max(1, 0.01 * len(meter_summary)))
    suspicious = meter_summary.head(top_1pct)

    with open(os.path.join(RESULTS_DIR, "metrics.txt"), "w", encoding="utf-8") as f:
        f.write("Training Summary (Unsupervised)\n")
        f.write("==============================\n")
        f.write(f"Rows: {len(df):,}\n")
        f.write(f"Unique meters: {df['Meter Number'].nunique():,}\n")
        f.write(f"Temporal enabled meters (>= {MIN_POINTS_FOR_TEMPORAL} points): {int(df['score_temp_iforest'].notna().groupby(df['Meter Number']).max().sum())}\n")
        f.write(f"Point model contamination: {CONTAMINATION}\n")
        f.write(f"Top 1% meters flagged count: {len(suspicious):,}\n")
        f.write("\nTop 10 meters by max score:\n")
        for _, r in meter_summary.head(10).iterrows():
            f.write(f"- {r['Meter Number']}: max={r['final_score_max']:.2f}, mean={r['final_score_mean']:.2f}, n={int(r['n_points'])}\n")

    print("✅ Training completed successfully.")
    print(f"📁 Models saved to: {MODELS_DIR}")
    print(f"📊 Results saved to: {RESULTS_DIR}")
    print("   - scores_rows.csv (row-level scores)")
    print("   - scores_meters.csv (meter-level ranking)")
    print("   - metrics.txt")


if __name__ == "__main__":
    main()