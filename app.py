import os
import io
import numpy as np
import pandas as pd
import streamlit as st
import joblib

# =========================
# CONFIG
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# =========================
# Template download
# =========================
TEMPLATE_PATH = os.path.join(BASE_DIR, "Data Template.xlsx")

def get_template_bytes() -> bytes:
    if os.path.exists(TEMPLATE_PATH):
        with open(TEMPLATE_PATH, "rb") as f:
            return f.read()
    return b""
MODELS_DIR = os.path.join(BASE_DIR, "models")
POINT_DIR  = os.path.join(MODELS_DIR, "point_models")
DIST_DIR   = os.path.join(MODELS_DIR, "distribution_models")
TEMP_DIR   = os.path.join(MODELS_DIR, "temporal_models")

MIN_POINTS_FOR_TEMPORAL = 8


# =========================
# Robust datetime parsing
# =========================
def parse_meter_datetime(series: pd.Series) -> pd.Series:
    """
    Handles strings like:
    'Feb 01, 2026, 23:45:00:000000'
    (note the last colon before microseconds)
    """
    s = series.astype(str).str.strip()
    # fix last ":" before microseconds -> "."
    s = s.str.replace(r"(?<=\d{2}:\d{2}:\d{2}):(?=\d{6}$)", ".", regex=True)

    out = pd.to_datetime(s, format="%b %d, %Y, %H:%M:%S.%f", errors="coerce")

    m = out.isna()
    if m.any():
        out.loc[m] = pd.to_datetime(s[m], format="%b %d, %Y, %H:%M:%S", errors="coerce")

    m = out.isna()
    if m.any():
        out.loc[m] = pd.to_datetime(s[m], errors="coerce")

    return out


def safe_div(a, b, eps=1e-6):
    return a / (b + eps)


def normalize_to_0_100(s: pd.Series) -> pd.Series:
    """Robust scaling to 0..100 using 5th/95th percentiles."""
    if len(s) == 0:
        return s
    lo = np.nanpercentile(s, 5)
    hi = np.nanpercentile(s, 95)
    if (hi - lo) < 1e-9:
        return pd.Series(np.zeros(len(s)), index=s.index, dtype=float)
    out = (s - lo) / (hi - lo)
    return np.clip(out, 0, 1) * 100.0


# =========================
# Feature Engineering
# =========================
def compute_point_features(df: pd.DataFrame) -> pd.DataFrame:
    V = df[["V1", "V2", "V3"]].astype(float)
    A = df[["A1", "A2", "A3"]].astype(float)

    X = pd.DataFrame(index=df.index)

    X["V_mean"] = V.mean(axis=1)
    X["V_std"]  = V.std(axis=1)
    X["V_min"]  = V.min(axis=1)
    X["V_max"]  = V.max(axis=1)
    X["V_imb"]  = safe_div((X["V_max"] - X["V_min"]), X["V_mean"])

    X["A_mean"] = A.mean(axis=1)
    X["A_std"]  = A.std(axis=1)
    X["A_min"]  = A.min(axis=1)
    X["A_max"]  = A.max(axis=1)
    X["A_imb"]  = safe_div((X["A_max"] - X["A_min"]), X["A_mean"])

    X["A_phase_ratio_max_min"] = safe_div(X["A_max"], X["A_min"])
    X["S_proxy"] = (V["V1"] * A["A1"]) + (V["V2"] * A["A2"]) + (V["V3"] * A["A3"])

    X["V1_V3"] = safe_div(V["V1"], V["V3"])
    X["A1_A3"] = safe_div(A["A1"], A["A3"])

    # flags
    X["zero_A1"] = (A["A1"] == 0).astype(int)
    X["zero_A2"] = (A["A2"] == 0).astype(int)
    X["zero_A3"] = (A["A3"] == 0).astype(int)
    X["zero_V1"] = (V["V1"] == 0).astype(int)
    X["zero_V2"] = (V["V2"] == 0).astype(int)
    X["zero_V3"] = (V["V3"] == 0).astype(int)

    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return X


def compute_distribution_features(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for meter, g in df.groupby("Meter Number", sort=False):
        V = g[["V1","V2","V3"]].astype(float)
        A = g[["A1","A2","A3"]].astype(float)

        V_mean_row = V.mean(axis=1)
        A_mean_row = A.mean(axis=1)

        V_imb_row = safe_div((V.max(axis=1) - V.min(axis=1)), V_mean_row)
        A_imb_row = safe_div((A.max(axis=1) - A.min(axis=1)), A_mean_row)

        rows.append({
            "Meter Number": meter,
            "n_points": len(g),

            "V_mean": V_mean_row.mean(),
            "V_std":  V_mean_row.std(ddof=0),
            "V_min":  V.min().min(),
            "V_max":  V.max().max(),
            "V_imb_mean": V_imb_row.mean(),
            "V_imb_std":  V_imb_row.std(ddof=0),

            "A_mean": A_mean_row.mean(),
            "A_std":  A_mean_row.std(ddof=0),
            "A_min":  A.min().min(),
            "A_max":  A.max().max(),
            "A_imb_mean": A_imb_row.mean(),
            "A_imb_std":  A_imb_row.std(ddof=0),

            "zeroA_ratio": (A == 0).mean().mean(),
            "zeroV_ratio": (V == 0).mean().mean(),
        })

    Xd = pd.DataFrame(rows).set_index("Meter Number")
    Xd = Xd.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return Xd


def compute_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for meter, g in df.groupby("Meter Number", sort=False):
        if len(g) < MIN_POINTS_FOR_TEMPORAL:
            continue
        g = g.sort_values("Meter Datetime")
        V_series = g[["V1","V2","V3"]].astype(float).mean(axis=1).values
        A_series = g[["A1","A2","A3"]].astype(float).mean(axis=1).values

        dV = np.diff(V_series)
        dA = np.diff(A_series)

        def mad(x):
            if len(x) == 0:
                return 1.0
            med = np.median(x)
            return np.median(np.abs(x - med)) + 1e-6

        spike_A = int(np.sum(np.abs(dA) > 6.0 * mad(dA))) if len(dA) else 0
        spike_V = int(np.sum(np.abs(dV) > 6.0 * mad(dV))) if len(dV) else 0

        t = np.arange(len(A_series))
        slope_A = float(np.polyfit(t, A_series, 1)[0]) if len(t) >= 2 else 0.0
        slope_V = float(np.polyfit(t, V_series, 1)[0]) if len(t) >= 2 else 0.0

        rows.append({
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
        })

    if not rows:
        return pd.DataFrame()
    Xt = pd.DataFrame(rows).set_index("Meter Number")
    Xt = Xt.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return Xt


# =========================
# Explainability (lightweight)
# =========================
def reason_from_features(pf_row: pd.Series) -> str:
    reasons = []
    if pf_row.get("A_imb", 0) >= 0.60:
        reasons.append("عدم اتزان تيار مرتفع")
    if pf_row.get("V_imb", 0) >= 0.08:
        reasons.append("عدم اتزان فولت مرتفع")
    if pf_row.get("A_phase_ratio_max_min", 0) >= 10:
        reasons.append("طور مسيطر/اختلال كبير بين الأطوار")
    if pf_row.get("zero_A2", 0) == 1:
        reasons.append("A2=0 (احتمال CT/طور/حمل)")
    if pf_row.get("S_proxy", 0) < 1:
        reasons.append("حمل شبه معدوم/قراءة غير معتادة")
    if not reasons:
        reasons.append("نمط غير معتاد (Ensemble)")
    return "، ".join(reasons[:3])


def confidence_label(agreement: float, duration_points: int) -> str:
    # agreement: 0..1 (how many models agree it is high)
    if agreement >= 0.75 and duration_points >= 4:
        return "High"
    if agreement >= 0.50 and duration_points >= 2:
        return "Medium"
    return "Low"


# =========================
# Period extraction (no plots)
# =========================
def extract_period_cases(g: pd.DataFrame, meter_thr: float, min_points: int = 2):
    """
    Extract contiguous segments where final_score >= meter_thr.
    Returns list of dicts (cases).
    """
    g = g.sort_values("Meter Datetime").copy()
    high = g["final_score"] >= meter_thr

    cases = []
    start_i = None

    for i, is_high in enumerate(high.values):
        if is_high and start_i is None:
            start_i = i
        elif (not is_high) and start_i is not None:
            seg = g.iloc[start_i:i]
            if len(seg) >= min_points:
                cases.append(seg)
            start_i = None

    if start_i is not None:
        seg = g.iloc[start_i:]
        if len(seg) >= min_points:
            cases.append(seg)

    return cases


# =========================
# Load models
# =========================
@st.cache_resource
def load_models():
    scaler_p = joblib.load(os.path.join(POINT_DIR, "scaler.pkl"))
    iso_point = joblib.load(os.path.join(POINT_DIR, "isolation_forest.pkl"))
    mcd_point = joblib.load(os.path.join(POINT_DIR, "robust_cov_mcd.pkl"))

    scaler_d = joblib.load(os.path.join(DIST_DIR, "scaler.pkl"))
    iso_dist = joblib.load(os.path.join(DIST_DIR, "isolation_forest.pkl"))

    scaler_t = joblib.load(os.path.join(TEMP_DIR, "scaler.pkl"))
    iso_temp = joblib.load(os.path.join(TEMP_DIR, "isolation_forest.pkl"))

    return scaler_p, iso_point, mcd_point, scaler_d, iso_dist, scaler_t, iso_temp


# =========================
# Inference
# =========================
@st.cache_data(show_spinner=False)
def run_inference(file_bytes: bytes,
                  top_percent: float,
                  meter_quantile_for_periods: float,
                  min_period_points: int,
                  compute_periods: bool):
    df = pd.read_excel(io.BytesIO(file_bytes))

    # Validate
    required_cols = ["Meter Number", "Meter Datetime", "Office", "V1","V2","V3","A1","A2","A3"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Parse datetime
    df["Meter Datetime"] = parse_meter_datetime(df["Meter Datetime"])
    df = df.dropna(subset=["Meter Datetime"]).copy()

    # numeric
    for c in ["V1","V2","V3","A1","A2","A3"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["V1","V2","V3","A1","A2","A3"]).copy()

    # sort + dedup
    df = df.sort_values(["Meter Number", "Meter Datetime"]).reset_index(drop=True)
    df = df.drop_duplicates(subset=["Meter Number", "Meter Datetime"], keep="last")

    scaler_p, iso_point, mcd_point, scaler_d, iso_dist, scaler_t, iso_temp = load_models()

    # Point features/scores (row-level)
    pf = compute_point_features(df)
    Xp_scaled = scaler_p.transform(pf)
    df["score_point_iforest"] = -iso_point.decision_function(Xp_scaled)
    df["score_point_mcd"] = mcd_point.mahalanobis(Xp_scaled)

    # Dist features/scores (meter-level)
    Xd = compute_distribution_features(df)
    Xd_scaled = scaler_d.transform(Xd)
    Xd["score_dist_iforest"] = -iso_dist.decision_function(Xd_scaled)
    df = df.merge(Xd[["score_dist_iforest","n_points"]], left_on="Meter Number", right_index=True, how="left")

    # Temporal features/scores (meter-level, optional if enough points)
    Xt = compute_temporal_features(df)
    if len(Xt) > 0:
        Xt_scaled = scaler_t.transform(Xt)
        Xt["score_temp_iforest"] = -iso_temp.decision_function(Xt_scaled)
        df = df.merge(Xt[["score_temp_iforest"]], left_on="Meter Number", right_index=True, how="left")
    else:
        df["score_temp_iforest"] = np.nan

    # Normalize model scores to 0..100
    df["n_p_if"] = normalize_to_0_100(df["score_point_iforest"])
    df["n_p_mcd"] = normalize_to_0_100(df["score_point_mcd"])
    df["n_d_if"] = normalize_to_0_100(df["score_dist_iforest"])
    df["n_t_if"] = normalize_to_0_100(df["score_temp_iforest"].fillna(df["score_temp_iforest"].median()))

    # Dynamic weights
    has_temp = df["score_temp_iforest"].notna().astype(float)
    w_point, w_dist, w_temp = 0.55, 0.30, 0.15

    w_temp_eff = w_temp * has_temp
    w_missing = w_temp * (1.0 - has_temp)
    w_point_eff = w_point + 0.6 * w_missing
    w_dist_eff = w_dist + 0.4 * w_missing

    # Final risk (0..100-ish)
    df["final_score"] = (
        w_point_eff * (0.7 * df["n_p_if"] + 0.3 * df["n_p_mcd"]) +
        w_dist_eff * df["n_d_if"] +
        w_temp_eff * df["n_t_if"]
    )

    # Meter summary
    meters = df.groupby("Meter Number", as_index=False).agg(
        Office=("Office","first"),
        n_points=("Meter Datetime","count"),
        start_time=("Meter Datetime","min"),
        end_time=("Meter Datetime","max"),
        risk_max=("final_score","max"),
        risk_mean=("final_score","mean"),
        A_mean=("A1","mean"),
        V_mean=("V1","mean"),
    ).sort_values("risk_max", ascending=False)

    # Choose global threshold by top_percent
    # e.g. top 1% meters
    k = max(1, int(np.ceil((top_percent / 100.0) * len(meters))))
    top_meters = meters.head(k).copy()
    top_set = set(top_meters["Meter Number"].astype(str).tolist())

    # Build a lightweight "cases table"
    # case = (meter + period/instant) + reason + confidence + suggested visit time
    cases_rows = []

    # Prepare pf merged for reasons (only for top meters to keep it light)
    df_top = df[df["Meter Number"].astype(str).isin(top_set)].copy()
    pf_top = compute_point_features(df_top).add_prefix("pf_")
    df_top = df_top.join(pf_top)

    # agreement proxy: how many model components are "high"
    # (use normalized components)
    df_top["agree_count"] = (
        (df_top["n_p_if"] >= 80).astype(int) +
        (df_top["n_d_if"] >= 80).astype(int) +
        (df_top["n_t_if"] >= 80).astype(int)
    )
    df_top["agreement"] = df_top["agree_count"] / 3.0

    if compute_periods:
        # per meter: threshold based on meter's own score quantile
        for meter, g in df_top.groupby("Meter Number", sort=False):
            g = g.sort_values("Meter Datetime")
            thr = g["final_score"].quantile(meter_quantile_for_periods)

            segs = extract_period_cases(g, meter_thr=thr, min_points=min_period_points)

            if not segs:
                # fallback: single highest timestamp
                top_row = g.sort_values("final_score", ascending=False).iloc[0]
                reason = reason_from_features(top_row.filter(like="pf_").rename(lambda x: x.replace("pf_", "")))
                conf = confidence_label(float(top_row["agreement"]), 1)
                cases_rows.append({
                    "case_type": "Instant",
                    "Meter Number": meter,
                    "Office": top_row["Office"],
                    "risk_%": round(float(top_row["final_score"]), 2),
                    "confidence": conf,
                    "reason": reason,
                    "start_time": top_row["Meter Datetime"],
                    "end_time": top_row["Meter Datetime"],
                    "duration_min": 0,
                    "points": 1,
                    "suggested_visit_time": top_row["Meter Datetime"],
                    "A_mean_case": round(float((top_row["A1"]+top_row["A2"]+top_row["A3"])/3.0), 3),
                    "S_proxy_case": round(float(top_row["pf_S_proxy"]), 3),
                })
                continue

            for seg in segs:
                # pick peak row for reason and suggested visit time
                seg = seg.copy()
                peak = seg.sort_values("final_score", ascending=False).iloc[0]

                reason = reason_from_features(peak.filter(like="pf_").rename(lambda x: x.replace("pf_", "")))
                conf = confidence_label(float(peak["agreement"]), len(seg))

                start = seg["Meter Datetime"].iloc[0]
                end   = seg["Meter Datetime"].iloc[-1]
                duration_min = int((end - start).total_seconds() / 60) if pd.notna(start) and pd.notna(end) else 0

                cases_rows.append({
                    "case_type": "Period",
                    "Meter Number": meter,
                    "Office": peak["Office"],
                    "risk_%": round(float(peak["final_score"]), 2),
                    "confidence": conf,
                    "reason": reason,
                    "start_time": start,
                    "end_time": end,
                    "duration_min": duration_min,
                    "points": int(len(seg)),
                    "suggested_visit_time": peak["Meter Datetime"],
                    "A_mean_case": round(float(seg[["A1","A2","A3"]].mean(axis=1).mean()), 3),
                    "S_proxy_case": round(float(seg["pf_S_proxy"].mean()), 3),
                })
    else:
        # instant-only cases (fastest)
        for meter, g in df_top.groupby("Meter Number", sort=False):
            peak = g.sort_values("final_score", ascending=False).iloc[0]
            reason = reason_from_features(peak.filter(like="pf_").rename(lambda x: x.replace("pf_", "")))
            conf = confidence_label(float(peak["agreement"]), 1)
            cases_rows.append({
                "case_type": "Instant",
                "Meter Number": meter,
                "Office": peak["Office"],
                "risk_%": round(float(peak["final_score"]), 2),
                "confidence": conf,
                "reason": reason,
                "start_time": peak["Meter Datetime"],
                "end_time": peak["Meter Datetime"],
                "duration_min": 0,
                "points": 1,
                "suggested_visit_time": peak["Meter Datetime"],
                "A_mean_case": round(float((peak["A1"]+peak["A2"]+peak["A3"])/3.0), 3),
                "S_proxy_case": round(float(peak["pf_S_proxy"]), 3),
            })

    cases = pd.DataFrame(cases_rows)
    if len(cases) > 0:
        cases = cases.sort_values(["risk_%", "confidence"], ascending=[False, True])

    # For large data: keep a small rows table (top risky timestamps only)
    # (So excel export isn't huge)
    df_top_rows_export = df_top.sort_values("final_score", ascending=False).head(5000)[
        ["Meter Number","Office","Meter Datetime","final_score","V1","V2","V3","A1","A2","A3"]
    ].copy()

    return df, meters, cases, df_top_rows_export


# =========================
# Excel export (single click)
# =========================
def build_excel_bytes(cases: pd.DataFrame, meters: pd.DataFrame, rows_top: pd.DataFrame) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        cases.to_excel(writer, sheet_name="Cases", index=False)
        meters.to_excel(writer, sheet_name="Meters", index=False)
        rows_top.to_excel(writer, sheet_name="TopRows", index=False)
    return output.getvalue()


# =========================
# UI (lightweight & official)
# =========================
st.set_page_config(page_title="NTL Detector - CT Meters", layout="wide")

st.title("NTL Detector – شاشة تشغيل كشف الفاقد المحتمل (عدادات CT)")

# Sidebar settings
st.sidebar.header("إعدادات التشغيل")
top_percent = st.sidebar.slider("نسبة العدادات الأعلى خطورة (Top %)", 0.1, 10.0, 1.0, 0.1)

compute_periods = st.sidebar.checkbox("استخراج فترات عبث (قد يكون أبطأ)", value=True)
meter_quantile_for_periods = st.sidebar.slider("حساسية تحديد الفترة داخل العداد (Quantile)", 0.80, 0.99, 0.95, 0.01)
min_period_points = st.sidebar.slider("أقل عدد نقاط لتكوين فترة", 1, 12, 2, 1)

st.sidebar.markdown("---")
st.sidebar.caption("ملاحظة: إذا الملف كبير جدًا، يمكن إلغاء استخراج الفترات والاكتفاء بأعلى وقت خطورة لكل عداد.")
st.subheader("قالب الإدخال")
st.caption("حمّل القالب، عبّيه بالبيانات المطلوبة، ثم ارفعه هنا للتحليل.")

template_bytes = get_template_bytes()
if template_bytes:
    st.download_button(
        label="📥 تحميل قالب Excel (Data Template)",
        data=template_bytes,
        file_name="Data_Template.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True
    )
else:
    st.warning("لم يتم العثور على ملف القالب داخل المشروع. تأكد أنه موجود باسم: Data Template.xlsx")
uploaded = st.file_uploader("ارفع ملف البيانات (.xlsx)", type=["xlsx"])

if uploaded is None:
    st.info("ارفع ملف Excel يحتوي الأعمدة: Meter Number, Meter Datetime, Office, V1..V3, A1..A3")
    st.stop()

try:
    with st.spinner("جاري التحليل..."):
        file_bytes = uploaded.getvalue()
        rows_scored, meters_ranked, cases_table, rows_top_export = run_inference(
            file_bytes=file_bytes,
            top_percent=top_percent,
            meter_quantile_for_periods=meter_quantile_for_periods,
            min_period_points=min_period_points,
            compute_periods=compute_periods
        )
except Exception as e:
    st.error("حدث خطأ أثناء التحليل.")
    st.exception(e)
    st.stop()

# KPIs
c1, c2, c3, c4 = st.columns(4)
c1.metric("عدد الصفوف", f"{len(rows_scored):,}")
c2.metric("عدد العدادات", f"{meters_ranked['Meter Number'].nunique():,}")
c3.metric("متوسط قراءات/عداد", f"{meters_ranked['n_points'].mean():.1f}")
c4.metric("عدد الحالات (Cases)", f"{len(cases_table):,}")

st.subheader("جدول الحالات الشاذة (Cases) – شامل")
st.caption("كل صف يمثل حالة: (عداد + فترة/وقت + نسبة احتمال + سبب + وقت زيارة مقترح).")

# Filters (light)
colf1, colf2, colf3 = st.columns([2, 1, 1])
with colf1:
    meter_search = st.text_input("بحث برقم العداد (اختياري)", value="")
with colf2:
    conf_filter = st.selectbox("فلتر الثقة", ["All", "High", "Medium", "Low"], index=0)
with colf3:
    min_risk = st.slider("حد أدنى لنسبة الاحتمال (Risk%)", 0, 100, 60, 5)

filtered = cases_table.copy()
if meter_search.strip():
    filtered = filtered[filtered["Meter Number"].astype(str).str.contains(meter_search.strip(), na=False)]
if conf_filter != "All":
    filtered = filtered[filtered["confidence"] == conf_filter]
filtered = filtered[filtered["risk_%"] >= float(min_risk)]

st.dataframe(filtered, use_container_width=True)

# Quick meter ranking (compact)
st.subheader("ملخص العدادات الأعلى خطورة (Top meters)")
st.dataframe(meters_ranked.head(50), use_container_width=True)

# Export
st.subheader("تصدير النتائج")
excel_bytes = build_excel_bytes(filtered, meters_ranked, rows_top_export)
st.download_button(
    label="تحميل النتائج كملف Excel",
    data=excel_bytes,
    file_name="ntl_results.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)

st.caption("Sheets داخل ملف Excel: Cases (الحالات) + Meters (ترتيب العدادات) + TopRows (أعلى قراءات خطورة - مختصرة).")
