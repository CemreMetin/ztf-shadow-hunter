"""Feature extraction and compact light-curve packing for ZTF alerts."""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

MAX_LC_POINTS = 64

# Coarse taxonomy mapping to compare stamp_classifier vs lc_classifier fairly
COARSE_GROUP = {
    # stamp_classifier classes
    "SN": "TRANSIENT",
    "AGN": "AGN",
    "VS": "VARIABLE",
    "asteroid": "SOLAR_SYSTEM",
    "bogus": "BOGUS",
    # lc_classifier classes
    "SNIa": "TRANSIENT",
    "SNIbc": "TRANSIENT",
    "SNII": "TRANSIENT",
    "SLSN": "TRANSIENT",
    "QSO": "AGN",
    "Blazar": "AGN",
    "CV/Nova": "VARIABLE",
    "YSO": "VARIABLE",
    "LPV": "VARIABLE",
    "E": "VARIABLE",
    "DSCT": "VARIABLE",
    "RRL": "VARIABLE",
    "CEP": "VARIABLE",
    "Periodic-Other": "VARIABLE",
}

FEATURE_NAMES = [
    "neg_frac",
    "amp_mag",
    "rise_slope",
    "fade_slope",
    "g_minus_r",
    "color_drift",
    "lc_entropy",
    "ai_disagreement",
    "timespan_days",
    "rb_median",
]


def normalized_entropy(probs: List[float]) -> float:
    arr = np.asarray([p for p in probs if p > 0], dtype=np.float64)
    if arr.size <= 1:
        return 0.0
    arr = arr / arr.sum()
    h = -float(np.sum(arr * np.log(arr)))
    return float(np.clip(h / np.log(arr.size), 0.0, 1.0))


def parse_classifier_summary(prob_df: Optional[pd.DataFrame]) -> Dict[str, Any]:
    """Extract top class, top probability, entropy and cross-classifier disagreement."""
    out = {
        "stamp_top_class": "UNKNOWN",
        "stamp_top_prob": 0.0,
        "lc_top_class": "UNKNOWN",
        "lc_top_prob": 0.0,
        "lc_entropy": 0.0,
        "ai_disagreement": 0.0,
    }
    if prob_df is None or prob_df.empty:
        return out

    # Stamp classifier
    stamp_rows = prob_df[prob_df["classifier_name"] == "stamp_classifier"]
    if not stamp_rows.empty:
        best = stamp_rows.sort_values("probability", ascending=False).iloc[0]
        out["stamp_top_class"] = str(best["class_name"])
        out["stamp_top_prob"] = float(best["probability"])

    # Light-curve classifier
    lc_rows = prob_df[prob_df["classifier_name"] == "lc_classifier"]
    if not lc_rows.empty:
        lc_sorted = lc_rows.sort_values("probability", ascending=False)
        best = lc_sorted.iloc[0]
        out["lc_top_class"] = str(best["class_name"])
        out["lc_top_prob"] = float(best["probability"])
        out["lc_entropy"] = normalized_entropy(lc_sorted["probability"].astype(float).tolist())

    # Cross-model disagreement (image AI vs light-curve AI)
    g_stamp = COARSE_GROUP.get(out["stamp_top_class"], out["stamp_top_class"])
    g_lc = COARSE_GROUP.get(out["lc_top_class"], out["lc_top_class"])
    if g_stamp != "UNKNOWN" and g_lc != "UNKNOWN":
        if g_stamp != g_lc:
            # Stronger conflict when both models are fairly confident in different things
            out["ai_disagreement"] = float(
                np.clip(0.5 + 0.5 * min(out["stamp_top_prob"], out["lc_top_prob"]), 0.0, 1.0)
            )
        else:
            out["ai_disagreement"] = float(abs(out["stamp_top_prob"] - out["lc_top_prob"]) * 0.4)

    return out


def compute_color_series(df: pd.DataFrame, max_pair_dt_days: float = 2.0) -> Tuple[float, float]:
    """Compute median (g - r) color and its drift (late minus early) from paired g/r points."""
    g_df = df[df["fid"] == 1][["mjd", "magpsf"]].sort_values("mjd")
    r_df = df[df["fid"] == 2][["mjd", "magpsf"]].sort_values("mjd")
    if g_df.empty or r_df.empty:
        return 0.0, 0.0

    r_mjd = r_df["mjd"].to_numpy(dtype=np.float64)
    r_mag = r_df["magpsf"].to_numpy(dtype=np.float64)

    pairs: List[Tuple[float, float]] = []
    for _, row in g_df.iterrows():
        t = float(row["mjd"])
        idx = int(np.argmin(np.abs(r_mjd - t)))
        if abs(r_mjd[idx] - t) <= max_pair_dt_days:
            pairs.append((t, float(row["magpsf"]) - float(r_mag[idx])))

    if not pairs:
        # Fallback to overall median difference if bands aren't tightly paired
        return float(g_df["magpsf"].median() - r_df["magpsf"].median()), 0.0

    colors = np.array([c for _, c in pairs], dtype=np.float64)
    med_color = float(np.median(colors))
    if len(pairs) >= 3:
        half = len(colors) // 2
        drift = float(np.median(colors[half:]) - np.median(colors[:half]))
    else:
        drift = 0.0
    return med_color, drift


def extract_features_and_pack(
    det_df: pd.DataFrame,
    prob_df: Optional[pd.DataFrame],
) -> Optional[Dict[str, Any]]:
    """Compute 10 physical/AI features and pack light curve into fixed-size arrays."""
    if det_df is None or det_df.empty:
        return None

    needed = {"mjd", "magpsf", "fid"}
    if not needed.issubset(det_df.columns):
        return None

    df = det_df.dropna(subset=["mjd", "magpsf", "fid"]).copy()
    if len(df) < 3:
        return None

    df = df.sort_values("mjd").reset_index(drop=True)
    if "sigmapsf" not in df.columns:
        df["sigmapsf"] = 0.05
    if "isdiffpos" not in df.columns:
        df["isdiffpos"] = 1
    if "rb" not in df.columns:
        df["rb"] = 0.8

    # Normalize isdiffpos: ALeRCE returns 1 / -1 (or 't' / 'f' in some schemas)
    def _norm_sign(v: Any) -> int:
        if isinstance(v, str):
            return -1 if v.strip().lower() in ("-1", "f", "false", "0") else 1
        try:
            return -1 if int(v) <= 0 else 1
        except Exception:
            return 1

    signs = df["isdiffpos"].map(_norm_sign).to_numpy(dtype=np.int8)
    mjd = df["mjd"].to_numpy(dtype=np.float64)
    mag = df["magpsf"].to_numpy(dtype=np.float64)
    mag_err = df["sigmapsf"].fillna(0.05).to_numpy(dtype=np.float64)
    fid = df["fid"].to_numpy(dtype=np.int8)
    rb = df["rb"].fillna(0.75).to_numpy(dtype=np.float64)

    neg_frac = float(np.mean(signs == -1))
    amp_mag = float(np.max(mag) - np.min(mag))
    timespan_days = float(mjd[-1] - mjd[0])
    rb_median = float(np.median(rb))

    # Rise and fade slopes around brightest point (min mag)
    idx_bright = int(np.argmin(mag))
    dt_rise = float(mjd[idx_bright] - mjd[0])
    rise_slope = float((mag[idx_bright] - mag[0]) / dt_rise) if dt_rise > 0.05 else 0.0
    dt_fade = float(mjd[-1] - mjd[idx_bright])
    fade_slope = float((mag[-1] - mag[idx_bright]) / dt_fade) if dt_fade > 0.05 else 0.0

    # Clip extreme slopes caused by noisy adjacent points
    rise_slope = float(np.clip(rise_slope, -5.0, 5.0))
    fade_slope = float(np.clip(fade_slope, -5.0, 5.0))

    g_minus_r, color_drift = compute_color_series(df)
    cls_info = parse_classifier_summary(prob_df)

    features = np.array(
        [
            neg_frac,
            amp_mag,
            rise_slope,
            fade_slope,
            g_minus_r,
            color_drift,
            cls_info["lc_entropy"],
            cls_info["ai_disagreement"],
            timespan_days,
            rb_median,
        ],
        dtype=np.float32,
    )

    # Pack last MAX_LC_POINTS detections into fixed-length padded arrays (NaN-padded)
    sub_n = min(len(df), MAX_LC_POINTS)
    tail_idx = slice(len(df) - sub_n, len(df))
    t0 = float(mjd[len(df) - sub_n])

    lc_dt = np.full(MAX_LC_POINTS, np.nan, dtype=np.float32)
    lc_mag = np.full(MAX_LC_POINTS, np.nan, dtype=np.float32)
    lc_err = np.full(MAX_LC_POINTS, np.nan, dtype=np.float32)
    lc_fid = np.zeros(MAX_LC_POINTS, dtype=np.int8)
    lc_sign = np.zeros(MAX_LC_POINTS, dtype=np.int8)

    lc_dt[:sub_n] = (mjd[tail_idx] - t0).astype(np.float32)
    lc_mag[:sub_n] = mag[tail_idx].astype(np.float32)
    lc_err[:sub_n] = mag_err[tail_idx].astype(np.float32)
    lc_fid[:sub_n] = fid[tail_idx]
    lc_sign[:sub_n] = signs[tail_idx]

    return {
        "features": features,
        "lc_dt": lc_dt,
        "lc_mag": lc_mag,
        "lc_err": lc_err,
        "lc_fid": lc_fid,
        "lc_sign": lc_sign,
        "ndet": int(len(df)),
        "t0_mjd": t0,
        "brightest_mag": float(np.min(mag)),
        **cls_info,
    }
