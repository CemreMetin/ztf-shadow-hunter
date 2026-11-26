"""Unorthodox selection rules, artifact vetos, and byproduct classification."""

from typing import Any, Dict, List, Tuple


def evaluate_candidate(packed: Dict[str, Any]) -> Tuple[str, str, List[str]]:
    """Classify an analyzed ZTF object into:
      - ("CANDIDATE", track, tags)  -> Keep for Sunday weirdness scoring & triage
      - ("BYPRODUCT", kind, tags)   -> Save to bonus CSV dataset instead of discarding
      - ("REJECT", reason, [])      -> Bogus / too short / boring textbook object

    Tracks:
      - "SHADOW"    : Sudden dimming / negative-difference star (isdiffpos == -1)
      - "BLINDSPOT" : AI conflict, high entropy, or persistent blue fire (TDE/FBOT)
      - "HYBRID"    : Matches both Shadow and Blindspot signals
    """
    f = packed["features"]
    (
        neg_frac,
        amp_mag,
        rise_slope,
        _fade_slope,
        g_minus_r,
        color_drift,
        lc_entropy,
        ai_disagreement,
        timespan_days,
        rb_median,
    ) = [float(x) for x in f]

    stamp_cls = str(packed.get("stamp_top_class", "UNKNOWN"))
    stamp_prob = float(packed.get("stamp_top_prob", 0.0))
    lc_cls = str(packed.get("lc_top_class", "UNKNOWN"))
    lc_prob = float(packed.get("lc_top_prob", 0.0))

    # 1. Hard quality vetos (junk / asteroids / single-night blips)
    if rb_median < 0.55:
        return "REJECT", "low_real_bogus", []
    if stamp_cls == "bogus" and stamp_prob >= 0.70:
        return "REJECT", "stamp_bogus", []
    if stamp_cls == "asteroid" and stamp_prob >= 0.70:
        return "REJECT", "asteroid", []
    if timespan_days < 0.25:
        return "REJECT", "too_short", []

    # 2. Bonus byproduct datasets (instead of throwing interesting known classes away)
    if lc_cls == "E" and lc_prob >= 0.80:
        return "BYPRODUCT", "eclipsing_binary", ["EB_BYPRODUCT"]
    if lc_cls in ("AGN", "QSO", "Blazar") and lc_prob >= 0.85 and amp_mag >= 0.75:
        return "BYPRODUCT", "agn_flare", ["AGN_BYPRODUCT"]

    # 3. Track A — SHADOW (Sudden Dimming / Negative-Difference Star)
    shadow_tags: List[str] = []
    if neg_frac >= 0.60 and amp_mag >= 0.50:
        shadow_tags.append("SHADOW_DIP")
        if neg_frac >= 0.80 and amp_mag >= 1.00:
            shadow_tags.append("DEEP_DIMMING")

    # 4. Track B — BLINDSPOT (AI Confusion / Disagreement / Persistent Blue Fire)
    blindspot_tags: List[str] = []
    if ai_disagreement >= 0.60:
        blindspot_tags.append("AI_CONFLICT")
    if lc_entropy >= 0.85:
        blindspot_tags.append("HIGH_ENTROPY")
    if g_minus_r <= -0.10 and (rise_slope < -0.04 or color_drift <= 0.05):
        blindspot_tags.append("BLUE_FIRE")

    # Reject textbook, super-obvious objects that everyone already reports
    if not shadow_tags and not blindspot_tags:
        return "REJECT", "textbook_common", []

    if shadow_tags and blindspot_tags:
        return "CANDIDATE", "HYBRID", shadow_tags + blindspot_tags
    if shadow_tags:
        return "CANDIDATE", "SHADOW", shadow_tags
    return "CANDIDATE", "BLINDSPOT", blindspot_tags
