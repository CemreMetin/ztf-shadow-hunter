"""Combine recent nightly .npz batches and score them with IsolationForest + domain weirdness.

Usage:
    python -m ai.weirdness                  # Merge & score last 7 days of batches
    python -m ai.weirdness --days 14        # Merge & score last 14 days
    python -m ai.weirdness path/to/batch.npz
"""

import argparse
import glob
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler

PENDING_DIR = Path("pending_candidates")


def compute_domain_weirdness(
    features: np.ndarray,
    tracks: np.ndarray,
    catalog_status: np.ndarray = None,
) -> np.ndarray:
    """Compute a 0..1 domain-informed weirdness score from the 10 physical/AI features.

    Feature order (pipeline.features.FEATURE_NAMES):
      0: neg_frac, 1: amp_mag, 2: rise_slope, 3: fade_slope,
      4: g_minus_r, 5: color_drift, 6: lc_entropy, 7: ai_disagreement,
      8: timespan_days, 9: rb_median
    """
    neg_frac = np.clip(features[:, 0], 0.0, 1.0)
    amp_mag = np.clip(features[:, 1], 0.0, 4.0)
    rise_slope = features[:, 2]
    g_minus_r = features[:, 4]
    lc_entropy = np.clip(features[:, 6], 0.0, 1.0)
    ai_disagreement = np.clip(features[:, 7], 0.0, 1.0)
    rb_median = np.clip(features[:, 9], 0.0, 1.0)

    # 1. Shadow Dipping strength (dominant negative alerts + meaningful depth)
    shadow_score = neg_frac * np.clip(amp_mag / 1.5, 0.0, 1.0)

    # 2. AI Blindspot strength (conflict + entropy)
    blindspot_score = 0.55 * ai_disagreement + 0.45 * lc_entropy

    # 3. Blue Fire strength (very blue color + fast brightening)
    blue_strength = np.clip((-g_minus_r - 0.05) / 0.6, 0.0, 1.0)
    fast_rise = np.clip((-rise_slope) / 0.5, 0.0, 1.0)
    blue_fire_score = 0.65 * blue_strength + 0.35 * fast_rise

    # Primary signal is the strongest unorthodox channel, boosted if HYBRID or UNCATALOGED
    primary = np.maximum.reduce([shadow_score, blindspot_score, blue_fire_score])
    hybrid_bonus = np.where(tracks == "HYBRID", 0.14, 0.0)
    novelty_bonus = (
        np.where(catalog_status == "UNCATALOGED", 0.12, 0.0)
        if catalog_status is not None
        else 0.0
    )

    # Slight weighting by real/bogus quality so crisp real signals rank above noisy ones
    quality = 0.75 + 0.25 * rb_median
    return np.clip((primary + hybrid_bonus + novelty_bonus) * quality, 0.0, 1.0).astype(np.float32)


def compute_isolation_score(features: np.ndarray) -> np.ndarray:
    """Unsupervised Isolation Forest outlier score normalized to [0, 1]."""
    n = features.shape[0]
    if n < 8:
        return np.full(n, 0.5, dtype=np.float32)

    clean = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    scaled = RobustScaler().fit_transform(clean)

    iso = IsolationForest(
        n_estimators=200,
        contamination="auto",
        random_state=42,
    )
    iso.fit(scaled)
    # decision_function: lower = more anomalous -> invert so higher = weirder
    raw = -iso.decision_function(scaled)
    lo, hi = float(np.min(raw)), float(np.max(raw))
    if hi - lo < 1e-6:
        return np.full(n, 0.5, dtype=np.float32)
    return ((raw - lo) / (hi - lo)).astype(np.float32)


def load_and_merge_batches(paths: List[Path]) -> Dict[str, np.ndarray]:
    """Merge multiple batch_*.npz files and deduplicate by OID (keeping latest)."""
    merged_lists: Dict[str, List[np.ndarray]] = {}
    feature_names = None

    for p in paths:
        data = np.load(p, allow_pickle=True)
        if "oids" not in data or len(data["oids"]) == 0:
            continue
        if feature_names is None and "feature_names" in data:
            feature_names = data["feature_names"]
        for k in data.files:
            if k == "feature_names":
                continue
            merged_lists.setdefault(k, []).append(data[k])

    if not merged_lists:
        return {}

    merged = {k: np.concatenate(v, axis=0) for k, v in merged_lists.items()}

    # Deduplicate by oid (keep last occurrence)
    oids = merged["oids"]
    seen_pos: Dict[str, int] = {}
    for i, oid in enumerate(oids):
        seen_pos[str(oid)] = i
    keep_idx = np.array(sorted(seen_pos.values()), dtype=np.int64)

    result = {k: v[keep_idx] for k, v in merged.items()}
    if feature_names is not None:
        result["feature_names"] = feature_names
    return result


def score_batches(batch_paths: List[Path], output_path: Path) -> Path:
    data = load_and_merge_batches(batch_paths)
    if not data or len(data.get("oids", [])) == 0:
        raise RuntimeError("Skorlanacak aday bulunamadi.")

    features = data["features"].astype(np.float32)
    tracks = data["tracks"]
    catalog_status = data["catalog_status"] if "catalog_status" in data else None

    domain_scores = compute_domain_weirdness(features, tracks, catalog_status)
    iso_scores = compute_isolation_score(features)

    # Blend: 65% physical/AI domain weirdness + 35% unsupervised IsolationForest outlier score
    if len(features) >= 8:
        weirdness = (0.65 * domain_scores + 0.35 * iso_scores).astype(np.float32)
    else:
        weirdness = domain_scores

    # Sort descending by weirdness score so the strangest objects appear first in triage.py
    order = np.argsort(-weirdness)
    sorted_data: Dict[str, np.ndarray] = {}
    for k, v in data.items():
        if k == "feature_names":
            sorted_data[k] = v
        else:
            sorted_data[k] = v[order]

    sorted_data["weirdness_scores"] = weirdness[order]
    sorted_data["domain_scores"] = domain_scores[order]
    sorted_data["iso_scores"] = iso_scores[order]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **sorted_data)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Score ZTF Shadow & Blindspot candidates for Sunday triage.")
    parser.add_argument("batch", nargs="?", help="Optional single .npz file to score")
    parser.add_argument("--days", type=int, default=7, help="How many latest batch_*.npz files to merge (default: 7)")
    args = parser.parse_args()

    if args.batch:
        paths = [Path(args.batch)]
    else:
        found = sorted(Path(p) for p in glob.glob(str(PENDING_DIR / "batch_*.npz")))
        paths = found[-args.days :]

    if not paths:
        print("pending_candidates/ icinde batch_*.npz dosyasi bulunamadi. Once `git pull` veya worker calistir.")
        return

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    out_path = PENDING_DIR / f"scored_sunday_{stamp}.npz"
    saved = score_batches(paths, out_path)

    scored = np.load(saved, allow_pickle=True)
    n = len(scored["oids"])
    top_oid = scored["oids"][0]
    top_score = float(scored["weirdness_scores"][0]) * 100.0
    print(f"[OK] {len(paths)} gece paketi birlestirildi -> {n} benzersiz aday skorlandi.")
    print(f"[TOP] En tuhaf aday: {top_oid} (Weirdness: %{top_score:.1f}) -> {saved}")
    print("Simdi Pazar triyajini baslat: python triage.py")


if __name__ == "__main__":
    main()
