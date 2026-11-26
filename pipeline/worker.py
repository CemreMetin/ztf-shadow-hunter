"""Nightly GitHub Actions worker for ZTF Shadow & Blindspot Hunter.

Pulls recent ZTF objects from ALeRCE across two unorthodox pools:
  1. SHADOW pool    : Variable / dimming candidates with dominant negative alerts (isdiffpos = -1)
  2. BLINDSPOT pool : Uncertain / cross-classifier conflict / blue-fire recent transients

Saves compact .npz vectors to pending_candidates/ and byproduct CSVs to data/.
"""

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Set

import numpy as np
import pandas as pd
from alerce.core import Alerce
from astropy.time import Time

from pipeline.catalog import check_catalog_novelty
from pipeline.features import FEATURE_NAMES, extract_features_and_pack
from pipeline.vetos import evaluate_candidate

SEEN_FILE = Path("data/seen_oids.json")
EB_CSV = Path("data/known_eclipsing_binaries.csv")
AGN_CSV = Path("data/agn_flares.csv")
OUT_DIR = Path("pending_candidates")

LOOKBACK_DAYS = int(os.environ.get("ZTF_LOOKBACK_DAYS", "14"))
MAX_PER_QUERY = int(os.environ.get("ZTF_MAX_PER_QUERY", "45"))
MAX_CANDIDATES_PER_NIGHT = int(os.environ.get("ZTF_MAX_CANDIDATES", "60"))


def load_seen() -> Set[str]:
    if not SEEN_FILE.exists():
        return set()
    try:
        data = json.loads(SEEN_FILE.read_text(encoding="utf-8"))
        return set(data) if isinstance(data, list) else set()
    except Exception:
        return set()


def save_seen(seen: Set[str], max_items: int = 15000) -> None:
    SEEN_FILE.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(seen)[-max_items:]
    SEEN_FILE.write_text(json.dumps(ordered, indent=2), encoding="utf-8")


def append_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    new_df = pd.DataFrame(rows)
    if path.exists() and path.stat().st_size > 0:
        old_df = pd.read_csv(path, dtype={"oid": str})
        merged = pd.concat([old_df, new_df], ignore_index=True).drop_duplicates(subset=["oid"], keep="last")
    else:
        merged = new_df
    merged.to_csv(path, index=False)


def safe_query_objects(client: Alerce, **kwargs: Any) -> pd.DataFrame:
    for attempt in range(1, 4):
        try:
            df = client.query_objects(format="pandas", **kwargs)
            return df if df is not None else pd.DataFrame()
        except Exception as exc:
            print(f"[WARN] query_objects deneme {attempt} hatasi ({kwargs.get('class_name')}): {exc}")
            if attempt < 3:
                time.sleep(2 ** attempt)
    return pd.DataFrame()


def harvest_candidate_pool(client: Alerce, mjd_now: float) -> pd.DataFrame:
    """Gather diverse unorthodox ZTF candidates from ALeRCE without focusing on textbook SNe."""
    mjd_start = mjd_now - LOOKBACK_DAYS
    frames: List[pd.DataFrame] = []

    # Pool 1: SHADOW — Variable stars & YSO/CV/Periodic-Other active recently (prime dipper hunting ground)
    for cls_name in ("YSO", "Periodic-Other", "CV/Nova"):
        frames.append(
            safe_query_objects(
                client,
                classifier="lc_classifier",
                class_name=cls_name,
                ndet=[5, 300],
                lastmjd=[mjd_start, mjd_now],
                order_by="lastmjd",
                order_mode="DESC",
                page_size=MAX_PER_QUERY,
            )
        )

    # Pool 2: BLINDSPOT — Borderline / uncertain stamp_classifier transients (where AI disagrees)
    for cls_name in ("SN", "VS"):
        frames.append(
            safe_query_objects(
                client,
                classifier="stamp_classifier",
                class_name=cls_name,
                probability=0.35,
                ndet=[3, 120],
                firstmjd=[mjd_start, mjd_now],
                order_by="firstmjd",
                order_mode="DESC",
                page_size=MAX_PER_QUERY,
            )
        )

    # Pool 3: RARE TRANSIENTS — SLSN / SNIbc / AGN-like rapid events
    for cls_name in ("SLSN", "SNIbc"):
        frames.append(
            safe_query_objects(
                client,
                classifier="lc_classifier",
                class_name=cls_name,
                ndet=[3, 120],
                lastmjd=[mjd_start, mjd_now],
                order_by="lastmjd",
                order_mode="DESC",
                page_size=25,
            )
        )

    non_empty = [f for f in frames if not f.empty and "oid" in f.columns]
    if not non_empty:
        return pd.DataFrame()

    combined = pd.concat(non_empty, ignore_index=True)
    combined = combined.drop_duplicates(subset=["oid"])
    # Shuffle deterministically by day so Shadow, Blindspot and Rare pools are sampled evenly
    combined = combined.sample(frac=1.0, random_state=int(mjd_now)).reset_index(drop=True)
    return combined


def run_nightly_worker() -> None:
    client = Alerce()
    seen = load_seen()
    mjd_now = float(Time(datetime.now(timezone.utc)).mjd)

    print(f"[INFO] ZTF Shadow & Blindspot gece taramasi basliyor (MJD={mjd_now:.2f}, seen={len(seen)})")
    pool = harvest_candidate_pool(client, mjd_now)
    if pool.empty:
        print("[WARN] ALeRCE havuzu bos dondu; bu gece cikti uretilmedi.")
        return

    print(f"[INFO] Toplam benzersiz havuz: {len(pool)} nesne")

    kept: List[Dict[str, Any]] = []
    eb_rows: List[Dict[str, Any]] = []
    agn_rows: List[Dict[str, Any]] = []
    newly_seen: Set[str] = set()
    rejected_counts: Dict[str, int] = {}

    for _, row in pool.iterrows():
        if len(kept) >= MAX_CANDIDATES_PER_NIGHT:
            break

        oid = str(row["oid"])
        if oid in seen:
            continue
        newly_seen.add(oid)

        ra = float(row.get("meanra", np.nan))
        dec = float(row.get("meandec", np.nan))

        try:
            det_df = client.query_detections(oid, format="pandas")
            prob_df = client.query_probabilities(oid, format="pandas")
        except Exception as exc:
            print(f"[WARN] {oid} veri cekilemedi: {exc}")
            continue

        packed = extract_features_and_pack(det_df, prob_df)
        if packed is None:
            rejected_counts["too_few_points"] = rejected_counts.get("too_few_points", 0) + 1
            continue

        verdict, label, tags = evaluate_candidate(packed)

        if verdict == "REJECT":
            rejected_counts[label] = rejected_counts.get(label, 0) + 1
            continue

        if verdict == "BYPRODUCT":
            rec = {
                "oid": oid,
                "ra": round(ra, 5),
                "dec": round(dec, 5),
                "ndet": packed["ndet"],
                "lc_class": packed["lc_top_class"],
                "lc_prob": round(packed["lc_top_prob"], 3),
                "amp_mag": round(float(packed["features"][1]), 3),
                "discovered_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            }
            if label == "eclipsing_binary":
                eb_rows.append(rec)
            elif label == "agn_flare":
                agn_rows.append(rec)
            continue

        is_uncataloged, catalog_status = check_catalog_novelty(ra, dec)
        if is_uncataloged:
            tags = list(tags) + ["UNCATALOGED"]

        kept.append(
            {
                "oid": oid,
                "ra": ra,
                "dec": dec,
                "track": label,
                "tags": ",".join(tags),
                "catalog_status": catalog_status,
                **packed,
            }
        )

    # Save byproducts
    append_csv(EB_CSV, eb_rows)
    append_csv(AGN_CSV, agn_rows)

    # Save compact .npz batch for local AI scoring & Sunday triage
    if kept:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
        out_path = OUT_DIR / f"batch_{stamp}.npz"

        np.savez_compressed(
            out_path,
            oids=np.array([k["oid"] for k in kept], dtype=object),
            ra=np.array([k["ra"] for k in kept], dtype=np.float64),
            dec=np.array([k["dec"] for k in kept], dtype=np.float64),
            tracks=np.array([k["track"] for k in kept], dtype=object),
            tags=np.array([k["tags"] for k in kept], dtype=object),
            catalog_status=np.array([k["catalog_status"] for k in kept], dtype=object),
            features=np.vstack([k["features"] for k in kept]).astype(np.float32),
            feature_names=np.array(FEATURE_NAMES, dtype=object),
            lc_dt=np.vstack([k["lc_dt"] for k in kept]).astype(np.float32),
            lc_mag=np.vstack([k["lc_mag"] for k in kept]).astype(np.float32),
            lc_err=np.vstack([k["lc_err"] for k in kept]).astype(np.float32),
            lc_fid=np.vstack([k["lc_fid"] for k in kept]).astype(np.int8),
            lc_sign=np.vstack([k["lc_sign"] for k in kept]).astype(np.int8),
            ndet=np.array([k["ndet"] for k in kept], dtype=np.int32),
            t0_mjd=np.array([k["t0_mjd"] for k in kept], dtype=np.float64),
            brightest_mag=np.array([k["brightest_mag"] for k in kept], dtype=np.float32),
            stamp_top_class=np.array([k["stamp_top_class"] for k in kept], dtype=object),
            stamp_top_prob=np.array([k["stamp_top_prob"] for k in kept], dtype=np.float32),
            lc_top_class=np.array([k["lc_top_class"] for k in kept], dtype=object),
            lc_top_prob=np.array([k["lc_top_prob"] for k in kept], dtype=np.float32),
        )
        print(f"[OK] {len(kept)} unorthodox aday kaydedildi -> {out_path}")
    else:
        print("[INFO] Bu gece yeni unorthodox aday cikmadi.")

    seen.update(newly_seen)
    save_seen(seen)
    print(
        f"[OZET] Yeni taranan: {len(newly_seen)} | Aday: {len(kept)} | "
        f"EB yan urun: {len(eb_rows)} | AGN yan urun: {len(agn_rows)} | Elenen: {rejected_counts}"
    )


if __name__ == "__main__":
    try:
        run_nightly_worker()
    except Exception as err:
        print(f"HATA: {err}", file=sys.stderr)
        sys.exit(1)
