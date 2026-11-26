"""Lightweight, API-key-free catalog cross-match against SIMBAD and AAVSO VSX (via VizieR).

Checks whether a ZTF coordinate is already a known variable/transient in major catalogs.
Fail-safe by design: transient network errors never crash the nightly pipeline.
"""

from typing import Tuple
import requests

SIMBAD_TAP_URLS = (
    "https://simbad.u-strasbg.fr/simbad/sim-tap/sync",
    "https://simbad.cfa.harvard.edu/simbad/sim-tap/sync",
)

# VizieR lightweight TSV cone-search endpoint for AAVSO VSX (B/vsx/vsx)
VIZIER_VSX_URLS = (
    "https://vizier.cds.unistra.fr/viz-bin/asu-tsv",
    "https://vizier.cfa.harvard.edu/viz-bin/asu-tsv",
)


def _query_simbad(ra: float, dec: float, radius_arcsec: float = 5.0) -> Tuple[bool, str]:
    radius_deg = radius_arcsec / 3600.0
    adql = (
        f"SELECT TOP 1 main_id, otype FROM basic "
        f"WHERE CONTAINS(POINT('ICRS', ra, dec), "
        f"CIRCLE('ICRS', {ra:.6f}, {dec:.6f}, {radius_deg:.6f})) = 1"
    )
    last_err = None
    for url in SIMBAD_TAP_URLS:
        try:
            resp = requests.get(
                url,
                params={"request": "doQuery", "lang": "adql", "format": "json", "query": adql},
                timeout=10,
            )
            resp.raise_for_status()
            rows = resp.json().get("data", [])
            if not rows:
                return False, ""
            main_id, otype = rows[0][0], rows[0][1]
            return True, f"SIMBAD:{otype} ({main_id})"
        except Exception as exc:
            last_err = exc
    raise RuntimeError(f"SIMBAD query failed: {last_err}")


def _query_vsx(ra: float, dec: float, radius_arcsec: float = 5.0) -> Tuple[bool, str]:
    params = {
        "-source": "B/vsx/vsx",
        "-c": f"{ra:.6f} {dec:+.6f}",
        "-c.rs": f"{radius_arcsec:.1f}",
        "-out": "Name,Type",
        "-out.max": "1",
    }
    last_err = None
    for url in VIZIER_VSX_URLS:
        try:
            resp = requests.get(url, params=params, timeout=10)
            resp.raise_for_status()
            lines = [
                ln.strip()
                for ln in resp.text.splitlines()
                if ln.strip() and not ln.startswith("#")
            ]
            # asu-tsv returns header, units, dashes, then data rows
            data_rows = []
            passed_dashes = False
            for ln in lines:
                if ln.startswith("---"):
                    passed_dashes = True
                    continue
                if passed_dashes:
                    data_rows.append(ln)
            if not data_rows:
                return False, ""
            parts = [p.strip() for p in data_rows[0].split("\t")]
            name = parts[0] if len(parts) > 0 else "VSX"
            vtype = parts[1] if len(parts) > 1 and parts[1] else "Var"
            return True, f"VSX:{vtype} ({name})"
        except Exception as exc:
            last_err = exc
    raise RuntimeError(f"VSX query failed: {last_err}")


def check_catalog_novelty(ra: float, dec: float, radius_arcsec: float = 5.0) -> Tuple[bool, str]:
    """Return (is_uncataloged, status_label) for a sky coordinate.

    - (True, "UNCATALOGED")           -> Not found in SIMBAD or AAVSO VSX within radius!
    - (False, "VSX:... | SIMBAD:...") -> Known cataloged object
    - (False, "CHECK_FAILED")         -> Both services timed out / errored
    """
    simbad_ok = False
    vsx_ok = False
    matches = []

    try:
        hit, label = _query_simbad(ra, dec, radius_arcsec)
        simbad_ok = True
        if hit:
            matches.append(label)
    except Exception:
        pass

    try:
        hit, label = _query_vsx(ra, dec, radius_arcsec)
        vsx_ok = True
        if hit:
            matches.append(label)
    except Exception:
        pass

    if matches:
        return False, " | ".join(matches)
    if simbad_ok or vsx_ok:
        return True, "UNCATALOGED"
    return False, "CHECK_FAILED"


if __name__ == "__main__":
    # Test 1: Boyajian's Star (KIC 8462852) -> should match SIMBAD & VSX
    print("Tabby's Star:", check_catalog_novelty(301.564386, 44.456886, radius_arcsec=10.0))
    # Test 2: Empty sky patch -> should return UNCATALOGED
    print("Empty patch :", check_catalog_novelty(150.0123, 2.0456, radius_arcsec=5.0))
