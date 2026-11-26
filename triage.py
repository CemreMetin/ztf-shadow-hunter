"""Sunday dark-mode keyboard triage UI for ZTF Shadow & Blindspot Hunter.

Controls:
  [Right]  Verify / Interesting candidate (saves to triage_logs/verified_candidates.csv)
  [Down]   Reject / Noise or boring       (saves to triage_logs/rejected_candidates.csv)
  [Up]     Flag for deep follow-up        (saves to triage_logs/flagged_candidates.csv)
  [Left]   Undo previous decision
  [o]      Open ALeRCE + Legacy Survey + TNS in browser for current candidate
  [q/Esc]  Save & quit
"""

import glob
import os
import sys
import webbrowser

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["toolbar"] = "None"


class ZTFWeirdTriage:
    def __init__(self, npz_path: str, log_dir: str = "triage_logs"):
        self.npz_path = npz_path
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)

        data = np.load(npz_path, allow_pickle=True)
        self.oids = data["oids"]
        self.ra = data["ra"]
        self.dec = data["dec"]
        self.tracks = data["tracks"]
        self.tags = data["tags"]
        self.catalog_status = (
            data["catalog_status"]
            if "catalog_status" in data
            else np.full(len(self.oids), "UNKNOWN", dtype=object)
        )
        self.features = data["features"]
        self.lc_dt = data["lc_dt"]
        self.lc_mag = data["lc_mag"]
        self.lc_err = data["lc_err"]
        self.lc_fid = data["lc_fid"]
        self.lc_sign = data["lc_sign"]
        self.ndet = data["ndet"]
        self.brightest_mag = data["brightest_mag"]
        self.stamp_cls = data["stamp_top_class"]
        self.stamp_prob = data["stamp_top_prob"]
        self.lc_cls = data["lc_top_class"]
        self.lc_prob = data["lc_top_prob"]
        self.scores = (
            data["weirdness_scores"]
            if "weirdness_scores" in data
            else np.full(len(self.oids), np.nan)
        )

        self.total = len(self.oids)
        self.idx = 0
        self.decisions = {}

        self.setup_ui()
        self.update_plot()

    def setup_ui(self) -> None:
        self.fig = plt.figure(figsize=(14, 7.2), facecolor="#0d1117")
        self.fig.canvas.manager.set_window_title(
            f"ZTF Shadow & Blindspot Triyaj - {os.path.basename(self.npz_path)}"
        )

        # Left: Light curve (g & r bands, positive vs negative alerts)
        self.ax_lc = self.fig.add_axes([0.06, 0.13, 0.50, 0.69], facecolor="#161b22")
        # Right-top: g - r color evolution
        self.ax_col = self.fig.add_axes([0.62, 0.51, 0.34, 0.31], facecolor="#161b22")
        # Right-bottom: Weirdness & AI confusion horizontal bars
        self.ax_bar = self.fig.add_axes([0.62, 0.13, 0.34, 0.29], facecolor="#161b22")

        self.hud = self.fig.text(
            0.06, 0.89, "", color="#58a6ff", fontsize=10.5, family="monospace", fontweight="bold"
        )
        self.subhud = self.fig.text(
            0.06, 0.85, "", color="#c9d1d9", fontsize=9.5, family="monospace"
        )
        self.fig.text(
            0.06,
            0.03,
            "[Sag] Onayla  |  [Asagi] Ele  |  [Yukari] Derin Incele  |  [Sol] Geri Al  |  [o] ALeRCE+Legacy+TNS Ac  |  [q] Kaydet & Cikis",
            color="#8b949e",
            fontsize=9,
            family="monospace",
        )

        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.fig.canvas.mpl_connect("close_event", self.on_close)

    def _compute_paired_color(self, dt, mag, fid):
        g_mask = (fid == 1) & np.isfinite(dt) & np.isfinite(mag)
        r_mask = (fid == 2) & np.isfinite(dt) & np.isfinite(mag)
        if not np.any(g_mask) or not np.any(r_mask):
            return np.array([]), np.array([])
        g_t, g_m = dt[g_mask], mag[g_mask]
        r_t, r_m = dt[r_mask], mag[r_mask]
        out_t, out_c = [], []
        for t, m in zip(g_t, g_m):
            j = int(np.argmin(np.abs(r_t - t)))
            if abs(r_t[j] - t) <= 2.5:
                out_t.append(t)
                out_c.append(m - r_m[j])
        return np.array(out_t), np.array(out_c)

    def update_plot(self) -> None:
        if self.idx >= self.total:
            self.hud.set_text("TUM ADAYLAR TARANDI! Kararlar kaydedildi — kapatmak icin [q]'ya bas.")
            self.subhud.set_text("")
            self.fig.canvas.draw_idle()
            return

        oid = str(self.oids[self.idx])
        ra = float(self.ra[self.idx])
        dec = float(self.dec[self.idx])
        track = str(self.tracks[self.idx])
        tags = str(self.tags[self.idx])
        f = self.features[self.idx]
        neg_frac, amp_mag, rise_slope, _fade, g_minus_r, _drift, lc_ent, ai_dis, span_d, rb_med = [
            float(x) for x in f
        ]
        score = float(self.scores[self.idx])
        score_str = f"%{score * 100:.1f}" if np.isfinite(score) else "N/A"

        track_color = "#ff7b72" if track == "SHADOW" else ("#d2a8ff" if track == "HYBRID" else "#58a6ff")
        cat_status = str(self.catalog_status[self.idx])
        cat_badge = "★ UNCATALOGED (SIMBAD/VSX YOK!)" if cat_status == "UNCATALOGED" else cat_status
        self.hud.set_color("#3fb950" if cat_status == "UNCATALOGED" else track_color)
        self.hud.set_text(
            f"[{track}] {oid}  |  Weirdness: {score_str}  |  Katalog: {cat_badge}  |  ({self.idx + 1}/{self.total})"
        )
        self.subhud.set_text(
            f"Etiket: {tags} | RA/Dec: {ra:.4f}, {dec:+.4f} | ndet: {int(self.ndet[self.idx])} | "
            f"Genlik: {amp_mag:.2f}m | Stamp AI: {self.stamp_cls[self.idx]}({self.stamp_prob[self.idx]*100:.0f}%) "
            f"vs LC AI: {self.lc_cls[self.idx]}({self.lc_prob[self.idx]*100:.0f}%)"
        )

        # --- Left Panel: Light Curve ---
        self.ax_lc.clear()
        self.ax_lc.set_facecolor("#161b22")
        dt = self.lc_dt[self.idx]
        mag = self.lc_mag[self.idx]
        err = self.lc_err[self.idx]
        fid = self.lc_fid[self.idx]
        sign = self.lc_sign[self.idx]
        valid = np.isfinite(dt) & np.isfinite(mag)

        specs = [
            ((fid == 1) & (sign >= 0) & valid, "o", "#39c5cf", "g-band parlama (+)"),
            ((fid == 1) & (sign < 0) & valid, "v", "#a371f7", "g-band KARARMA (-)"),
            ((fid == 2) & (sign >= 0) & valid, "o", "#f0883e", "r-band parlama (+)"),
            ((fid == 2) & (sign < 0) & valid, "v", "#ff4d6d", "r-band KARARMA (-)"),
        ]
        for mask, marker, color, lbl in specs:
            if np.any(mask):
                self.ax_lc.errorbar(
                    dt[mask],
                    mag[mask],
                    yerr=np.clip(err[mask], 0.0, 1.0),
                    fmt=marker,
                    color=color,
                    ms=6.5 if marker == "v" else 5.5,
                    elinewidth=1.0,
                    capsize=2,
                    label=lbl,
                )

        self.ax_lc.invert_yaxis()  # Astronomy convention: smaller mag = brighter (up)
        self.ax_lc.set_title("ZTF Isik Egrisi (Yukari = Parlak | Asagi = Kararma)", color="#c9d1d9", fontsize=10)
        self.ax_lc.set_xlabel("Ilk gozlemden itibaren gecen gun", color="#8b949e", fontsize=9)
        self.ax_lc.set_ylabel("Parlaklik (mag)", color="#8b949e", fontsize=9)
        self.ax_lc.grid(True, color="#21262d", linestyle="--")
        self.ax_lc.tick_params(colors="#8b949e")
        self.ax_lc.legend(loc="best", facecolor="#0d1117", edgecolor="#30363d", labelcolor="#c9d1d9", fontsize=8)

        # --- Right-Top Panel: g - r Color Evolution ---
        self.ax_col.clear()
        self.ax_col.set_facecolor("#161b22")
        ct, cc = self._compute_paired_color(dt[valid], mag[valid], fid[valid])
        self.ax_col.axhspan(-2.0, 0.0, color="#1f6feb", alpha=0.12)
        self.ax_col.axhspan(0.5, 3.0, color="#da3633", alpha=0.10)
        self.ax_col.axhline(0.0, color="#58a6ff", linestyle=":", lw=1)
        if len(ct) > 0:
            self.ax_col.plot(ct, cc, "s-", color="#79c0ff", ms=5, lw=1.2)
            ylo = min(-0.6, float(np.min(cc)) - 0.25)
            yhi = max(0.8, float(np.max(cc)) + 0.25)
            self.ax_col.set_ylim(ylo, yhi)
        else:
            self.ax_col.text(
                0.5, 0.5, f"Eslesmis g-r yok\n(Medyan g-r: {g_minus_r:+.2f})",
                transform=self.ax_col.transAxes, ha="center", va="center", color="#8b949e", fontsize=9
            )
            self.ax_col.set_ylim(-0.8, 1.0)

        self.ax_col.set_title("Renk Evrimi (g - r) | Mavi Bolge < 0 = Sicak / TDE-FBOT", color="#c9d1d9", fontsize=9.5)
        self.ax_col.set_ylabel("g - r (mag)", color="#8b949e", fontsize=8.5)
        self.ax_col.grid(True, color="#21262d", linestyle="--")
        self.ax_col.tick_params(colors="#8b949e")

        # --- Right-Bottom Panel: Weirdness & AI Metrics ---
        self.ax_bar.clear()
        self.ax_bar.set_facecolor("#161b22")
        labels = ["Real/Bogus", "AI Entropi", "AI Kavga", "Kararma (-)", "Weirdness"]
        vals = [
            rb_med,
            lc_ent,
            ai_dis,
            neg_frac,
            score if np.isfinite(score) else 0.0,
        ]
        colors = ["#3fb950", "#d29922", "#f78166", "#a371f7", "#58a6ff"]
        y_pos = np.arange(len(labels))
        self.ax_bar.barh(y_pos, vals, color=colors, height=0.6)
        self.ax_bar.set_yticks(y_pos)
        self.ax_bar.set_yticklabels(labels, color="#c9d1d9", fontsize=8.5)
        self.ax_bar.set_xlim(0.0, 1.05)
        for yp, v in zip(y_pos, vals):
            self.ax_bar.text(min(v + 0.02, 0.90), yp, f"{v*100:.0f}%", va="center", color="#c9d1d9", fontsize=8)
        self.ax_bar.set_title("Anomali & AI Sinyal Kirilimi", color="#c9d1d9", fontsize=9.5)
        self.ax_bar.grid(True, axis="x", color="#21262d", linestyle="--")
        self.ax_bar.tick_params(colors="#8b949e")

        self.fig.canvas.draw_idle()

    def open_in_browser(self) -> None:
        if self.idx >= self.total:
            return
        oid = str(self.oids[self.idx])
        ra = float(self.ra[self.idx])
        dec = float(self.dec[self.idx])
        urls = [
            f"https://alerce.online/object/{oid}",
            f"https://www.legacysurvey.org/viewer?ra={ra:.6f}&dec={dec:.6f}&layer=ls-dr10&zoom=15",
            f"https://www.wis-tns.org/search?ra={ra:.5f}&decl={dec:.5f}&radius=5&coords_unit=arcsec",
            f"https://www.aavso.org/vsx/index.php?view=results.submit1&coords={ra:.5f}+{dec:+.5f}&size=10&unit=3",
        ]
        for u in urls:
            webbrowser.open_new_tab(u)
        print(f"[WEB] {oid} tarayicida acildi (ALeRCE + Legacy Survey + TNS + AAVSO VSX)")

    def record_decision(self, label: str) -> None:
        if self.idx >= self.total:
            return
        f = self.features[self.idx]
        score = float(self.scores[self.idx]) if np.isfinite(self.scores[self.idx]) else -1.0
        self.decisions[self.idx] = {
            "oid": str(self.oids[self.idx]),
            "decision": label,
            "track": str(self.tracks[self.idx]),
            "tags": str(self.tags[self.idx]),
            "catalog_status": str(self.catalog_status[self.idx]),
            "weirdness_score": round(score, 4),
            "ra": round(float(self.ra[self.idx]), 5),
            "dec": round(float(self.dec[self.idx]), 5),
            "neg_frac": round(float(f[0]), 3),
            "amp_mag": round(float(f[1]), 3),
            "g_minus_r": round(float(f[4]), 3),
            "lc_entropy": round(float(f[6]), 3),
            "ai_disagreement": round(float(f[7]), 3),
            "stamp_class": str(self.stamp_cls[self.idx]),
            "lc_class": str(self.lc_cls[self.idx]),
            "source_batch": os.path.basename(self.npz_path),
            "alerce_url": f"https://alerce.online/object/{self.oids[self.idx]}",
        }
        self.idx += 1
        self.update_plot()

    def undo(self) -> None:
        if self.idx <= 0:
            return
        self.idx -= 1
        self.decisions.pop(self.idx, None)
        self.update_plot()

    def save_decisions(self) -> None:
        if not self.decisions:
            return
        batch_name = os.path.basename(self.npz_path)
        for label in ("verified", "rejected", "flagged"):
            rows = [r for r in self.decisions.values() if r["decision"] == label]
            if not rows:
                continue
            csv_path = os.path.join(self.log_dir, f"{label}_candidates.csv")
            new_df = pd.DataFrame(rows)
            if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
                old_df = pd.read_csv(csv_path, dtype={"oid": str})
                if "source_batch" in old_df.columns:
                    keep_mask = ~(
                        (old_df["source_batch"] == batch_name)
                        & (old_df["oid"].isin(new_df["oid"]))
                    )
                    old_df = old_df[keep_mask]
                out_df = pd.concat([old_df, new_df], ignore_index=True)
            else:
                out_df = new_df
            out_df.to_csv(csv_path, index=False)
        print(f"Triyaj kararlari kaydedildi ({len(self.decisions)} aday) -> {self.log_dir}/")

    def on_close(self, _event) -> None:
        self.save_decisions()

    def on_key(self, event) -> None:
        if event.key == "right":
            self.record_decision("verified")
        elif event.key == "down":
            self.record_decision("rejected")
        elif event.key == "up":
            self.record_decision("flagged")
        elif event.key == "left":
            self.undo()
        elif event.key == "o":
            self.open_in_browser()
        elif event.key in ("q", "escape"):
            self.save_decisions()
            plt.close(self.fig)
            sys.exit(0)


def find_latest_npz():
    for pattern in ("pending_candidates/scored_*.npz", "pending_candidates/batch_*.npz"):
        files = sorted(glob.glob(pattern))
        if files:
            return files[-1]
    return None


if __name__ == "__main__":
    target_npz = sys.argv[1] if len(sys.argv) >= 2 else find_latest_npz()
    if not target_npz or not os.path.exists(target_npz):
        print("Kullanim: python triage.py [pending_candidates/scored_sunday_xxx.npz]")
        print("pending_candidates/ icinde henuz paket bulunamadi.")
        sys.exit(1)
    ZTFWeirdTriage(target_npz)
    plt.show()
