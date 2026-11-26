"""Generate a visual dark-mode cheat sheet explaining how to read ZTF light curves in triage.py."""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def build_cheat_sheet(out_path: str = "triage_cheat_sheet.png") -> Path:
    rng = np.random.default_rng(42)
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), facecolor="#0d1117")
    fig.suptitle(
        "ZTF Shadow & Blindspot — Sunday Triage Light Curve Cheat Sheet",
        color="#e6edf3",
        fontsize=14,
        fontweight="bold",
        y=0.97,
    )

    t = np.linspace(0, 30, 18)

    # 1. Shadow Dipper (KEEP!)
    ax = axes[0, 0]
    ax.set_facecolor("#161b22")
    dip = 16.0 + 1.3 * np.exp(-0.5 * ((t - 15) / 3.2) ** 2) + rng.normal(0, 0.04, size=len(t))
    dip_r = 15.6 + 1.1 * np.exp(-0.5 * ((t - 15) / 3.2) ** 2) + rng.normal(0, 0.04, size=len(t))
    ax.plot(t, dip, "v-", color="#a371f7", ms=7, label="g DIMMING (-)")
    ax.plot(t, dip_r, "v-", color="#ff4d6d", ms=7, label="r DIMMING (-)")
    ax.invert_yaxis()
    ax.set_title("[VERIFY / FLAG] SHADOW DIPPER (Sudden Dimming Star)", color="#a371f7", fontsize=10.5, fontweight="bold")
    ax.text(
        0.03, 0.08,
        "Downward triangles (isdiffpos = -1) drop sharply.\nDust clouds, giant rings, or Tabby's Star analogs.",
        transform=ax.transAxes, color="#c9d1d9", fontsize=8.5,
        bbox=dict(facecolor="#0d1117", edgecolor="#30363d"),
    )

    # 2. Blue Fire / TDE / FBOT (KEEP!)
    ax = axes[0, 1]
    ax.set_facecolor("#161b22")
    g_bf = 19.8 - 2.6 * np.exp(-np.abs(t - 6) / 7.0) + rng.normal(0, 0.05, size=len(t))
    r_bf = g_bf + 0.45 + rng.normal(0, 0.04, size=len(t))  # g brighter than r -> g - r < 0 (blue!)
    ax.plot(t, g_bf, "o-", color="#39c5cf", ms=6, label="g brightening (brighter)")
    ax.plot(t, r_bf, "o-", color="#f0883e", ms=6, label="r brightening")
    ax.invert_yaxis()
    ax.set_title("[VERIFY / CHECK TNS] BLUE FIRE (TDE / FBOT Candidate)", color="#58a6ff", fontsize=10.5, fontweight="bold")
    ax.text(
        0.03, 0.08,
        "Fast rise and g-band stays ABOVE r-band (g - r < 0).\nStays hot and blue instead of cooling like a normal SN.",
        transform=ax.transAxes, color="#c9d1d9", fontsize=8.5,
        bbox=dict(facecolor="#0d1117", edgecolor="#30363d"),
    )

    # 3. Textbook Supernova (Boring / Already Reported)
    ax = axes[1, 0]
    ax.set_facecolor("#161b22")
    r_sn = 19.5 - 2.0 * np.exp(-((t - 8) ** 2) / 90.0) + rng.normal(0, 0.05, size=len(t))
    g_sn = r_sn + np.linspace(-0.1, 1.1, len(t)) + rng.normal(0, 0.05, size=len(t))
    ax.plot(t, g_sn, "o-", color="#39c5cf", ms=5.5, label="g (cools/fades fast)")
    ax.plot(t, r_sn, "o-", color="#f0883e", ms=5.5, label="r (fades slower)")
    ax.invert_yaxis()
    ax.set_title("[REJECT / ALREADY REPORTED] Textbook Supernova", color="#d29922", fontsize=10.5, fontweight="bold")
    ax.text(
        0.03, 0.08,
        "Classic bell curve + rapid reddening (g - r increases).\nSurvey bots report these to TNS within hours.",
        transform=ax.transAxes, color="#c9d1d9", fontsize=8.5,
        bbox=dict(facecolor="#0d1117", edgecolor="#30363d"),
    )

    # 4. Artifact / Bogus / Noise (REJECT!)
    ax = axes[1, 1]
    ax.set_facecolor("#161b22")
    noise_g = 19.0 + rng.normal(0, 0.65, size=len(t))
    noise_r = 19.0 + rng.normal(0, 0.65, size=len(t))
    ax.plot(t, noise_g, "o", color="#39c5cf", ms=5, label="g scattered")
    ax.plot(t, noise_r, "o", color="#f0883e", ms=5, label="r scattered")
    ax.invert_yaxis()
    ax.set_title("[REJECT] Sensor Noise / Artifact", color="#f85149", fontsize=10.5, fontweight="bold")
    ax.text(
        0.03, 0.08,
        "No coherent curve, random jumps across epochs.\nPress [o] to inspect ALeRCE stamps for streaks/cosmic rays.",
        transform=ax.transAxes, color="#c9d1d9", fontsize=8.5,
        bbox=dict(facecolor="#0d1117", edgecolor="#30363d"),
    )

    for ax in axes.flat:
        ax.grid(True, color="#21262d", linestyle="--")
        ax.tick_params(colors="#8b949e")
        ax.set_xlabel("Days since first alert", color="#8b949e", fontsize=8.5)
        ax.set_ylabel("Brightness (mag)", color="#8b949e", fontsize=8.5)
        ax.legend(loc="upper right", facecolor="#0d1117", edgecolor="#30363d", labelcolor="#c9d1d9", fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    path = Path(out_path)
    fig.savefig(path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


if __name__ == "__main__":
    saved = build_cheat_sheet()
    print(f"[OK] Cheat sheet generated -> {saved}")
