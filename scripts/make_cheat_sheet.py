"""Generate a visual dark-mode cheat sheet explaining how to read ZTF light curves in triage.py."""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def build_cheat_sheet(out_path: str = "triage_cheat_sheet.png") -> Path:
    rng = np.random.default_rng(42)
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), facecolor="#0d1117")
    fig.suptitle(
        "ZTF Shadow & Blindspot — Pazar Triyaji Grafik Okuma Kopya Kagidi",
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
    ax.plot(t, dip, "v-", color="#a371f7", ms=7, label="g KARARMA (-)")
    ax.plot(t, dip_r, "v-", color="#ff4d6d", ms=7, label="r KARARMA (-)")
    ax.invert_yaxis()
    ax.set_title("[ONAYLA / DERIN INCELE] SHADOW DIPPER (Kararan Yildiz)", color="#a371f7", fontsize=10.5, fontweight="bold")
    ax.text(0.03, 0.08, "Ters ucgen (isdiffpos=-1) asagi coker.\nToz bulutu, dev halka veya Tabby's Star benzeri!", transform=ax.transAxes, color="#c9d1d9", fontsize=8.5, bbox=dict(facecolor="#0d1117", edgecolor="#30363d"))

    # 2. Blue Fire / TDE / FBOT (KEEP!)
    ax = axes[0, 1]
    ax.set_facecolor("#161b22")
    g_bf = 19.8 - 2.6 * np.exp(-np.abs(t - 6) / 7.0) + rng.normal(0, 0.05, size=len(t))
    r_bf = g_bf + 0.45 + rng.normal(0, 0.04, size=len(t))  # g brighter than r -> g - r < 0 (blue!)
    ax.plot(t, g_bf, "o-", color="#39c5cf", ms=6, label="g parlama (daha parlak!)")
    ax.plot(t, r_bf, "o-", color="#f0883e", ms=6, label="r parlama")
    ax.invert_yaxis()
    ax.set_title("[ONAYLA / TNS KONTROL] BLUE FIRE (TDE / FBOT Adayi)", color="#58a6ff", fontsize=10.5, fontweight="bold")
    ax.text(0.03, 0.08, "Hizla parlar ve g bandi hep r'nin USTUNDE kalir\n(g - r < 0). Normal SN gibi hizla kizarmaz!", transform=ax.transAxes, color="#c9d1d9", fontsize=8.5, bbox=dict(facecolor="#0d1117", edgecolor="#30363d"))

    # 3. Textbook Supernova (Boring / Already Reported)
    ax = axes[1, 0]
    ax.set_facecolor("#161b22")
    r_sn = 19.5 - 2.0 * np.exp(-((t - 8) ** 2) / 90.0) + rng.normal(0, 0.05, size=len(t))
    g_sn = r_sn + np.linspace(-0.1, 1.1, len(t)) + rng.normal(0, 0.05, size=len(t))
    ax.plot(t, g_sn, "o-", color="#39c5cf", ms=5.5, label="g (hizla soner/kizarir)")
    ax.plot(t, r_sn, "o-", color="#f0883e", ms=5.5, label="r (daha yavas soner)")
    ax.invert_yaxis()
    ax.set_title("[ELE / ZATEN BILINIYOR] Siradan Supernova", color="#d29922", fontsize=10.5, fontweight="bold")
    ax.text(0.03, 0.08, "Klasik can egrisi + hizla kizaran g-r.\nProfesyonel anketler bunu saatler icinde bildirir.", transform=ax.transAxes, color="#c9d1d9", fontsize=8.5, bbox=dict(facecolor="#0d1117", edgecolor="#30363d"))

    # 4. Artifact / Bogus / Noise (REJECT!)
    ax = axes[1, 1]
    ax.set_facecolor("#161b22")
    noise_g = 19.0 + rng.normal(0, 0.65, size=len(t))
    noise_r = 19.0 + rng.normal(0, 0.65, size=len(t))
    ax.plot(t, noise_g, "o", color="#39c5cf", ms=5, label="g daginik")
    ax.plot(t, noise_r, "o", color="#f0883e", ms=5, label="r daginik")
    ax.invert_yaxis()
    ax.set_title("[ELE] Sensor Gurultusu / Artefakt", color="#f85149", fontsize=10.5, fontweight="bold")
    ax.text(0.03, 0.08, "Tutarli egrisi yok, rastgele sicramalar.\n[o] ile ALeRCE'ye bakarsan fark goruntusu cizgili/kirli olur.", transform=ax.transAxes, color="#c9d1d9", fontsize=8.5, bbox=dict(facecolor="#0d1117", edgecolor="#30363d"))

    for ax in axes.flat:
        ax.grid(True, color="#21262d", linestyle="--")
        ax.tick_params(colors="#8b949e")
        ax.set_xlabel("Gun", color="#8b949e", fontsize=8.5)
        ax.set_ylabel("Parlaklik (mag)", color="#8b949e", fontsize=8.5)
        ax.legend(loc="upper right", facecolor="#0d1117", edgecolor="#30363d", labelcolor="#c9d1d9", fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    path = Path(out_path)
    fig.savefig(path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


if __name__ == "__main__":
    saved = build_cheat_sheet()
    print(f"[OK] Grafik okuma kopya kagidi olusturuldu -> {saved}")
