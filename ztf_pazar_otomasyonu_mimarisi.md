# ZTF / ALeRCE Haftalık Otomatik Tarama Sistemi (Pazar Raporu)

Bu belge, Zwicky Transient Facility (ZTF) verilerini **ALeRCE Broker** üzerinden her Pazar otomatik tarayan, bilinen gök cisimlerini eleyen ve yalnızca yüksek olasılıklı yeni geçici gök olaylarını (süpernova vb.) bildirim kanalına gönderen public GitHub Actions sisteminin tam kurulum kılavuzudur.

---

## 1. Mimari Prensipler ve Teknik Borç Koruması

* **Neden ALeRCE Broker?** Ham teleskop CCD görüntülerini indirip fark görüntüsü (difference imaging) çıkarmak yüksek disk/GPU maliyeti yaratır. ALeRCE bu işlemi zaten yapar; sistem sadece temizlenmiş ışık eğrilerini ve sınıflandırıcı skorlarını çeker.
* **Haftalık Pencere ve Durum Yönetimi (Statefulness):** GitHub Actions runner'ları her çalıştırmada sıfırlanır. Yalnızca `firstmjd` tarih filtresine güvenmek sınır vakalarda aynı nesnenin tekrar raporlanmasına veya geç işlenen nesnelerin kaçırılmasına yol açar. Bu yüzden tarama penceresi **10 gün** (güvenlik paylı) tutulur, işlenmiş nesneler `state/ztf_seen.json` dosyasında saklanır.
* **Public Repo Güvenliği:** Telegram/Discord token'ları yalnızca GitHub Secrets içinde tutulur. `state/ztf_seen.json` dosyasında yalnızca halka açık ZTF nesne kimlikleri (`ZTF...`) bulunur; kişisel veya gizli veri içermez.
* **Zamanlama:** Her Pazar `05:17 UTC` (Türkiye saatiyle `08:17`). Küsüratlı dakika seçimi GitHub Actions saat başı kuyruk gecikmesini önler.

---

## 2. Klasör Yapısı

```text
watcher-engine/
├── .github/
│   └── workflows/
│       └── ztf_sunday.yml
├── core/
│   ├── __init__.py
│   ├── state.py
│   └── notifier.py
├── pipelines/
│   └── ztf_hunter.py
├── state/
│   └── ztf_seen.json
└── requirements.txt
```

Başlangıçta `state/ztf_seen.json` dosyasının içeriği yalnızca `[]` olmalıdır.

---

## 3. GitHub Repository Secrets

Repo ayarlarında **Settings - Secrets and variables - Actions** altına aşağıdaki anahtarları ekle:

| Secret Adı | Açıklama |
| :--- | :--- |
| `TELEGRAM_BOT_TOKEN` | BotFather üzerinden alınan Telegram bot token'ı |
| `TELEGRAM_CHAT_ID` | Raporun düşeceği kişisel sohbet veya özel kanal ID'si |

---

## 4. Bağımlılıklar (`requirements.txt`)

```txt
alerce>=1.1.0
astropy>=6.0.0
pandas>=2.1.0
requests>=2.31.0
```

---

## 5. Ortak Çekirdek Modüller

### `core/state.py`
```python
import json
from pathlib import Path
from typing import List, Set


def load_seen_ids(filepath: str) -> Set[str]:
    path = Path(filepath)
    if not path.exists():
        return set()
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return set(data) if isinstance(data, list) else set()


def save_seen_ids(filepath: str, seen_ids: Set[str], max_items: int = 5000) -> None:
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Dosya boyutunun sınırsız büyümesini engellemek için son N kaydı tut
    ordered: List[str] = sorted(seen_ids)[-max_items:]
    with path.open("w", encoding="utf-8") as f:
        json.dump(ordered, f, indent=2)
```

### `core/notifier.py`
```python
import os
import time
import requests


def send_telegram_message(text: str, retries: int = 3) -> None:
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        raise RuntimeError("TELEGRAM_BOT_TOKEN veya TELEGRAM_CHAT_ID tanımlı değil.")

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "Markdown",
        "disable_web_page_preview": True,
    }

    for attempt in range(1, retries + 1):
        resp = requests.post(url, json=payload, timeout=15)
        if resp.status_code == 200:
            return
        if attempt < retries:
            time.sleep(2 ** attempt)
    resp.raise_for_status()
```

---

## 6. Ana Tarama Betiği (`pipelines/ztf_hunter.py`)

Bu betik şu 4 filtreyi sırayla uygular:
1. Son 10 günde tespit edilmiş (`firstmjd`), en az 2 farklı gözlemde görülmüş (`ndet >= 2`) ve `stamp_classifier` süpernova (`SN`) olasılığı $0.70$ üzerinde olan nesneleri çeker.
2. `state/ztf_seen.json` ile daha önce incelenenleri eler.
3. `catshtm_conesearch` üzerinden GAIA kataloğuna sorarak 2 yay saniyesi (arcsec) yarıçapında bilinen parlak yıldız eşleşmesi olan yanlış pozitifleri düşürür.
4. Işık eğrisinden parlaklık yükselme eğimini ($\Delta m / \Delta t$) hesaplayıp en hızlı parlayan ilk 5 adayı Pazar raporuna yazar.

```python
import sys
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional

from alerce.core import Alerce
from astropy.time import Time

from core.notifier import send_telegram_message
from core.state import load_seen_ids, save_seen_ids

STATE_FILE = "state/ztf_seen.json"
LOOKBACK_DAYS = 10
MIN_NDET = 2
MIN_PROB = 0.70
MAX_REPORT_ITEMS = 5


def current_mjd() -> float:
    return float(Time(datetime.now(timezone.utc)).mjd)


def compute_rise_rate(client: Alerce, oid: str) -> Optional[float]:
    """Işık eğrisindeki ilk ve en parlak nokta arasındaki mag/gün eğimini hesaplar.
    Negatif değer parlaklığın arttığını gösterir (astronomide düşük mag daha parlaktır).
    """
    try:
        lc = client.query_detections(oid, format="pandas")
        if lc is None or lc.empty or len(lc) < 2:
            return None
        lc = lc.sort_values("mjd")
        first_row = lc.iloc[0]
        brightest_row = lc.loc[lc["magpsf"].idxmin()]
        dt = float(brightest_row["mjd"] - first_row["mjd"])
        dm = float(brightest_row["magpsf"] - first_row["magpsf"])
        if dt <= 0.05:
            return 0.0
        return dm / dt
    except Exception:
        return None


def has_gaia_star_match(client: Alerce, ra: float, dec: float, radius_arcsec: float = 2.0) -> bool:
    """Koordinatta bilinen bir GAIA yıldızı varsa True döner."""
    try:
        res = client.catshtm_conesearch(
            ra=ra,
            dec=dec,
            radius=radius_arcsec,
            catalog_name="GAIA/DR2",
            format="json",
        )
        return bool(res)
    except Exception:
        # Katalog servisi geçici hata verirse adayı kaybetmemek için eleme yapma
        return False


def run_ztf_pipeline() -> None:
    client = Alerce()
    seen_ids = load_seen_ids(STATE_FILE)
    mjd_now = current_mjd()
    mjd_start = mjd_now - LOOKBACK_DAYS

    # 1. ALeRCE sorgusu (exponential backoff ile)
    df = None
    for attempt in range(1, 4):
        try:
            df = client.query_objects(
                classifier="stamp_classifier",
                class_name="SN",
                probability=MIN_PROB,
                ndet=[MIN_NDET, 100],
                firstmjd=[mjd_start, mjd_now],
                order_by="probability",
                order_mode="DESC",
                page_size=50,
                format="pandas",
            )
            break
        except Exception as exc:
            if attempt == 3:
                raise RuntimeError(f"ALeRCE sorgusu başarısız: {exc}") from exc
            time.sleep(2 ** attempt)

    if df is None or df.empty:
        send_telegram_message("🔭 *ZTF Pazar Raporu:* Bu hafta eşik üstü yeni aday bulunmadı.")
        return

    candidates: List[Dict] = []
    newly_processed = set()

    for _, row in df.iterrows():
        oid = str(row["oid"])
        if oid in seen_ids:
            continue
        newly_processed.add(oid)

        ra = float(row["meanra"])
        dec = float(row["meandec"])
        prob = float(row.get("probability", 0.0))
        ndet = int(row.get("ndet", 0))

        # 2. Bilinen yıldız elemesi
        if has_gaia_star_match(client, ra, dec):
            continue

        # 3. Parlaklık yükselme eğimi
        slope = compute_rise_rate(client, oid)
        candidates.append(
            {
                "oid": oid,
                "ra": ra,
                "dec": dec,
                "prob": prob,
                "ndet": ndet,
                "slope": slope if slope is not None else 0.0,
            }
        )

    # En hızlı parlayanları (en negatif slope) ve yüksek olasılıklıları öne al
    candidates.sort(key=lambda x: (x["slope"], -x["prob"]))
    top_hits = candidates[:MAX_REPORT_ITEMS]

    if not top_hits:
        send_telegram_message("🔭 *ZTF Pazar Raporu:* Yeni adayların tamamı katalog filtresinde elendi.")
    else:
        lines = [f"🔭 *ZTF Pazar Raporu ({len(top_hits)} Aday)*\n"]
        for idx, item in enumerate(top_hits, start=1):
            url = f"https://alerce.online/object/{item['oid']}"
            lines.append(
                f"{idx}. `{item['oid']}` | Olasılık: `{item['prob']:.2f}` | Gözlem: `{item['ndet']}`\n"
                f"   Koordinat: `{item['ra']:.4f}, {item['dec']:.4f}` | Eğim: `{item['slope']:.2f} mag/g`\n"
                f"   [ALeRCE İncele]({url})"
            )
        send_telegram_message("\n".join(lines))

    # 4. State güncellemesi (sadece bildirim başarılıysa kaydedilir)
    seen_ids.update(newly_processed)
    save_seen_ids(STATE_FILE, seen_ids)


if __name__ == "__main__":
    try:
        run_ztf_pipeline()
    except Exception as err:
        print(f"HATA: {err}", file=sys.stderr)
        sys.exit(1)
```

---

## 7. GitHub Actions Workflow (`.github/workflows/ztf_sunday.yml`)

```yaml
name: ZTF Sunday Hunter

on:
  schedule:
    # Her Pazar 05:17 UTC (08:17 TR)
    - cron: '17 5 * * 0'
  workflow_dispatch:

permissions:
  contents: write

jobs:
  run-ztf:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout repository
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
          cache: 'pip'

      - name: Install dependencies
        run: pip install -r requirements.txt

      - name: Run ZTF Pipeline
        env:
          PYTHONPATH: .
          TELEGRAM_BOT_TOKEN: ${{ secrets.TELEGRAM_BOT_TOKEN }}
          TELEGRAM_CHAT_ID: ${{ secrets.TELEGRAM_CHAT_ID }}
        run: python pipelines/ztf_hunter.py

      - name: Commit updated state
        run: |
          git config user.name "github-actions[bot]"
          git config user.email "github-actions[bot]@users.noreply.github.com"
          git add state/ztf_seen.json
          if ! git diff --cached --quiet; then
            git commit -m "chore(state): update ZTF seen OIDs [skip ci]"
            git pull --rebase
            git push
          fi
```

---

## 8. 25 Dakikalık Kurulum ve Doğrulama Listesi

1. Public repoyu oluştur ve yukarıdaki klasör yapısını yerleştir (`state/ztf_seen.json` içi `[]`).
2. `TELEGRAM_BOT_TOKEN` ve `TELEGRAM_CHAT_ID` değerlerini GitHub Secrets'a gir.
3. GitHub arayüzünde **Actions - ZTF Sunday Hunter - Run workflow** düğmesine bas.
4. **Doğrulama:** Telegram'a rapor mesajının düştüğünü ve repoda `chore(state): update ZTF seen OIDs [skip ci]` commit'inin oluştuğunu kontrol et.