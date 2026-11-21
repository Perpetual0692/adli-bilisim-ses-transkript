"""Egitilmis modeli turbo + Buzz ile val setinde karsilastir.
Kullanim (venv): python code\\eval_models.py
Cikti: D:\\Adli Bilişim\\Ses\\eval-sonuc.txt + konsol tablosu
"""
import json
import re
import unicodedata
from pathlib import Path

BASE = Path("D:/Adli Bilişim/Ses")
MODELS = {
    "turbo": "large-v3-turbo",
    "yeni": str(BASE / "models/whisper-tr-phone-ct2"),
    "e2": str(BASE / "models/whisper-tr-e2-ct2"),
    "turbo-tr": str(BASE / "models/whisper-tr-turbo-ct2"),
}
CACHE = BASE / "models"


def norm(s: str) -> str:
    s = re.sub(r"^\s*\[.*?\]+\s*(?:SPEAKER_\S+|[^:]{1,30}):\s?", "", s, flags=re.M)
    s = "".join(c for c in unicodedata.normalize("NFKC", s.lower()) if not unicodedata.category(c).startswith("P"))
    return re.sub(r"\s+", " ", s).strip().split()


def lev(a, b):
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        ai = a[i - 1]
        for j in range(1, m + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ai != b[j - 1]))
        prev = cur
    return prev[m]


def main():
    import torch  # noqa: F401
    from faster_whisper import WhisperModel
    rows = [json.loads(l) for l in open(BASE / "dataset/val.jsonl", encoding="utf-8") if l.strip()]
    results = {}
    for name, mid in MODELS.items():
        kw = dict(device="cuda", compute_type="float16")
        if (CACHE / "models--mobiuslabsgmbh--faster-whisper-large-v3-turbo").exists():
            kw["download_root"] = str(CACHE)
        m = WhisperModel(mid, **kw)
        tot_err, tot_n, tot_cerr, tot_cn = 0, 0, 0, 0
        for r in rows:
            segs, _ = m.transcribe(r["audio"], language="tr", beam_size=5,
                                   vad_filter=True, condition_on_previous_text=False)
            hyp = " ".join(s.text.strip() for s in segs)
            rw, hw = norm(r["text"]), norm(hyp)
            rc, hc = "".join(rw).replace(" ", ""), "".join(hw).replace(" ", "")
            tot_err += lev(rw, hw)
            tot_n += max(1, len(rw))
            tot_cerr += lev(rc, hc)
            tot_cn += max(1, len(rc))
        results[name] = (tot_err / tot_n, tot_cerr / tot_cn)
        print(f"{name}: WER={results[name][0]:.3f} CER={results[name][1]:.3f}", flush=True)
    with open(BASE / "eval-sonuc.txt", "w", encoding="utf-8") as f:
        for name, (w, c) in sorted(results.items(), key=lambda x: x[1][0]):
            f.write(f"{name}: WER={w:.3f} CER={c:.3f}\n")
    print("yazildi: eval-sonuc.txt")


if __name__ == "__main__":
    main()
