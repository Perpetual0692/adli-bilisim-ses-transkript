"""Referans metne karsi WER/CER skorla (duzeltilmis transkript = hakem).
Kullanim:
  python score_wer.py -r referans.txt -i hipotez.txt
  python score_wer.py -r referans.txt -d ./ab-test/   (klasordeki tum .tr.txt)
Turkce normalize: kucuk harf, noktalama sil, sayilar oldugu gibi.
"""
import argparse
import re
import unicodedata
from pathlib import Path


SPK_PREFIX = re.compile(r"^\s*\[.*?\]+\s*(?:SPEAKER_\S+|[^:]{1,30}):\s?", re.MULTILINE)


def norm(s: str) -> str:
    s = SPK_PREFIX.sub("", s)  # "[00:00:01 - ...] SPEAKER_00:" oneklerini at
    s = s.lower()
    s = "".join(c for c in unicodedata.normalize("NFKC", s) if not unicodedata.category(c).startswith("P"))
    return re.sub(r"\s+", " ", s).strip()


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


def score(ref: str, hyp: str):
    rw, hw = norm(ref).split(), norm(hyp).split()
    wer = lev(rw, hw) / max(1, len(rw))
    rc, hc = norm(ref).replace(" ", ""), norm(hyp).replace(" ", "")
    cer = lev(rc, hc) / max(1, len(rc))
    return wer, cer, len(rw), len(hw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-r", "--ref", required=True, help="duzeltilmis referans txt")
    ap.add_argument("-i", "--hyp", default=None, help="tek hipotez txt")
    ap.add_argument("-d", "--dir", default=None, help="hipotez klasoru (tum .tr.txt)")
    args = ap.parse_args()
    ref = Path(args.ref).read_text(encoding="utf-8")
    hyps = [Path(args.hyp)] if args.hyp else sorted(Path(args.dir).rglob("*.tr.txt"))
    if not hyps:
        print("hipotez yok")
        return
    rows = []
    for h in hyps:
        try:
            hyp = h.read_text(encoding="utf-8")
        except Exception as e:
            print(f"atlandi {h}: {e}")
            continue
        wer, cer, nw, hw = score(ref, hyp)
        rows.append((wer, cer, nw, hw, str(h)))
    rows.sort()
    print(f"{'WER':>7} {'CER':>7} {'ref_kelime':>10} {'hyp_kelime':>10}  dosya")
    for wer, cer, nw, hw, h in rows:
        print(f"{wer:7.3f} {cer:7.3f} {nw:10d} {hw:10d}  {h}")


if __name__ == "__main__":
    main()
