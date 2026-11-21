"""Egitim klasorlerindeki (ses + duzeltilmis .txt) ciftlerinden ASR veri seti kurar.
Cikti: dataset/wavs/*.wav (16kHz mono) + train.jsonl / val.jsonl (audio,text)
30sn'den uzun kayitlar cumle sinirinda ~25sn'lik parcaclara bolunur.
Kullanim: python code\\build_dataset.py
"""
import argparse
import json
import re
import subprocess
from pathlib import Path

SRC_DIRS: list = [
    # Ornek: (r"C:\Ses\Egitim", "E1"),
    # Komut satirindan verilir: --src E1=C:/Ses/Egitim (birden fazla verilebilir)
]
DST = Path(__file__).resolve().parents[1] / "dataset"
WAVDIR = DST / "wavs"
VAL_NAMES = {"E1_11", "E1_13", "E1_18",
             "E2_1d20260420150507p+905437843275.mp3",
             "E2_1d20260520124208pnull.mp3"}  # kok = prefix_mpga-stem (icinde .mp3 olabilir)

LINE_RE = re.compile(r"^\[(\d+:\d+:\d+(?:\.\d+)?)\s*-\s*(\d+:\d+:\d+(?:\.\d+)?)\]\s*\d+\.Şahıs\s?:\s?(.*)$")


def ts(s: str) -> float:
    h, m, sec = s.split(":")
    return int(h) * 3600 + int(m) * 60 + float(sec)


def to_wav(src: Path, dst: Path, sr: int = 16000):
    if dst.exists() and dst.stat().st_size > 1000:
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(src),
                    "-ac", "1", "-ar", str(sr), "-c:a", "pcm_s16le", str(dst)], check=True)
    return dst


def parse_txt(p: Path):
    segs = []
    for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
        m = LINE_RE.match(line.strip())
        if not m:
            continue
        t1, t2, txt = m.group(1), m.group(2), m.group(3).strip()
        if txt:
            segs.append((ts(t1), ts(t2), txt))
    return segs


def cut(src_wav: Path, start: float, end: float, dst: Path):
    if dst.exists() and dst.stat().st_size > 1000:
        return
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", str(start), "-t", str(end - start),
                    "-i", str(src_wav), "-c:a", "pcm_s16le", str(dst)], check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["all", "E1", "E2"], default="all")
    ap.add_argument("--src", action="append", default=[],
                    help="Orn: --src E1=C:/Ses/Egitim (birden fazla verilebilir)")
    args = ap.parse_args()
    srcs = SRC_DIRS
    if args.src:
        srcs = []
        for _s in args.src:
            tag, _, _p = _s.partition("=")
            srcs.append((_p.strip(), tag.strip() or "D"))
    if not srcs:
        ap.error("kaynak yok: --src TAG=Klasor ile en az bir veri klasoru verin")
    sel = [s for s in srcs if args.only == "all" or s[1] == args.only]
    WAVDIR.mkdir(parents=True, exist_ok=True)
    train, val = [], []
    n_short = n_chunk = 0
    for src_dir, prefix in sel:
        SRC = Path(src_dir)
        for mpga in sorted(SRC.glob("*.MPGA")):
            stem = f"{prefix}_{mpga.stem}"
            txt = SRC / (mpga.stem + ".txt")
            if not txt.exists():
                print(f"ATLANDI (txt yok): {stem}")
                continue
            segs = parse_txt(txt)
            if not segs:
                print(f"ATLANDI (bos): {stem}")
                continue
            full = WAVDIR / (stem + ".full.wav")
            to_wav(mpga, full)
            target = val if stem in VAL_NAMES else train
            cur, cstart, cend, ctext = [], None, None, []

            def flush():
                nonlocal cur, cstart, cend, ctext, n_short, n_chunk
                if not cur:
                    return
                text = " ".join(ctext)
                if len(text) > 1500:
                    print(f"ATLANDI (uzun): {stem} @{cstart:.0f} ({len(text)} karakter)")
                    cur, cstart, cend, ctext = [], None, None, []
                    return
                if len(cur) == len(segs) and (cend - cstart) <= 30.5:
                    (val if stem in VAL_NAMES else train).append(
                        {"audio": str(full), "text": text})
                    n_short += 1
                else:
                    out = WAVDIR / f"{stem}_{cstart:.0f}_{cend:.0f}.wav"
                    cut(full, cstart, min(cend, cstart + 30), out)
                    (val if stem in VAL_NAMES else train).append(
                        {"audio": str(out), "text": " ".join(ctext)})
                    n_chunk += 1
                cur, cstart, cend, ctext = [], None, None, []

            for s, e, t in segs:
                if cstart is None:
                    cstart = s
                if e - cstart > 29.0 and cur:
                    flush()
                    cstart = s
                cur.append((s, e, t))
                cend = e
                ctext.append(t)
            flush()
    for name, rows in (("train", train), ("val", val)):
        with open(DST / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"train: {len(train)} parca | val: {len(val)} parca | tek: {n_short} bolunmus: {n_chunk}")
    print(f"cikti: {DST}")


if __name__ == "__main__":
    main()
