"""
Telefon kayitlari icin tasinabilir transkript (offline USB):
  raw vs deep-filter.exe (DeepFilterNet binary) vs noisereduce

Kullanim (hedef PC, internetsiz):
  set HF_HUB_OFFLINE=1
  python code\transcribe_compare.py -i kayit.wav --denoise deepfilter --model large-v3-turbo --lang tr -o out
  python code\transcribe_compare.py -i ./kayitlar/ --denoise all --device cpu --compute int8

Notlar:
- deep-filter.exe yalnizca 48kHz wav ister -> script otomatik cevirir.
- faster-whisper modeli models/ altinda aranir (download_root).
- UVR5 pakete dahil degil (muzikli kayit yoksa gerekmez).
"""
import argparse
import os
import subprocess
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message=".*pynvml.*")
warnings.filterwarnings("ignore", message=".*nvidia-ml-py.*")

import logging
for _ln in ("torch", "torch.utils.flop_counter"):
    logging.getLogger(_ln).setLevel(logging.ERROR)

import numpy as np

BASE = Path(__file__).resolve().parents[1]  # proje kok klasoru (code/ bir ustu)
TOOLS_DF = BASE / "tools" / "deep-filter.exe"
MODELS_CT2 = BASE / "models" / "faster-whisper-large-v3-turbo"
MODEL_ALIASES = {
    "turbo": "large-v3-turbo",
    "yeni": str(BASE / "models" / "whisper-tr-phone-ct2"),
    "e2": str(BASE / "models" / "whisper-tr-e2-ct2"),
    "turbo-tr": str(BASE / "models" / "whisper-tr-turbo-ct2"),
    "tr-phone": str(BASE / "models" / "whisper-tr-phone-ct2"),
    "large-v3": "large-v3",
}


def resolve_model(name: str) -> str:
    return MODEL_ALIASES.get(name, name)


def _run_quiet(cmd: list) -> None:
    """Harici programi penceresiz + sessiz calistir (deep-filter/ffmpeg konsol acmasin)."""
    subprocess.run(cmd, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                   creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))


def to_wav(src: Path, dst: Path, sr: int, loudnorm: bool = False) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    filt = ["-ac", "1", "-ar", str(sr)]
    if loudnorm:
        filt = ["-filter_complex", "loudnorm=I=-16:TP=-1.5:LRA=11", "-ac", "1", "-ar", str(sr)]
    cmd = ["ffmpeg", "-y", "-v", "error", "-i", str(src)] + filt + ["-c:a", "pcm_s16le", str(dst)]
    _run_quiet(cmd)
    return dst


def run_deepfilter_binary(inp16k: Path, outdir: Path) -> Path:
    """tools/deep-filter.exe kullanir. 48kHz ister."""
    if not TOOLS_DF.exists():
        raise FileNotFoundError(f"Bulunamadi: {TOOLS_DF}")
    outdir.mkdir(parents=True, exist_ok=True)
    wav48 = outdir / (inp16k.stem + "_48k.wav")
    to_wav(inp16k, wav48, 48000)
    before = set(outdir.glob("*.wav"))
    cmd = [str(TOOLS_DF), str(wav48), "-o", str(outdir), "--pf"]
    _run_quiet(cmd)
    # binary ciktiyi ayni isimle outdir'e yazar (uzerine yazar)
    out48 = outdir / wav48.name
    if not out48.exists():
        after = [c for c in outdir.glob("*.wav") if c not in before]
        if not after:
            raise FileNotFoundError(f"DeepFilter ciktisi yok: {outdir}")
        out48 = max(after, key=lambda p: p.stat().st_mtime)
    return to_wav(out48, outdir / (inp16k.stem + "_df16k.wav"), 16000)


def run_noisereduce(inp16k: Path, outdir: Path) -> Path:
    import librosa
    import soundfile as sf
    import noisereduce as nr
    outdir.mkdir(parents=True, exist_ok=True)
    y, sr = librosa.load(str(inp16k), sr=16000, mono=True)
    clean = nr.reduce_noise(y=y, sr=sr, prop_decrease=0.8)
    out = outdir / (inp16k.stem + "_nr16k.wav")
    sf.write(str(out), clean, sr, subtype="PCM_16")
    return out


DIAR_LOCAL = BASE / "models" / "pyannote-diar-3.1"


def fmt_ts(s: float) -> str:
    s = max(0.0, float(s))
    h, r = divmod(int(s * 1000), 3600000)
    m, r = divmod(r, 60000)
    sec, milli = divmod(r, 1000)
    return f"{h:02d}:{m:02d}:{sec:02d}.{milli:03d}"


def run_diarization(wav16k: Path, num_speakers: int | None = None):
    """pyannote diarization. Donus: [(start, end, speaker), ...]. Offline USB oncelikli."""
    hf_cache = BASE / "models" / "hf-cache"
    if hf_cache.exists():
        os.environ["HF_HOME"] = str(hf_cache.parent / "hf-home")
        os.environ["HUGGINGFACE_HUB_CACHE"] = str(hf_cache)
    if os.environ.get("HF_HUB_OFFLINE") == "1":
        os.environ["HF_HUB_OFFLINE"] = "1"
    # torchcodec, FFmpeg paylasimli DLL'lerini ister (static ffmpeg yetmez).
    # PyAV wheel'i bunlari av.libs ile getirir -> DLL arama yoluna ekle.
    try:
        import site
        for _sp in site.getsitepackages() + [os.path.dirname(sys.executable)]:
            _av = Path(_sp) / "av.libs"
            if _av.is_dir():
                os.add_dll_directory(str(_av))
                os.environ["PATH"] = str(_av) + os.pathsep + os.environ["PATH"]
                break
    except Exception:
        pass
    from pyannote.audio import Pipeline
    kw = {}
    if DIAR_LOCAL.exists():
        pipe = Pipeline.from_pretrained(str(DIAR_LOCAL))
    else:
        pipe = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
    if num_speakers:
        kw["num_speakers"] = num_speakers
    # torchcodec FFmpeg DLL istedigi icin onu atliyoruz:
    # wav'i soundfile ile yukleyip tensör olarak veriyoruz.
    import soundfile as sf
    import torch
    y, sr = sf.read(str(wav16k), dtype="float32", always_2d=True)
    waveform = torch.from_numpy(y.T.copy())
    diar = pipe({"waveform": waveform, "sample_rate": int(sr)}, **kw)
    # pyannote 4.x DiarizeOutput doner (eski surumde direkt Annotation)
    ann = getattr(diar, "exclusive_speaker_diarization",
                  getattr(diar, "speaker_diarization", diar))
    turns = []
    for turn, _, speaker in ann.itertracks(yield_label=True):
        turns.append((float(turn.start), float(turn.end), str(speaker)))
    # cakisma bolgeleri: exclusive olmayan haritadan ayni anda konusanlar
    overlaps = []
    try:
        full = getattr(diar, "speaker_diarization", None)
        if full is not None and full is not ann:
            segs = [(float(t.start), float(t.end)) for t, _, _ in full.itertracks(yield_label=True)]
            segs.sort()
            for k in range(1, len(segs)):
                ps, pe = segs[k - 1]
                cs, ce = segs[k]
                if cs < pe - 0.1 and ce > cs + 0.3:
                    overlaps.append((max(ps, cs), min(pe, ce)))
    except Exception:
        pass
    return turns, overlaps


def tag_speakers(segments, turns):
    """Her whisper segmentine en cok ortusen konusmaciyi ata."""
    out = []
    for s in segments:
        best, best_ov = "SPEAKER_??", 0.0
        for ts, te, spk in turns:
            ov = max(0.0, min(s.end, te) - max(s.start, ts))
            if ov > best_ov:
                best, best_ov = spk, ov
        out.append((s, best))
    return out


def safe_write(path: Path, text: str) -> Path:
    """Uzerine yazmadan once mevcut dosyayi zaman damgali yedekle."""
    if path.exists():
        import datetime
        stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        bak = path.with_name(path.name + f".bak-{stamp}")
        bak.write_bytes(path.read_bytes())
    path.write_text(text, encoding="utf-8")
    return path


def write_spk_txt(path: Path, tagged):
    lines = []
    for s, spk in tagged:
        end = max(float(s.end), float(s.start) + 0.2)  # sifir sureli damga olmasin
        lines.append(f"[{fmt_ts(s.start)} - {fmt_ts(end)}] {spk}: {s.text.strip()}")
    safe_write(path, "\n".join(lines) + "\n")
    return path


def transcribe_faster_whisper(wav: Path, model: str, lang: str, device: str, compute: str,
                          vad_threshold: float = 0.5, min_silence: int = 500,
                          speech_pad: int = 200, no_speech: float = 0.6,
                          beam: int = 5, condition: bool = False,
                          prompt: str | None = None, hotwords: str | None = None):
    if device == "cuda":
        try:
            import torch  # noqa: F401  -- torch/lib CUDA DLL'lerini yukler (ctranslate2 icin sart)
        except ImportError:
            pass
    from faster_whisper import WhisperModel
    model = resolve_model(model)
    kw = dict(device=device, compute_type=compute)
    cache = BASE / "models"
    if (cache / "models--mobiuslabsgmbh--faster-whisper-large-v3-turbo").exists():
        kw["download_root"] = str(cache)
    if os.environ.get("HF_HUB_OFFLINE") == "1":
        kw["local_files_only"] = True
    m = WhisperModel(model, **kw)
    segments, info = m.transcribe(
        str(wav),
        language=lang,
        beam_size=beam,
        vad_filter=True,
        vad_parameters=dict(threshold=vad_threshold,
                            min_silence_duration_ms=min_silence,
                            speech_pad_ms=speech_pad),
        condition_on_previous_text=condition,
        no_speech_threshold=no_speech,
        compression_ratio_threshold=2.4,
        initial_prompt=prompt or "Bu bir telefon gorusmesi kaydi. Turkce, dogal konusma dili.",
        hotwords=hotwords,
    )
    segs = list(segments)
    text = " ".join(s.text.strip() for s in segs)
    avg_logprob = float(np.mean([s.avg_logprob for s in segs])) if segs else -99.0
    no_speech = float(np.mean([s.no_speech_prob for s in segs])) if segs else 1.0
    dur = sum(s.end - s.start for s in segs) if segs else 0.0
    return {"text": text, "segments": segs, "avg_logprob": avg_logprob,
            "no_speech_prob": no_speech, "speech_dur": dur,
            "detected_lang": getattr(info, "language", lang)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True, help="wav/mp3/m4a dosya veya klasor")
    ap.add_argument("--model", default="turbo-tr",
                    help="turbo-tr (=egitilmis, varsayilan) | e2 | yeni | turbo | large-v3 | CT2 klasor yolu")
    ap.add_argument("--lang", default="tr")
    ap.add_argument("--denoise", default="deepfilter",
                    choices=["all", "deepfilter", "noisereduce", "raw"])
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--compute", default="float16",
                    choices=["float16", "int8_float16", "int8"])
    ap.add_argument("-o", "--out", default="./out")
    ap.add_argument("--diarize", action="store_true",
                    help="pyannote ile konusmaci ayrimi (SPEAKER_00/01 etiketli .spk.txt yazar)")
    ap.add_argument("--speakers", type=int, default=None,
                    help="Konusmaci sayisi ipucu (telefon icin 2). Bos birakilirsa otomatik.")
    ap.add_argument("--vad-threshold", type=float, default=0.5,
                    help="VAD esigi (dusuk=duyarli: 0.3)")
    ap.add_argument("--min-silence", type=int, default=500,
                    help="ms: konusma arasi sessizlik (kisa tut: 200)")
    ap.add_argument("--speech-pad", type=int, default=200,
                    help="ms: konusma kenar payi (genis tut: 400)")
    ap.add_argument("--no-speech", type=float, default=0.6,
                    help="Ustu atilan sessizlik olasiligi (duyarli: 0.8)")
    ap.add_argument("--beam", type=int, default=5, help="Isin genisligi (yavas ama dikkatli: 10)")
    ap.add_argument("--condition", action="store_true",
                    help="Onceki cumle baglamini kullan (uzun konusmalarda tutarlilik)")
    ap.add_argument("--prompt", type=str, default=None, help="Ozel baslangic ipucu")
    ap.add_argument("--loudnorm", action=argparse.BooleanOptionalAction, default=True,
                    help="Girisi -16 LUFS'a normalize et (varsayilan acik, --no-loudnorm ile kapat)")
    ap.add_argument("--hotwords", type=str, default=None,
                    help="Virgullu kelime listesi, modele oncelik verdirir. Orn: Dörtdivan,Karabük,komiserim")
    args = ap.parse_args()

    in_path = Path(args.input)
    files = [in_path] if in_path.is_file() else sorted(
        [p for p in in_path.iterdir() if p.suffix.lower() in (".wav", ".mp3", ".m4a", ".ogg", ".flac")])
    if not files:
        print("Ses dosyasi bulunamadi.", file=sys.stderr)
        sys.exit(1)

    outdir = Path(args.out)
    (outdir / "norm").mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    for f in files:
        print(f"\n=== {f.name} ===")
        norm = to_wav(f, outdir / "norm" / (f.stem + "_16k.wav"), 16000,
                      loudnorm=args.loudnorm)
        branches = {}
        if args.denoise in ("all", "raw"):
            branches["raw"] = norm
        if args.denoise in ("all", "deepfilter"):
            try:
                t0 = time.time()
                df_16k = run_deepfilter_binary(norm, outdir / "df" / f.stem)
                branches["deepfilter"] = df_16k
                print(f"[deepfilter] {time.time()-t0:.1f}s -> {df_16k.name}")
            except Exception as e:
                print(f"[deepfilter] atlandi: {e}")
        if args.denoise in ("all", "noisereduce"):
            try:
                t0 = time.time()
                nr_16k = run_noisereduce(norm, outdir / "nr" / f.stem)
                branches["noisereduce"] = nr_16k
                print(f"[noisereduce] {time.time()-t0:.1f}s -> {nr_16k.name}")
            except Exception as e:
                print(f"[noisereduce] atlandi: {e}")

        turns, overlaps = [], []
        if args.diarize:
            try:
                t1 = time.time()
                turns, overlaps = run_diarization(norm, args.speakers)
                spks = sorted({s for _, _, s in turns})
                print(f"[diarize] {time.time()-t1:.1f}s -> {len(turns)} donus, konusmacilar: {spks}, cakisma: {len(overlaps)}")
                if overlaps:
                    ov_path = outdir / (f.stem + ".overlap.txt")
                    safe_write(ov_path, "\n".join(
                        f"[{fmt_ts(a)} - {fmt_ts(b)}] USTUSTE KONUSMA - dinle" for a, b in overlaps) + "\n")
                    print(f"  -> {ov_path}")
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"[diarize] atlandi: {e}")

        for name, wav in branches.items():
            t0 = time.time()
            r = transcribe_faster_whisper(wav, args.model, args.lang, args.device, args.compute,
                                        args.vad_threshold, args.min_silence,
                                        args.speech_pad, args.no_speech,
                                        args.beam, args.condition, args.prompt, args.hotwords)
            dt = time.time() - t0
            txt_path = outdir / f"{f.stem}.{name}.tr.txt"
            safe_write(txt_path, r["text"])
            print(f"[{name}] {dt:.1f}s logprob={r['avg_logprob']:.2f} "
                  f"no_speech={r['no_speech_prob']:.2f} konusma={r['speech_dur']:.1f}s")
            print(f"  -> {txt_path}")
            print(f"  ORN: {r['text'][:220]}...")
            if turns:
                tagged = tag_speakers(r["segments"], turns)
                spk_path = outdir / f"{f.stem}.{name}.spk.txt"
                write_spk_txt(spk_path, tagged)
                print(f"  -> {spk_path}")


if __name__ == "__main__":
    main()
