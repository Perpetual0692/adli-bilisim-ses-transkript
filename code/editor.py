"""Konusmaci etiketli transkript duzeltme arayuzu (CustomTkinter, koyu tema).
Kullanim:  python editor.py [klasor | dosya.spk.txt]
Gereken:   customtkinter (venv'de kurulu), ffmpeg (dinleme icin PATH'te).
"""
import re
import subprocess
import sys
import threading
import tkinter as tk
import tkinter.filedialog
import tkinter.messagebox
from pathlib import Path

import customtkinter as ctk

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

LINE_RE = re.compile(r"^\[(\d+:\d+:\d+(?:\.\d+)?)\s*-\s*(\d+:\d+:\d+(?:\.\d+)?)\]\s*(SPEAKER_\S+|[^:]+):\s?(.*)")

SPK_COLORS = ["#38bdf8", "#fb923c", "#34d399", "#f472b6",
              "#a78bfa", "#facc15", "#2dd4bf", "#fb7185"]

BG_TEXT = "#1a1b26"
FG_TEXT = "#e8e8ef"


def ts_to_sec(s: str) -> float:
    h, m, sec = s.split(":")
    return int(h) * 3600 + int(m) * 60 + float(sec)


def sec_to_ts(s: float, ms: bool = True) -> str:
    s = max(0.0, float(s))
    h, r = divmod(int(s * 1000), 3600000)
    m, r = divmod(r, 60000)
    sec, milli = divmod(r, 1000)
    base = f"{h:02d}:{m:02d}:{sec:02d}"
    return f"{base}.{milli:03d}" if ms else base


class Editor(ctk.CTk):
    def __init__(self, start_path: Path | None = None):
        super().__init__()
        self.title("Ses Transkript Duzeltme")
        self.geometry("1200x760")
        self.minsize(900, 600)
        self.wav: Path | None = None
        self.file: Path | None = None
        self.sel_idx: int | None = None
        self.seg_btns: list = []
        self.seg_starts: list = []
        self.spk_index: dict[str, int] = {}
        self._sync_job = None
        self._build()
        if start_path:
            if start_path.is_dir():
                self.open_folder(start_path)
            else:
                self.open_file(start_path)

    # ---------- arayuz ----------
    def _build(self):
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=0)
        self.grid_columnconfigure(1, weight=1)

        top = ctk.CTkFrame(self, corner_radius=0)
        top.grid(row=0, column=0, columnspan=2, sticky="ew", padx=8, pady=(8, 4))
        ctk.CTkButton(top, text="Klasor Ac", width=90, command=self.ask_folder).pack(side="left", padx=4, pady=6)
        ctk.CTkButton(top, text="Dosya Ac", width=90, command=self.ask_file).pack(side="left", padx=4, pady=6)
        ctk.CTkButton(top, text="Transkript Uret", width=120, fg_color="#047857", hover_color="#065f46",
                      command=self.open_transcribe).pack(side="left", padx=4, pady=6)
        ctk.CTkButton(top, text="Kaydet (Ctrl+S)", width=120, command=self.save).pack(side="left", padx=4, pady=6)
        ctk.CTkButton(top, text="Yenile", width=80, command=self.resync).pack(side="left", padx=4, pady=6)
        ctk.CTkButton(top, text="Durdur", width=80, fg_color="#7f1d1d", hover_color="#991b1b",
                      command=self.stop).pack(side="left", padx=4, pady=6)
        ctk.CTkLabel(top, text="Ara:").pack(side="left", padx=(12, 2))
        self.search = ctk.CTkEntry(top, width=170)
        self.search.pack(side="left", padx=2)
        self.search.bind("<Return>", lambda e: self.find_next())
        ctk.CTkButton(top, text="Sonraki (F3)", width=100, command=self.find_next).pack(side="left", padx=4, pady=6)
        self.status = ctk.CTkLabel(top, text="hazir", text_color="gray")
        self.status.pack(side="right", padx=8)

        left = ctk.CTkFrame(self, width=330)
        left.grid(row=1, column=0, sticky="nsew", padx=(8, 4), pady=4)
        left.grid_propagate(False)
        ctk.CTkLabel(left, text="Bolumler — tikla: git + dinle",
                     font=("Segoe UI", 11, "bold")).pack(anchor="w", padx=8, pady=(8, 2))
        self.seglist = ctk.CTkScrollableFrame(left, fg_color="transparent")
        self.seglist.pack(fill="both", expand=True, padx=4, pady=4)

        right = ctk.CTkFrame(self)
        right.grid(row=1, column=1, sticky="nsew", padx=(4, 8), pady=4)
        right.grid_rowconfigure(1, weight=1)
        right.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(right, text="Metin — saate tikla dinle, duzenle",
                     font=("Segoe UI", 11, "bold")).grid(row=0, column=0, sticky="w", padx=8, pady=(8, 2))
        txt_frame = ctk.CTkFrame(right, fg_color="transparent")
        txt_frame.grid(row=1, column=0, sticky="nsew", padx=4, pady=4)
        txt_frame.grid_rowconfigure(0, weight=1)
        txt_frame.grid_columnconfigure(0, weight=1)
        self.text = tk.Text(txt_frame, wrap="word", undo=True, font=("Consolas", 12),
                            bg=BG_TEXT, fg=FG_TEXT, insertbackground="white",
                            selectbackground="#1d4ed8", relief="flat", spacing1=2, spacing3=2)
        self.text.grid(row=0, column=0, sticky="nsew")
        sb = ctk.CTkScrollbar(txt_frame, command=self.text.yview)
        sb.grid(row=0, column=1, sticky="ns")
        self.text.configure(yscrollcommand=sb.set)
        self.text.bind("<<Modified>>", self._on_modified)

        bot = ctk.CTkFrame(self, corner_radius=0)
        bot.grid(row=2, column=0, columnspan=2, sticky="ew", padx=8, pady=(4, 8))
        ctk.CTkLabel(bot, text="Kisi:").pack(side="left", padx=(8, 2), pady=6)
        self.old_spk = ctk.CTkEntry(bot, width=110)
        self.old_spk.pack(side="left", padx=2)
        self.old_spk.insert(0, "SPEAKER_01")
        ctk.CTkLabel(bot, text="->").pack(side="left", padx=2)
        self.new_spk = ctk.CTkEntry(bot, width=130)
        self.new_spk.pack(side="left", padx=2)
        self.new_spk.insert(0, "Arayan")
        ctk.CTkButton(bot, text="Degistir", width=90, command=self.rename_speaker).pack(side="left", padx=4, pady=6)
        ctk.CTkLabel(bot, text="Secili:").pack(side="left", padx=(12, 2))
        ctk.CTkButton(bot, text="Bas -50", width=70, command=lambda: self.nudge_edge("start", -50)).pack(side="left", padx=2, pady=6)
        ctk.CTkButton(bot, text="Bas +50", width=70, command=lambda: self.nudge_edge("start", 50)).pack(side="left", padx=2, pady=6)
        ctk.CTkButton(bot, text="Bit -50", width=70, command=lambda: self.nudge_edge("end", -50)).pack(side="left", padx=2, pady=6)
        ctk.CTkButton(bot, text="Bit +50", width=70, command=lambda: self.nudge_edge("end", 50)).pack(side="left", padx=2, pady=6)
        self.ripple = tk.BooleanVar(value=True)
        ctk.CTkCheckBox(bot, text="Sonraki sinirla kaydir", variable=self.ripple).pack(side="left", padx=(10, 4))

        self.bind("<Control-s>", lambda e: self.save())
        self.bind("<Control-f>", lambda e: self.search.focus())
        self.bind("<F3>", lambda e: self.find_next())
        self.text.tag_configure("ts", foreground="#7d8590", underline=True)
        for i, col in enumerate(SPK_COLORS):
            self.text.tag_configure(f"spk{i}", foreground=col, font=("Consolas", 12, "bold"))

    # ---------- dosya ----------
    def ask_folder(self):
        d = tk.filedialog.askdirectory()
        if d:
            self.open_folder(Path(d))

    def ask_file(self):
        f = tk.filedialog.askopenfilename(filetypes=[("Transkript", "*.spk.txt *.tr.txt *.txt")])
        if f:
            self.open_file(Path(f))

    def open_folder(self, d: Path):
        txts = sorted(d.glob("*.spk.txt")) or sorted(d.glob("*.tr.txt"))
        if not txts:
            tk.messagebox.showinfo("Bilgi", f"{d} icinde .spk.txt / .tr.txt yok, klasor sec")
            nd = tk.filedialog.askdirectory()
            if nd:
                self.open_folder(Path(nd))
            return
        self.open_file(txts[0])

    def open_file(self, f: Path):
        self.file = f
        stem = f.name.split(".")[0]
        cands = []
        for _ext in ("*.wav", "*.mp3", "*.MPGA", "*.mpga", "*.m4a", "*.ogg", "*.flac"):
            cands += list(f.parent.glob(stem + _ext))
            cands += list((f.parent / "norm").glob(stem + _ext))
        self.wav = cands[0] if cands else None
        self.text.delete("1.0", "end")
        self.spk_index = {}
        for line in f.read_text(encoding="utf-8").splitlines():
            m = LINE_RE.match(line)
            if not m:
                self.text.insert("end", line + "\n")
                continue
            t1, t2, spk, txt = m.groups()
            t1 = sec_to_ts(ts_to_sec(t1))
            t2 = sec_to_ts(ts_to_sec(t2))
            if spk not in self.spk_index:
                self.spk_index[spk] = len(self.spk_index)
            tag = f"spk{self.spk_index[spk] % 8}"
            self.text.insert("end", f"[{t1} - {t2}] ", ("ts", f"ts:{t1}:{t2}"))
            self.text.tag_bind(f"ts:{t1}:{t2}", "<Button-1>",
                               lambda e, a=t1, b=t2: self.play(a, b))
            self.text.insert("end", f"{spk}: ", (tag,))
            self.text.insert("end", txt + "\n")
        self.sel_idx = None
        self.refresh_seglist()
        n = len(self.seg_starts)
        self.status.configure(text=f"{f.name} | {n} bolum | ses: {self.wav.name if self.wav else 'YOK'}")
        self.title(f"Ses Transkript Duzeltme - {f.name}")

    def on_seg_select(self, i: int):
        self.sel_idx = i
        for j, b in enumerate(self.seg_btns):
            b.configure(fg_color="#1d4ed8" if j == i else "#2b2b2b")
        if 0 <= i < len(self.seg_starts):
            t1, t2 = self.seg_starts[i]
            if t1:
                self.play(t1, t2)
                idx = self.text.search(t1, "1.0", stopindex="end")
                if idx:
                    self.text.see(idx)
                    self.text.mark_set("insert", idx)

    # ---------- ses ----------
    def play(self, t1: str, t2: str):
        self.stop()
        if not self.wav:
            tk.messagebox.showinfo("Bilgi", "Eslesen ses bulunamadi")
            return
        import tempfile
        import winsound
        s = ts_to_sec(t1)
        d = max(0.5, ts_to_sec(t2) - s + 0.4)
        seg = Path(tempfile.gettempdir()) / "editor_seg.wav"
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-v", "error", "-ss", str(s), "-t", str(d),
                 "-i", str(self.wav), "-ac", "1", "-ar", "16000",
                 "-c:a", "pcm_s16le", str(seg)],
                check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            winsound.PlaySound(str(seg), winsound.SND_FILENAME | winsound.SND_ASYNC)
            self.status.configure(text=f"caliyor: {t1} ({d:.1f}sn)")
        except FileNotFoundError:
            tk.messagebox.showerror("Hata", "ffmpeg PATH'te olmali")
        except Exception as e:
            tk.messagebox.showerror("Hata", str(e))

    def stop(self):
        try:
            import winsound
            winsound.PlaySound(None, winsound.SND_PURGE)
        except Exception:
            pass

    # ---------- arama / duzenleme ----------
    def find_next(self):
        q = self.search.get()
        if not q:
            return
        idx = self.text.search(q, "insert+1c", nocase=True, stopindex="end")
        if not idx:
            idx = self.text.search(q, "1.0", nocase=True, stopindex="end")
        if idx:
            self.text.see(idx)
            self.text.mark_set("insert", idx)
            self.text.tag_remove("sel", "1.0", "end")
            self.text.tag_add("sel", idx, f"{idx}+{len(q)}c")
        else:
            self.status.configure(text="bulunamadi")

    def rename_speaker(self):
        old = self.old_spk.get().strip()
        new = self.new_spk.get().strip()
        if not old or not new:
            return
        content = self.text.get("1.0", "end")
        n = content.count(old)
        self.text.delete("1.0", "end")
        self.text.insert("1.0", content.replace(old, new))
        self.resync()
        self.status.configure(text=f"{n} yerde {old} -> {new}")

    def shift_all(self, ms: int):
        out, n = [], 0
        for line in self.text.get("1.0", "end").splitlines():
            m = LINE_RE.match(line)
            if m:
                t1, t2, spk, txt = m.groups()
                t1 = sec_to_ts(ts_to_sec(t1) + ms / 1000)
                t2 = sec_to_ts(ts_to_sec(t2) + ms / 1000)
                line = f"[{t1} - {t2}] {spk}: {txt}"
                n += 1
            out.append(line)
        self.text.delete("1.0", "end")
        self.text.insert("1.0", "\n".join(out) + "\n")
        self.resync()

    def nudge_sel(self, ms: int):
        out, n = [], 0
        for line in self.text.get("1.0", "end").splitlines():
            m = LINE_RE.match(line)
            if m:
                t1, t2, spk, txt = m.groups()
                t1 = sec_to_ts(ts_to_sec(t1) + ms / 1000)
                t2 = sec_to_ts(ts_to_sec(t2) + ms / 1000)
                line = f"[{t1} - {t2}] {spk}: {txt}"
                n += 1
            out.append(line)
        self.text.delete("1.0", "end")
        self.text.insert("1.0", "\n".join(out) + "\n")
        self.refresh_seglist()
        self.status.configure(text=f"{n} satir {ms:+d}ms kaydirildi")

    def nudge_edge(self, which: str, ms: int):
        if self.sel_idx is None:
            # listeden secim yoksa metin imlecindeki satiri kullan
            try:
                cur = int(self.text.index("insert").split(".")[0]) - 1
            except Exception:
                self.status.configure(text="once soldan bolum sec")
                return
            lines = self.text.get("1.0", "end").splitlines()
            count = -1
            for li, line in enumerate(lines):
                if LINE_RE.match(line):
                    count += 1
                    if count == cur:
                        pass
            # imlecten satir bul:
            i = None
            count = -1
            for li, line in enumerate(lines):
                if LINE_RE.match(line):
                    count += 1
                    if li == int(self.text.index("insert").split(".")[0]) - 1:
                        i = count
                        break
            if i is None or i < 0:
                self.status.configure(text="once soldan bolum sec")
                return
        else:
            i = self.sel_idx
        lines = self.text.get("1.0", "end").splitlines()
        ts_lines = [(li, l) for li, l in enumerate(lines) if LINE_RE.match(l)]
        if i < 0 or i >= len(ts_lines):
            return
        li = ts_lines[i][0]
        m = LINE_RE.match(lines[li])
        assert m is not None
        t1, t2, spk, txt = m.groups()
        a, b = ts_to_sec(t1), ts_to_sec(t2)
        oa, ob = a, b
        if which == "start":
            a = max(0.0, a + ms / 1000)
            if a >= b:
                self.status.configure(text="baslangic bitisi gecemez")
                return
        else:
            b = max(a + 0.1, b + ms / 1000)
        repl = {li: (a, b, spk, txt)}
        if self.ripple.get():
            if which == "end":
                nxt = [x for x in ts_lines if x[0] > li]
                if nxt:
                    nli, nl = nxt[0]
                    nm = LINE_RE.match(nl)
                    assert nm is not None
                    na, nb, nspk, ntxt = ts_to_sec(nm.group(1)), ts_to_sec(nm.group(2)), nm.group(3), nm.group(4)
                    if abs(na - ob) < 0.001:
                        repl[nli] = (b, nb, nspk, ntxt)
        for rli in sorted(repl):
            ra, rb, rspk, rtxt = repl[rli]
            tlines = self.text.get("1.0", "end").splitlines()
            # satir numarasi degismedi (satir ekleme/cikarma yok)
            self.text.delete(f"{rli + 1}.0", f"{rli + 1}.end")
            self.text.insert(f"{rli + 1}.0",
                             f"[{sec_to_ts(ra)} - {sec_to_ts(rb)}] {rspk}: {rtxt}")
        self.resync()
        self.status.configure(text=f"{which} {ms:+d}ms")
        self.sel_idx = i
        if 0 <= i < len(self.seg_btns):
            self.on_seg_select(i)

    # ---------- senkron ----------
    def _on_modified(self, _e=None):
        try:
            if not self.text.edit_modified():
                return
            self.text.edit_modified(False)
        except Exception:
            return
        if self._sync_job is not None:
            try:
                self.after_cancel(self._sync_job)
            except Exception:
                pass
        self._sync_job = self.after(800, self._auto_sync)

    def _auto_sync(self):
        self._sync_job = None
        try:
            if not self.file:
                return
            keep = self.sel_idx
            self.resync(quiet=True)
            if keep is not None and 0 <= keep < len(self.seg_btns):
                self.sel_idx = keep
                self.seg_btns[keep].configure(fg_color="#1d4ed8")
        except Exception:
            pass

    def resync(self, quiet: bool = False):
        for tag in list(self.text.tag_names()):
            if tag.startswith("ts:") or tag == "ts":
                self.text.tag_delete(tag)
        self.text.tag_configure("ts", foreground="#7d8590", underline=True)
        idx = "1.0"
        while True:
            idx = self.text.search(r"\[\d+:\d+:\d+", idx, regexp=True, stopindex="end")
            if not idx:
                break
            line_end = self.text.index(f"{idx} lineend")
            line = self.text.get(idx, line_end)
            m = LINE_RE.match(line)
            if m:
                t1, t2 = m.group(1), m.group(2)
                tag = f"ts:{t1}:{t2}"
                span = f"{idx}+{len(t1) + len(t2) + 6}c"
                self.text.tag_add("ts", idx, span)
                self.text.tag_add(tag, idx, span)
                self.text.tag_bind(tag, "<Button-1>",
                                    lambda e, a=t1, b=t2: self.play(a, b))
            idx = line_end
        try:
            self.text.tag_lower("ts")
        except Exception:
            pass
        self.recolor()
        self.refresh_seglist()
        if not quiet:
            self.status.configure(text="esitlendi")

    def recolor(self):
        # konusmaciya gore boyamayi tazele (satir ortasindaki etiketi bul)
        for tag in list(self.text.tag_names()):
            if tag.startswith("spk"):
                self.text.tag_remove(tag, "1.0", "end")
        seen: dict[str, int] = {}
        ci = 0
        for lineno, line in enumerate(self.text.get("1.0", "end").splitlines(), start=1):
            m = LINE_RE.match(line)
            if not m:
                continue
            spk = m.group(3)
            if spk not in seen:
                seen[spk] = ci
                ci += 1
            start = line.find(spk)
            if start < 0:
                continue
            self.text.tag_add(f"spk{seen[spk] % 8}",
                              f"{lineno}.{start}", f"{lineno}.{start + len(spk) + 1}")

    def refresh_seglist(self):
        for w in self.seglist.winfo_children():
            w.destroy()
        self.seg_btns = []
        self.seg_starts = []
        for line in self.text.get("1.0", "end").splitlines():
            m = LINE_RE.match(line)
            if not m:
                continue
            t1, t2, spk, txt = m.groups()
            self.seg_starts.append((t1, t2))
            i = len(self.seg_btns)
            b = ctk.CTkButton(self.seglist, text=f"[{t1}]\n{spk}: {txt[:48]}",
                              anchor="w", height=52, fg_color="#2b2b2b",
                              command=lambda k=i: self.on_seg_select(k))
            b.pack(fill="x", padx=2, pady=2)
            self.seg_btns.append(b)
        for i in range(0, len(self.seg_btns), 2):
            try:
                self.seg_btns[i].configure(fg_color="#26262e")
            except Exception:
                pass
        if self.sel_idx is not None and 0 <= self.sel_idx < len(self.seg_btns):
            self.seg_btns[self.sel_idx].configure(fg_color="#1d4ed8")

    def save(self):
        if not self.file:
            return
        self.resync()
        bak = self.file.with_suffix(self.file.suffix + ".bak")
        bak.write_text(self.file.read_text(encoding="utf-8"), encoding="utf-8")
        self.file.write_text(self.text.get("1.0", "end").rstrip() + "\n", encoding="utf-8")
        self.status.configure(text=f"kaydedildi (+yedek: {bak.name})")

    # ---------- transkript uret ----------
    def open_transcribe(self):
        TranscribeDialog(self, self.wav)


class TranscribeDialog(ctk.CTkToplevel):
    """Ses dosyasindan transkript uret (arka planda calisir)."""

    def __init__(self, master: Editor, wav: Path | None):
        super().__init__(master)
        self.master_editor = master
        self.title("Transkript Uret")
        self.geometry("480x470")
        self.resizable(False, False)
        self.proc = None
        self.after(10, self.lift)

        frm = ctk.CTkFrame(self, fg_color="transparent")
        frm.pack(fill="both", expand=True, padx=14, pady=14)

        ctk.CTkLabel(frm, text="Ses dosyasi:").pack(anchor="w")
        row = ctk.CTkFrame(frm, fg_color="transparent")
        row.pack(fill="x", pady=2)
        self.wav_var = tk.StringVar(value=str(wav) if wav else "")
        ctk.CTkEntry(row, textvariable=self.wav_var, width=380).pack(side="left", fill="x", expand=True)
        ctk.CTkButton(row, text="...", width=36, command=self.pick).pack(side="left", padx=(6, 0))

        opts = ctk.CTkFrame(frm, fg_color="transparent")
        opts.pack(fill="x", pady=6)
        ctk.CTkLabel(opts, text="Denoise:").grid(row=0, column=0, sticky="w", padx=2, pady=2)
        self.denoise = tk.StringVar(value="raw")
        ctk.CTkOptionMenu(opts, variable=self.denoise,
                         values=["raw", "noisereduce", "deepfilter", "all"]).grid(row=0, column=1, sticky="w", padx=2)
        ctk.CTkLabel(opts, text="Model:").grid(row=1, column=0, sticky="w", padx=2, pady=2)
        self.model = tk.StringVar(value="turbo-tr")
        ctk.CTkOptionMenu(opts, variable=self.model,
                         values=["turbo-tr", "e2", "yeni", "turbo", "large-v3"]).grid(row=1, column=1, sticky="w", padx=2)
        ctk.CTkLabel(opts, text="Cihaz:").grid(row=2, column=0, sticky="w", padx=2, pady=2)
        self.device = tk.StringVar(value="cuda")
        ctk.CTkOptionMenu(opts, variable=self.device, values=["cuda", "cpu"]).grid(row=2, column=1, sticky="w", padx=2)
        ctk.CTkLabel(opts, text="Dil:").grid(row=3, column=0, sticky="w", padx=2, pady=2)
        self.lang = tk.StringVar(value="tr")
        ctk.CTkEntry(opts, textvariable=self.lang, width=80).grid(row=3, column=1, sticky="w", padx=2)
        self.diarize = tk.BooleanVar(value=True)
        ctk.CTkCheckBox(opts, text="Konusmaci ayrimi", variable=self.diarize).grid(row=4, column=0, columnspan=2, sticky="w", pady=2)
        ctk.CTkLabel(opts, text="Kisi sayisi:").grid(row=5, column=0, sticky="w", padx=2, pady=2)
        self.speakers = tk.StringVar(value="2")
        ctk.CTkEntry(opts, textvariable=self.speakers, width=80).grid(row=5, column=1, sticky="w", padx=2)
        self.loudnorm = tk.BooleanVar(value=True)
        ctk.CTkCheckBox(opts, text="Ses normalizasyonu (-16 LUFS)", variable=self.loudnorm).grid(row=6, column=0, columnspan=2, sticky="w", pady=2)

        self.run_btn = ctk.CTkButton(frm, text="Baslat", command=self.start)
        self.run_btn.pack(pady=6)
        self.prog = ctk.CTkProgressBar(frm, mode="indeterminate")
        self.prog.pack(fill="x", pady=2)
        self.prog.set(0)
        self.log = tk.Text(frm, height=8, wrap="word", font=("Consolas", 9),
                           bg=BG_TEXT, fg=FG_TEXT, relief="flat")
        self.log.pack(fill="both", expand=True)

    def pick(self):
        f = tk.filedialog.askopenfilename(
            filetypes=[("Ses", "*.wav *.mp3 *.m4a *.ogg *.flac")])
        if f:
            self.wav_var.set(f)

    def start(self):
        wav = self.wav_var.get().strip()
        if not wav or not Path(wav).exists():
            tk.messagebox.showinfo("Bilgi", "Ses dosyasi sec")
            return
        script = Path(__file__).parent / "transcribe_compare.py"
        outdir = Path(wav).parent / (Path(wav).stem + "-transkript")
        compute = "float16" if self.device.get() == "cuda" else "int8"
        cmd = [sys.executable, str(script), "-i", wav,
               "--denoise", self.denoise.get(),
               "--model", self.model.get(),
               "--loudnorm" if self.loudnorm.get() else "--no-loudnorm",
               "--lang", self.lang.get().strip() or "tr",
               "--device", self.device.get(), "--compute", compute,
               "-o", str(outdir)]
        if self.diarize.get():
            cmd += ["--diarize"]
            try:
                cmd += ["--speakers", str(int(self.speakers.get()))]
            except ValueError:
                pass
        self.log.delete("1.0", "end")
        self.log.insert("end", "Calisiyor...\n")
        self.run_btn.configure(state="disabled")
        self.prog.start()
        import queue
        self.q: queue.Queue = queue.Queue()
        env = dict(__import__("os").environ)
        env["HF_HUB_OFFLINE"] = "1"
        env["PYTHONWARNINGS"] = "ignore"
        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT, text=True,
                                     encoding="utf-8", errors="replace", env=env,
                                     creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))

        def worker():
            assert self.proc and self.proc.stdout
            for line in self.proc.stdout:
                self.q.put(line)
            self.q.put(None)

        threading.Thread(target=worker, daemon=True).start()
        self.outdir = outdir
        self.after(200, self.poll)

    def poll(self):
        assert self.proc is not None
        while not self.q.empty():
            line = self.q.get()
            if line is None:
                return self.finish()
            self.log.insert("end", line)
            self.log.see("end")
        if self.proc.poll() is None:
            self.after(300, self.poll)
        else:
            self.finish()

    def finish(self):
        self.prog.stop()
        self.run_btn.configure(state="normal")
        rc = self.proc.poll() if self.proc else -1
        if rc == 0:
            self.log.insert("end", "\nBITTI.\n")
            spks = sorted(self.outdir.glob("*.spk.txt"))
            trs = sorted(self.outdir.glob("*.tr.txt"))
            target = spks[0] if spks else (trs[0] if trs else None)
            if target:
                self.master_editor.open_file(target)
                self.destroy()
            else:
                tk.messagebox.showinfo("Bilgi", "Transkript bulunamadi, loga bak")
        else:
            self.log.insert("end", f"\nHATA (kod {rc}). Log yukarida.\n")


if __name__ == "__main__":
    start = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    Editor(start).mainloop()
