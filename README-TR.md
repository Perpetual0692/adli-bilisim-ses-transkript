# Adli Bilişim — Türkçe Telefon Konuşması Transkript Sistemi

> [English version](README.md) | Türkçe sürümdesiniz

Gerçek telefon kayıtlarını **metne çeviren, konuşmacıları ayıran, düzeltilmesine imkân veren ve kendi verinizle eğitilebilen** uçtan uca açık kaynak hattı. Özellikle Türkçe telefon görüşmeleri (8 kHz, dar bant, gürültülü, üst üste konuşmalı) için test edildi ve optimize edildi.

## Özellikler

- **Transkripsiyon** — faster-whisper `large-v3-turbo` (varsayılan), `large-v3`, Türkçe ince ayarlı modeller
- **Denoise karşılaştırma** — `raw` / `noisereduce` / `deep-filter.exe` dallarını aynı kayıtta koşurup skorlar
- **Konuşmacı ayrımı (diarization)** — pyannote 3.1, `[00:00:24 - 00:00:51] SPEAKER_00:` formatında zaman damgalı çıktı + **çakışma (overlap) listesi**
- **Düzeltme arayüzü** (`editor.py`) — koyu tema, dinle-düzelt-kaydet, milisaniye kaydırma, bitişik sınır dalga etkisi, toplu yeniden adlandırma, transkript üretme penceresi
- **Değerlendirme** (`score_wer.py`, `eval_models.py`) — hakem metne karşı WER/CER, model kapıştırması
- **Eğitim** (`build_dataset.py` + `train_full.py`) — düzeltilmiş kayıtlarınızdan LoRA/tam ince ayar, CT2 dönüşümüyle hatta geri dönüş

## Ölçülen Sonuçlar

8.6 dk gerçek Türkçe telefon görüşmesi, faster-whisper large-v3-turbo, GPU:

| Varyant | avg_logprob | Not |
|---|---|---|
| raw (önerilen) | **-0.24** | En iyi denge |
| raw + loudnorm | -0.23 | Kırpılmalı kayıtlarda artı |
| noisereduce | -0.25 | Temiz kayıtta etkisiz, gürültülüde veri kaybı riski |
| deepfilter | -0.53 | **Önerilmez** — telefonda konuşmayı da kırpıyor |
| condition_on_previous_text + beam 10 | -0.25, %7 kayıp | Önerilmez, konuşma yutuyor |

Düzeltilmiş 92 sn gerçek kayıt, hakemli WER:

| Model | WER | CER |
|---|---|---|
| turbo (raw) | **0.247** | 0.166 |
| large-v3 | 0.354 | 0.214 |
| BuzzASR/turkish (CT2) | 0.392 | 0.195 |
| noisereduce | 0.481 | 0.453 |
| deepfilter | 0.582 | 0.510 |
| **eğitilmiş model (bu veri, 4.4 sa)** | **0.194** | **0.118** |

> Sonuç: aynı veri üst üste konuşma içerdiği için WER şişer; tablo göreli sıralama içindir. Mutlak doğruluk için referans metin şarttır (`score_wer.py`).

## Kurulum

Python 3.12 64-bit + ffmpeg gerekir.

```bash
git clone <repo> && cd <repo>
python -m venv venv
venv\Scripts\activate        # Linux/macOS: source venv/bin/activate
pip install -r requirements.txt
```

GPU (NVIDIA) için PyTorch CUDA sürümü:

```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu126
```

Diarization modeli korumalıdır (ücretsiz, şart kabulü gerekir):

1. [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1) (+ `segmentation-3.0`, `speaker-diarization-community-1`) sayfalarında Accept'e basın
2. Read izinli bir HF token oluşturup ilk çalıştırmada kullanın:
   `set HF_TOKEN=hf_...` (hedef makinede gerekmez; modeller `models/` altına kopyalanır)

## Hızlı Başlangıç

```bash
# Temel transkript (varsayılan: eğitilmiş model + loudnorm + raw)
python code/transcribe_compare.py -i kayit.wav --lang tr -o out

# Konuşmacı ayrımlı
python code/transcribe_compare.py -i kayit.wav --diarize --speakers 2 --lang tr -o out

# Denoise karşılaştırması
python code/transcribe_compare.py -i kayit.wav --denoise all --lang tr -o out

# İnce ayar: hotwords, beam, VAD eşikleri
python code/transcribe_compare.py -i kayit.wav --hotwords Dörtdivan,Karabük \
  --beam 10 --vad-threshold 0.3 --min-silence 200 --speech-pad 400 --no-speech 0.8 -o out

# Model seçimi: turbo | buzz | large-v3 | e2 | CT2-klasör-yolu
python code/transcribe_compare.py -i kayit.wav --model large-v3 --lang tr -o out

# Hakeme karşı skor
python code/score_wer.py -r referans.txt -d out/
```

**Arayüz:**

```bash
python code/editor.py [klasor | dosya.spk.txt]
```

Klasör aç → `.spk.txt` seç → saate tıkla dinle → metni düzelt → `Ctrl+S`. Transkript üretme, konuşmacı yeniden adlandırma, milisaniye kaydırma ve çakışma listesi dahili.

## Kendi Modelinizi Eğitme

1. Düzeltilmiş `.spk.txt` + ses çiftlerini `data/{Egitim,Egitim2}/` gibi klasörlere koyun (`[hh:mm:ss.mmm - ...] 1.Şahıs: metin` formatı — editör üretir).
2. Veri setini kurun:
   ```bash
   python code/build_dataset.py            # hepsi
   python code/build_dataset.py --only E1  # alt kume
   ```
   Çıktı: `dataset/train.jsonl`, `dataset/val.jsonl`, `dataset/wavs/`.
3. Eğitin (RTX 3090 Ti ~30-45 dk):
   ```bash
   python code/train_full.py   # MODEL_DIR/OUT_DIR script icinde ayarlanir
   ```
4. CT2'ye çevirip hatta takın:
   ```bash
   ct2-transformers-converter --model models/whisper-tr-xxx --output_dir models/xxx-ct2 --quantization float16
   # + preprocessor_config.json ve tokenizer.json kopyalanir (ornek: train scriptine bakin)
   ```
5. `code/eval_models.py` ile doğrulama setinde kapıştırın, kazananı varsayılan yapın (`transcribe_compare.py` içinde `MODEL_ALIASES`).

İpuçları: 1. turda durun (bizde 2. tur ezberledi), doğrulama WER'i artarsa eğitimi kesin, E2 gibi yeni alanlar için 1-2 epoch yeter.

## Proje Yapısı

```
code/
  transcribe_compare.py  # ana hat: denoise + VAD + whisper + diarization
  editor.py              # duzeltme arayuzu (CustomTkinter)
  build_dataset.py       # .spk.txt + wav -> jsonl veri seti
  train_full.py          # tam ince ayar (transformers Trainer)
  train_lora.py          # LoRA alternatifi (peft uyumluluguna dikkat)
  eval_models.py         # val setinde model kapismasi
  score_wer.py           # hakem metne karsi WER/CER
requirements.txt
```

## Modeller ve Lisanslar

Bu repo kendi kodunu MIT ile lisanslar. Kullandığı modellerin lisans/şartları:

- Whisper large-v3 / turbo (OpenAI, MIT), BuzzASR/turkish (araştırma amaçlı — kartını okuyun)
- pyannote diarization 3.1 (gated — HF'de şart kabulü gerekir)
- faster-whisper (MIT), ctranslate2, torch, transformers (Apache-2.0)

Modeller repoya **dahil değildir** (`.gitignore` kapsam dışı bırakır); ilk çevrimiçi çalıştırmada indirilir veya USB/offline kurulumda `models/` altına kopyalanır.

## Adli Not

Transkript yardımcı delildir, asıl delil kaydın kendisidir. Kritik bölümleri kulakla doğrulayın; üst üste konuşmalar (`*.overlap.txt`) daima insan kontrolü ister. Çıktılar otomatik yedeklenir (`*.bak-*`), üzerine yazma korumalıdır.
