"""Whisper TAM ince ayar (telefon Turkcesi) - PEFT yok, dogrudan full fine-tune.
Taban: models/buzzasr-turkish | Veri: dataset/train.jsonl + val.jsonl
Cikti: models/whisper-tr-phone/
Kullanim (venv): python code\\train_full.py
3090 Ti 24GB: batch 4 x grad-accum 8, fp16.
"""
import json
import torch
import soundfile as sf
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from transformers import (Seq2SeqTrainer, Seq2SeqTrainingArguments,
                          WhisperForConditionalGeneration, WhisperProcessor,
                           EarlyStoppingCallback)
import jiwer

BASE = Path(__file__).resolve().parents[1]
MODEL_DIR = BASE / "models" / "whisper-turbo-hf"  # turbo taban
OUT_DIR = BASE / "models" / "whisper-tr-turbo"
DATA = BASE / "dataset"


class SpeechDS(torch.utils.data.Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return self.rows[i]


@dataclass
class Collator:
    processor: Any

    def __call__(self, batch):
        audios, texts = [], []
        for r in batch:
            y, sr = sf.read(r["audio"], dtype="float32")
            assert sr == 16000
            audios.append(y)
            texts.append(r["text"])
        inp = self.processor.feature_extractor(audios, sampling_rate=16000, return_tensors="pt")
        lab = self.processor.tokenizer(texts, return_tensors="pt", padding=True).input_ids
        lab[lab == self.processor.tokenizer.pad_token_id] = -100
        inp["labels"] = lab
        return inp


def load_rows(name: str):
    return [json.loads(l) for l in open(DATA / f"{name}.jsonl", encoding="utf-8") if l.strip()]


def main():
    print("model yukleniyor:", MODEL_DIR, flush=True)
    processor = WhisperProcessor.from_pretrained(str(MODEL_DIR), language="tr", task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(str(MODEL_DIR), torch_dtype=torch.float32)
    model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language="tr", task="transcribe")
    model.config.suppress_tokens = []
    model.generation_config.language = "tr"
    model.generation_config.task = "transcribe"
    model.config.use_cache = False
    for p in model.parameters():
        p.requires_grad = True

    train_rows, val_rows = load_rows("train"), load_rows("val")
    print(f"train: {len(train_rows)} val: {len(val_rows)}", flush=True)

    def wer_metric(pred):
        pred_ids = pred.predictions
        if isinstance(pred_ids, tuple):
            pred_ids = pred_ids[0]
        pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        lab = pred.label_ids
        lab[lab == -100] = processor.tokenizer.pad_token_id
        ref_str = processor.tokenizer.batch_decode(lab, skip_special_tokens=True)
        return {"wer": jiwer.wer(ref_str, pred_str)}

    args = Seq2SeqTrainingArguments(
        output_dir=str(OUT_DIR), per_device_train_batch_size=2,
        per_device_eval_batch_size=4, gradient_accumulation_steps=16,
        learning_rate=5e-6, warmup_steps=20, num_train_epochs=2,
        fp16=True, optim="adafactor", eval_strategy="epoch", save_strategy="epoch",
        save_total_limit=2, load_best_model_at_end=True,
        metric_for_best_model="wer", greater_is_better=False,
        predict_with_generate=True, generation_max_length=225,
        logging_steps=10, report_to="none",
        remove_unused_columns=False, label_names=["labels"],
    )
    trainer = Seq2SeqTrainer(args=args, model=model,
                             train_dataset=SpeechDS(train_rows),
                             eval_dataset=SpeechDS(val_rows),
                             data_collator=Collator(processor),
                             compute_metrics=wer_metric,
                             processing_class=processor.tokenizer,
                             callbacks=[EarlyStoppingCallback(early_stopping_patience=1)])
    trainer.train()
    trainer.save_model(str(OUT_DIR))
    processor.save_pretrained(str(OUT_DIR))
    print("kaydedildi:", OUT_DIR, flush=True)


if __name__ == "__main__":
    main()
