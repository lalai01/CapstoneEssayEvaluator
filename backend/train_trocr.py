"""
Fine-tune microsoft/trocr-base-handwritten on line crops exported by
export_training_data.py.

Designed for Google Colab with a T4 GPU (also runs on CPU, slowly).

Input (relative to the working directory):
  training-data/labels.csv   columns: file_name, text
  training-data/crops/       PNG files named in labels.csv

Output:
  ./trocr-finetuned/         model + processor, ready to copy to /opt/trocr

Colab setup:
  !pip install transformers editdistance pandas scikit-learn
  # upload training-data/ next to this script, then:
  !python train_trocr.py
"""
import os

import editdistance
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrOCRProcessor,
    VisionEncoderDecoderModel,
)

MODEL_NAME = "microsoft/trocr-base-handwritten"
DATA_DIR = "training-data"
LABELS_PATH = os.path.join(DATA_DIR, "labels.csv")
CROPS_DIR = os.path.join(DATA_DIR, "crops")
OUTPUT_DIR = "./trocr-finetuned"
MAX_LENGTH = 128


class LineDataset(Dataset):
    """One line crop + its transcription per item."""

    def __init__(self, df, processor):
        self.df = df.reset_index(drop=True)
        self.processor = processor

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(
            os.path.join(CROPS_DIR, row["file_name"])
        ).convert("RGB")

        pixel_values = self.processor(
            images=image, return_tensors="pt"
        ).pixel_values.squeeze(0)

        labels = self.processor.tokenizer(
            row["text"],
            padding="max_length",
            max_length=MAX_LENGTH,
            truncation=True,
            return_tensors="pt",
        ).input_ids.squeeze(0)

        # Padding positions are ignored by the loss.
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        return {"pixel_values": pixel_values, "labels": labels}


def compute_cer(model, processor, df, batch_size=8):
    """Character error rate over the whole split: total edits / total chars."""
    device = model.device
    model.eval()
    total_edits = 0
    total_chars = 0

    for start in range(0, len(df), batch_size):
        batch = df.iloc[start:start + batch_size]
        images = [
            Image.open(os.path.join(CROPS_DIR, f)).convert("RGB")
            for f in batch["file_name"]
        ]
        pixel_values = processor(
            images=images, return_tensors="pt"
        ).pixel_values.to(device)

        with torch.no_grad():
            generated = model.generate(pixel_values, max_new_tokens=MAX_LENGTH)
        preds = processor.batch_decode(generated, skip_special_tokens=True)

        for pred, truth in zip(preds, batch["text"]):
            total_edits += editdistance.eval(pred, truth)
            total_chars += len(truth)

    return total_edits / max(1, total_chars)


def main():
    # keep_default_na=False so lines like "NA" or "null" stay as text.
    df = pd.read_csv(LABELS_PATH, dtype=str, keep_default_na=False)
    train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)
    print(f"Train: {len(train_df)} samples, validation: {len(val_df)} samples")

    processor = TrOCRProcessor.from_pretrained(MODEL_NAME)
    model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME)

    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.vocab_size = model.config.decoder.vocab_size

    # generate() reads generation_config, not config. Keep them in sync so
    # the saved model decodes with the same start token it was trained on.
    model.generation_config.decoder_start_token_id = model.config.decoder_start_token_id
    model.generation_config.pad_token_id = model.config.pad_token_id

    train_dataset = LineDataset(train_df, processor)
    val_dataset = LineDataset(val_df, processor)

    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=10,
        learning_rate=5e-5,
        predict_with_generate=True,
        generation_max_length=MAX_LENGTH,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=20,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=torch.cuda.is_available(),
        report_to="none",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    trainer.train()

    # trainer.model is the best checkpoint (load_best_model_at_end=True).
    model = trainer.model
    cer = compute_cer(model, processor, val_df)
    print(f"Validation CER: {cer:.2%}")
    print(f"Validation accuracy: {1 - cer:.2%}")

    model.save_pretrained(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)

    print("Fine-tuned model saved to ./trocr-finetuned/")
    print("To deploy: zip this directory and scp it to the Contabo VPS at\n"
          " ~/CapstoneEssayEvaluator/backend/trocr-finetuned/, then replace\n"
          " the TrOCR download RUN block in backend/Dockerfile with:\n"
          "   COPY trocr-finetuned /opt/trocr\n"
          " and rebuild: docker compose build backend && docker compose up -d backend")


if __name__ == "__main__":
    main()
