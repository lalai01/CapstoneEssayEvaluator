"""
Fine-tune microsoft/trocr-base-handwritten for the app's handwriting reader.

Designed for Google Colab with a T4 GPU (also runs on CPU, very slowly).

Training data (relative to the working directory; each source is optional,
but at least one must exist):
  training-data/labels.csv + crops/      your own lines from export_training_data.py
                                         (OCR Labeler samples and --pages folders);
                                         split 90% train / 10% validation
  handwriting_dataset/train/image/       IAM handwriting lines; labels in
                                         handwriting_dataset/train/labels/ and
                                         full_dataset/<style>/<level>/
  handwriting_dataset/validation/images/ IAM validation lines; labels in
                                         handwriting_dataset/validation/label/
Lines from full_dataset are grouped by writing style and level. 20% of each
group goes to validation so the report shows CER per style.

To make the character error rate (CER) as low as possible:
  - training images are randomly varied (slant, rotation, stroke thickness,
    blur, contrast) so the model doesn't memorise the few writers it sees;
  - the checkpoint with the lowest validation CER is kept, not the lowest loss;
  - training stops early once CER stops improving;
  - IAM's spaced punctuation ("word ." / "Roy 's") is normalised to normal
    English, so the model learns to write punctuation the way essays do.

Output in ./trocr-finetuned/:
  model + processor           ready to copy to /opt/trocr
  training_report.png         loss, CER and accuracy per epoch
                              (epoch 0 = the model before fine-tuning)
                              and CER per data group before vs after
  sample_predictions.png      validation lines with the truth and the prediction
  training_history.csv        the numbers behind the chart

Colab setup (match the Docker image's transformers so the model loads there):
  !pip install transformers==4.44.0 accelerate editdistance pandas scikit-learn matplotlib
  # upload and unzip training-data/, handwriting_dataset/ and full_dataset/
  # next to this script, then:
  !python train_trocr.py
"""
import glob
import os
import random
import re
import shutil

import editdistance
import matplotlib
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import (
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrOCRProcessor,
    VisionEncoderDecoderModel,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

MODEL_NAME = "microsoft/trocr-base-handwritten"
OUTPUT_DIR = "./trocr-finetuned"

USER_LABELS = "training-data/labels.csv"
USER_CROPS = "training-data/crops"
IAM_DIR = "handwriting_dataset"
STYLES_DIR = "full_dataset"

# A handwritten line is rarely over ~30 tokens; the app reads at most 64.
MAX_LENGTH = 64
NUM_BEAMS = 4
EPOCHS = 20
EARLY_STOP_PATIENCE = 4   # epochs without a lower CER before stopping
SEED = 42


# ---------------------------------------------------------------- data

def clean_text(text):
    """One line of normal English: IAM's 'word .' and 'Roy 's' become 'word.' and "Roy's"."""
    t = re.sub(r"\s+", " ", str(text)).strip()
    t = re.sub(r"\s+([.,;:!?)\]])", r"\1", t)
    t = re.sub(r"([(\[])\s+", r"\1", t)
    t = re.sub(r"\s+'(s|t|re|ve|ll|d|m)\b", r"'\1", t, flags=re.IGNORECASE)
    return t


def read_text(path):
    with open(path, encoding="utf-8", errors="replace") as f:
        return f.read()


def load_rows():
    """Every labeled line as {path, text, group, split}."""
    rows = []

    # Your own lines: random 90/10 split.
    if os.path.exists(USER_LABELS):
        df = pd.read_csv(USER_LABELS, dtype=str, keep_default_na=False)
        df = df[df["file_name"].map(lambda f: os.path.exists(os.path.join(USER_CROPS, f)))]
        if len(df) >= 10:
            train_df, val_df = train_test_split(df, test_size=0.1, random_state=SEED)
        else:
            train_df, val_df = df, df.iloc[0:0]
        for split, part in (("train", train_df), ("validation", val_df)):
            for _, r in part.iterrows():
                rows.append({"path": os.path.join(USER_CROPS, r["file_name"]),
                             "text": r["text"], "group": "Your handwriting", "split": split})

    # IAM lines. Images live in train/image and validation/images; labels in
    # train/labels, validation/label, or full_dataset/<style>/<level>/.
    images = {os.path.splitext(os.path.basename(p))[0]: p
              for p in glob.glob(os.path.join(IAM_DIR, "**", "*.png"), recursive=True)}

    styled = []
    for p in sorted(glob.glob(os.path.join(STYLES_DIR, "*", "*", "*.txt"))):
        name = os.path.splitext(os.path.basename(p))[0]
        style = os.path.basename(os.path.dirname(os.path.dirname(p)))
        level = os.path.basename(os.path.dirname(p))
        if name in images:
            styled.append((name, read_text(p), f"{style} / {level}"))
    styled_names = {n for n, _, _ in styled}

    # 20% of each style/level group is held out for the per-style report.
    by_group = {}
    for item in styled:
        by_group.setdefault(item[2], []).append(item)
    rng = random.Random(SEED)
    for group, items in sorted(by_group.items()):
        rng.shuffle(items)
        n_val = max(1, round(len(items) * 0.2)) if len(items) >= 3 else 0
        for i, (name, text, _) in enumerate(items):
            rows.append({"path": images[name], "text": text, "group": group,
                         "split": "validation" if i < n_val else "train"})

    for split, label_dir in (("train", "train/labels"), ("validation", "validation/label")):
        for p in sorted(glob.glob(os.path.join(IAM_DIR, label_dir, "*.txt"))):
            name = os.path.splitext(os.path.basename(p))[0]
            if name in images and name not in styled_names:
                rows.append({"path": images[name], "text": read_text(p),
                             "group": "IAM lines", "split": split})

    df = pd.DataFrame(rows, columns=["path", "text", "group", "split"])
    df["text"] = df["text"].map(clean_text)
    return df[df["text"] != ""].reset_index(drop=True)


def augment(image):
    """A slightly different version of a line each time it's seen."""
    if random.random() < 0.2:
        return image
    white = (255, 255, 255)
    # Thicker or thinner pen.
    r = random.random()
    if r < 0.15:
        image = image.filter(ImageFilter.MinFilter(3))
    elif r < 0.30:
        image = image.filter(ImageFilter.MaxFilter(3))
    # Slant (shear) and a small rotation.
    w, h = image.size
    shear = random.uniform(-0.35, 0.35)
    margin = int(abs(shear) * h / 2) + 4
    image = ImageOps.expand(image, border=(margin, 0), fill=white)
    w = image.size[0]
    image = image.transform((w, h), Image.Transform.AFFINE, (1, shear, -shear * h / 2, 0, 1, 0),
                            resample=Image.Resampling.BICUBIC, fillcolor=white)
    image = image.rotate(random.uniform(-2.5, 2.5), resample=Image.Resampling.BICUBIC,
                         expand=True, fillcolor=white)
    # Narrower or wider writing.
    if random.random() < 0.5:
        w, h = image.size
        image = image.resize((max(8, int(w * random.uniform(0.8, 1.2))), h), Image.Resampling.BICUBIC)
    # Camera blur, contrast, lighting, margins.
    if random.random() < 0.25:
        image = image.filter(ImageFilter.GaussianBlur(random.uniform(0.3, 1.1)))
    image = ImageEnhance.Contrast(image).enhance(random.uniform(0.6, 1.3))
    image = ImageEnhance.Brightness(image).enhance(random.uniform(0.85, 1.1))
    pad = [random.randint(0, 16) for _ in range(4)]
    return ImageOps.expand(image, border=tuple(pad), fill=white)


class LineDataset(Dataset):
    """One line image + its text per item."""

    def __init__(self, df, processor, train=False):
        self.df = df.reset_index(drop=True)
        self.processor = processor
        self.train = train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row["path"]).convert("RGB")
        if self.train:
            image = augment(image)

        pixel_values = self.processor(
            images=image, return_tensors="pt"
        ).pixel_values.squeeze(0)

        # The tokenizer gives "<s> text </s>", but TrOCR starts each line from
        # </s> and goes straight to the text; it never writes <s>. Training on
        # the <s> costs ~500x the loss and teaches the model a token it never
        # uses, so the labels are "text </s>".
        labels = self.processor.tokenizer(
            row["text"],
            padding="max_length",
            max_length=MAX_LENGTH + 1,
            truncation=True,
            return_tensors="pt",
        ).input_ids.squeeze(0)[1:]

        # Padding positions are ignored by the loss.
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        return {"pixel_values": pixel_values, "labels": labels}


# ---------------------------------------------------------------- metrics

def corpus_cer(preds, truths):
    """Total character edits / total characters."""
    edits = sum(editdistance.eval(p.strip(), t.strip()) for p, t in zip(preds, truths))
    chars = sum(len(t.strip()) for t in truths)
    return edits / max(1, chars)


def build_compute_metrics(processor):
    tokenizer = processor.tokenizer

    def compute_metrics(eval_pred):
        pred_ids, label_ids = eval_pred.predictions, eval_pred.label_ids
        if isinstance(pred_ids, tuple):
            pred_ids = pred_ids[0]
        pred_ids = np.where(pred_ids < 0, tokenizer.pad_token_id, pred_ids)
        label_ids = np.where(label_ids == -100, tokenizer.pad_token_id, label_ids)
        preds = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        truths = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        cer = corpus_cer(preds, truths)
        return {"cer": cer, "accuracy": 1 - cer}

    return compute_metrics


def predict(model, processor, df, batch_size=8):
    """Read every line in df with the same decoding the app uses."""
    model.eval()
    preds = []
    for start in range(0, len(df), batch_size):
        batch = df.iloc[start:start + batch_size]
        images = [Image.open(p).convert("RGB") for p in batch["path"]]
        pixel_values = processor(images=images, return_tensors="pt").pixel_values.to(model.device)
        with torch.no_grad():
            generated = model.generate(pixel_values)
        preds += processor.batch_decode(generated, skip_special_tokens=True)
    return [p.strip() for p in preds]


def cer_by_group(df, preds):
    out = {"All validation lines": corpus_cer(preds, list(df["text"]))}
    for group in sorted(df["group"].unique()):
        mask = (df["group"] == group).to_numpy()
        out[group] = corpus_cer([p for p, m in zip(preds, mask) if m],
                                list(df["text"][mask]))
    return out


# ---------------------------------------------------------------- report

def epoch_history(log_history, before):
    """One row per epoch: train loss, validation loss, CER, accuracy."""
    rows = [{"epoch": 0, "train_loss": np.nan, "eval_loss": before["eval_loss"],
             "cer": before["eval_cer"], "accuracy": 1 - before["eval_cer"]}]
    train_losses = [(e["epoch"], e["loss"]) for e in log_history if "loss" in e and "epoch" in e]
    for e in log_history:
        if "eval_cer" not in e or "epoch" not in e:
            continue
        epoch = round(e["epoch"])
        losses = [l for ep, l in train_losses if epoch - 1 < ep <= epoch + 1e-6]
        rows.append({"epoch": epoch,
                     "train_loss": float(np.mean(losses)) if losses else np.nan,
                     "eval_loss": e["eval_loss"], "cer": e["eval_cer"],
                     "accuracy": 1 - e["eval_cer"]})
    return pd.DataFrame(rows)


def plot_report(history, before_groups, after_groups, counts, path):
    fig = plt.figure(figsize=(15, 10))
    grid = fig.add_gridspec(2, 2, height_ratios=[1, 1.1])
    best = history.iloc[history["cer"].idxmin()]

    ax = fig.add_subplot(grid[0, 0])
    ax.plot(history["epoch"], history["train_loss"], "o-", label="Training loss")
    ax.plot(history["epoch"], history["eval_loss"], "s-", label="Validation loss")
    ax.set_title("Loss per epoch")
    ax.set_xlabel("Epoch")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_ylabel("Loss")
    ax.grid(alpha=0.3)
    ax.legend()

    ax = fig.add_subplot(grid[0, 1])
    ax.plot(history["epoch"], history["cer"] * 100, "o-", color="#dc2626", label="CER")
    ax.plot(history["epoch"], history["accuracy"] * 100, "o-", color="#16a34a",
            label="Accuracy (1 − CER)")
    for _, r in history.iterrows():
        ax.annotate(f"{r['cer']:.1%}", (r["epoch"], r["cer"] * 100),
                    textcoords="offset points", xytext=(0, 7), ha="center", fontsize=8)
    ax.axvline(best["epoch"], color="gray", ls="--", lw=1)
    ax.annotate(f"best: epoch {int(best['epoch'])}\nCER {best['cer']:.2%}",
                (best["epoch"], 50), textcoords="offset points", xytext=(6, 0), fontsize=9)
    ax.set_ylim(0, 100)
    ax.set_title("Validation CER and accuracy per epoch (epoch 0 = before training)")
    ax.set_xlabel("Epoch")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_ylabel("%")
    ax.grid(alpha=0.3)
    ax.legend(loc="center right")

    ax = fig.add_subplot(grid[1, :])
    groups = list(after_groups)
    x = np.arange(len(groups))
    b = [before_groups[g] * 100 for g in groups]
    a = [after_groups[g] * 100 for g in groups]
    ax.bar(x - 0.2, b, 0.4, label="Before fine-tuning", color="#9ca3af")
    ax.bar(x + 0.2, a, 0.4, label="After fine-tuning", color="#4f46e5")
    for i, (vb, va) in enumerate(zip(b, a)):
        ax.text(i - 0.2, vb + 0.5, f"{vb:.1f}", ha="center", fontsize=7)
        ax.text(i + 0.2, va + 0.5, f"{va:.1f}", ha="center", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{g}\n({counts.get(g, 0)} lines)" for g in groups],
                       rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("CER %")
    ax.set_title("Validation CER by kind of handwriting")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()

    fig.suptitle(f"TrOCR fine-tuning: CER {history['cer'].iloc[0]:.2%} → {best['cer']:.2%} "
                 f"(accuracy {1 - best['cer']:.2%})", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_samples(df, before_preds, after_preds, path, n=12):
    """Validation lines with truth, prediction before and after fine-tuning."""
    rng = random.Random(SEED)
    idx = rng.sample(range(len(df)), min(n, len(df)))
    fig, axes = plt.subplots(len(idx), 1, figsize=(12, 1.9 * len(idx)))
    axes = np.atleast_1d(axes)
    for ax, i in zip(axes, idx):
        row = df.iloc[i]
        ax.imshow(Image.open(row["path"]).convert("RGB"))
        ax.set_xticks([])
        ax.set_yticks([])
        cer_b = corpus_cer([before_preds[i]], [row["text"]])
        cer_a = corpus_cer([after_preds[i]], [row["text"]])
        color = "#16a34a" if cer_a == 0 else ("#ca8a04" if cer_a < 0.1 else "#dc2626")
        ax.set_title(f"[{row['group']}]  truth: {row['text']}", fontsize=9, loc="left")
        ax.set_xlabel(f"before ({cer_b:.0%} CER): {before_preds[i]}\n"
                      f"after  ({cer_a:.0%} CER): {after_preds[i]}",
                      fontsize=9, loc="left", color=color)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# ---------------------------------------------------------------- main

def main():
    random.seed(SEED)
    torch.manual_seed(SEED)

    df = load_rows()
    if df.empty:
        raise SystemExit("No training data found. See the top of this file for the folders it reads.")
    train_df = df[df["split"] == "train"].reset_index(drop=True)
    val_df = df[df["split"] == "validation"].reset_index(drop=True)
    print(f"Train: {len(train_df)} lines, validation: {len(val_df)} lines")
    print(df.groupby(["group", "split"]).size().unstack(fill_value=0).to_string())

    processor = TrOCRProcessor.from_pretrained(MODEL_NAME)
    model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME)

    # TrOCR starts every line from </s> (id 2). The checkpoint keeps that in
    # the decoder's config; training and generate() read the top-level one.
    start = model.config.decoder.decoder_start_token_id
    model.config.decoder_start_token_id = (
        start if start is not None else processor.tokenizer.sep_token_id
    )
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.vocab_size = model.config.decoder.vocab_size

    # generate() reads generation_config, not config. Keep them in sync so
    # the saved model decodes the way it was evaluated here.
    gen = model.generation_config
    gen.decoder_start_token_id = model.config.decoder_start_token_id
    gen.pad_token_id = model.config.pad_token_id
    gen.eos_token_id = processor.tokenizer.sep_token_id
    gen.max_length = MAX_LENGTH
    gen.num_beams = NUM_BEAMS
    gen.early_stopping = True
    gen.no_repeat_ngram_size = 3

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(f"Reading the validation lines with the original model ({device})...")
    before_preds = predict(model, processor, val_df)
    before_groups = cer_by_group(val_df, before_preds)

    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=2,          # 16 lines per update
        num_train_epochs=EPOCHS,
        learning_rate=5e-5,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        weight_decay=0.01,
        predict_with_generate=True,
        generation_max_length=MAX_LENGTH,
        generation_num_beams=NUM_BEAMS,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=20,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="cer",
        greater_is_better=False,
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=2,
        seed=SEED,
        report_to="none",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=LineDataset(train_df, processor, train=True),
        eval_dataset=LineDataset(val_df, processor),
        compute_metrics=build_compute_metrics(processor),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=EARLY_STOP_PATIENCE)],
    )

    before = trainer.evaluate()
    print(f"Before fine-tuning: CER {before['eval_cer']:.2%}")
    trainer.train()

    # trainer.model is the checkpoint with the lowest CER (load_best_model_at_end).
    model = trainer.model
    after_preds = predict(model, processor, val_df)
    after_groups = cer_by_group(val_df, after_preds)
    cer = after_groups["All validation lines"]

    model.save_pretrained(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)
    # The checkpoints hold optimizer state too (several GB); the final model is all the app needs.
    for ckpt in glob.glob(os.path.join(OUTPUT_DIR, "checkpoint-*")):
        shutil.rmtree(ckpt, ignore_errors=True)

    history = epoch_history(trainer.state.log_history, before)
    history.to_csv(os.path.join(OUTPUT_DIR, "training_history.csv"), index=False)
    counts = val_df["group"].value_counts().to_dict()
    counts["All validation lines"] = len(val_df)
    plot_report(history, before_groups, after_groups, counts,
                os.path.join(OUTPUT_DIR, "training_report.png"))
    plot_samples(val_df, before_preds, after_preds,
                 os.path.join(OUTPUT_DIR, "sample_predictions.png"))

    print("\nEpoch  train loss  val loss    CER   accuracy")
    for _, r in history.iterrows():
        print(f"{int(r['epoch']):5d}  {r['train_loss']:10.4f}  {r['eval_loss']:8.4f}  "
              f"{r['cer']:6.2%}  {r['accuracy']:8.2%}")
    print("\nCER by kind of handwriting (before → after):")
    for g in after_groups:
        print(f"  {g:32s} {before_groups[g]:7.2%} → {after_groups[g]:7.2%}")

    print(f"\nValidation CER: {cer:.2%}")
    print(f"Validation accuracy: {1 - cer:.2%}")
    print(f"Charts: {OUTPUT_DIR}/training_report.png, {OUTPUT_DIR}/sample_predictions.png")

    print("Fine-tuned model saved to ./trocr-finetuned/")
    print("To deploy: zip this directory and scp it to the Contabo VPS at\n"
          " ~/CapstoneEssayEvaluator/backend/trocr-finetuned/, then replace\n"
          " the TrOCR download RUN block in backend/Dockerfile with:\n"
          "   COPY trocr-finetuned /opt/trocr\n"
          " and rebuild: docker compose build backend && docker compose up -d backend")


if __name__ == "__main__":
    main()
