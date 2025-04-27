from clearml import Task, Dataset
Task.add_requirements("requirements.txt")

import os
import json
import logging
import zipfile
from pathlib import Path
import torch
from torch.utils.data import Dataset as TorchDataset, DataLoader
from PIL import Image
from transformers import (
    VisionEncoderDecoderModel,
    ViTFeatureExtractor,
    AutoTokenizer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)
import evaluate
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.spice.spice import Spice
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.nn.utils.rnn import pad_sequence
from transformers.models.bart.modeling_bart import shift_tokens_right

task = Task.init(
    project_name="Description",
    task_name="step4_desc_model_training",
    task_type=Task.TaskTypes.training
)
logger = task.get_logger()
# 1. Initialize ClearML Task
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

# 2. Fetch split JSON dataset from "Desc_final_dataset" under "Description" project
split_ds = Dataset.get(dataset_id="41511324658b4cc0a49d3e1c771415f4", only_completed=True, alias="split_data")
splits_path = Path(split_ds.get_local_copy())
TRAIN_CAPTIONS_JSON = splits_path / "train.json"
VAL_CAPTIONS_JSON = splits_path / "val.json"
TEST_CAPTIONS_JSON = splits_path / "test.json"
logging.info(f"Split JSONs located at: {splits_path}")

# 3. Fetch images ZIP from "base_dataset_zip" under "Detection" project
img_ds = Dataset.get(dataset_project="Detection", dataset_name="base_dataset_zip", only_completed=True, alias="image_data")

images_data = Dataset.get(
    dataset_id= 'd8316762cb3844569f4c1fbe643ed7f4', #"2231b5b121924ed684d6560cf6839619",
    only_completed=True,
    alias="base_images"  
)
raw_path = Path(images_data.get_local_copy())
if raw_path.is_dir():
    inner_zips = list(raw_path.glob("*.zip"))
    if inner_zips:
        zip_path = inner_zips[0]
        logging.info(f"Found inner zip: {zip_path.name}, will extract that")
        raw_path = zip_path
# unzip all contents
if raw_path.is_file() and raw_path.suffix.lower() == ".zip":
    extract_root = raw_path.parent / raw_path.stem
    extract_root.mkdir(exist_ok=True)
    logging.info(f"Unpacking {raw_path.name} → {extract_root}")
    with zipfile.ZipFile(raw_path, "r") as zp:
        zp.extractall(path=extract_root)
    extract_path = extract_root
else:
    extract_path = raw_path

# ─── AUTO-DETECT images/ AND labels/ ────────────────────────────────────────────
def find_dir_with_most_files(root: Path, name: str) -> Path:
    """Search recursively for folders named `name` and return the one containing the most files."""
    best_dir = None
    best_count = 0
    for candidate in root.rglob(name):
        if candidate.is_dir():
            cnt = sum(1 for _ in candidate.iterdir() if _.is_file())
            if cnt > best_count:
                best_dir, best_count = candidate, cnt
    if not best_dir:
        raise FileNotFoundError(f"No directory named '{name}' found under {root}")
    return best_dir

IMAGE_ROOT = find_dir_with_most_files(extract_path, "images")
logging.info(f"Images located at: {IMAGE_ROOT}")

# 4. Student & training config
STUDENT_CONFIG = {"encoder": "google/vit-base-patch16-224-in21k", "decoder": "distilgpt2"}
TRAIN_BATCH_SIZE = 16
EVAL_BATCH_SIZE = 16
NUM_EPOCHS = 10
LR = 5e-5
MAX_TARGET_LEN = 64
BEAM_SIZE = 4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 5. Dataset class
class CaptionDataset(Dataset):
    def __init__(self, captions_json, image_root, feature_extractor, tokenizer, max_len):
        """
        captions_json: list of {"image": "path.jpg", "caption": "…"}
        """
        import json
        raw = json.load(open(captions_json))
        if isinstance(raw, dict) and all(isinstance(v, str) for v in raw.values()):
            # transform to list of {"image": ..., "caption": ...}
            self.data = [{"image": img_name, "caption": cap_text} for img_name, cap_text in raw.items()]
        else:
            self.data = raw
        self.image_root = image_root
        self.fe = feature_extractor
        self.tk = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        img = Image.open(os.path.join(self.image_root, item["image"])).convert("RGB")
        pixel_values = self.fe(images=img, return_tensors="pt").pixel_values.squeeze()
        tokenized = self.tk(
            item["caption"], padding="max_length", truncation=True, max_length=self.max_len, return_tensors="pt")
        labels = tokenized.input_ids.squeeze()
        labels[labels == self.tk.pad_token_id] = -100  # ignore pad in loss
        return {"pixel_values": pixel_values, "labels": labels}


# 6. Load model and preprocessors
feature_extractor = ViTFeatureExtractor.from_pretrained(STUDENT_CONFIG["encoder"])
tokenizer         = AutoTokenizer.from_pretrained(STUDENT_CONFIG["decoder"])
model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
    STUDENT_CONFIG["encoder"],
    STUDENT_CONFIG["decoder"],
)
tokenizer.add_special_tokens({"pad_token": "[PAD]"})
# 2) Resize the decoder’s embeddings to match new vocab size
model.decoder.resize_token_embeddings(len(tokenizer))
model.config.pad_token_id = tokenizer.pad_token_id
# Tie special tokens
model.config.decoder_start_token_id = tokenizer.bos_token_id
model.config.eos_token_id           = tokenizer.eos_token_id
model.config.vocab_size             = model.config.decoder.vocab_size
model.config.max_length             = MAX_TARGET_LEN
model.config.early_stopping         = True
model.config.no_repeat_ngram_size   = 3
model.config.num_beams              = BEAM_SIZE
model.config.length_penalty         = 2.0
model.to(device)

# 7. Prepare datasets & dataloaders
def collate_fn(batch):
    # 1) Stack the images
    pixel_values = torch.stack([ex["pixel_values"] for ex in batch])  # (B, C, H, W)

    # 2) Pad & stack the labels
    label_seqs = [ex["labels"] for ex in batch]                       # list of [L_i]
    labels = pad_sequence(label_seqs, batch_first=True,
                          padding_value=tokenizer.pad_token_id)       # (B, L_max)
    # Make sure pad tokens are ignored by the loss
    labels_masked = labels.clone()
    labels_masked[labels_masked == tokenizer.pad_token_id] = -100

    # 3) Build explicit decoder_input_ids
    decoder_input_ids = shift_tokens_right(
        labels,
        pad_token_id=tokenizer.pad_token_id,
        decoder_start_token_id=model.config.decoder_start_token_id
    )

    return {
        "pixel_values":       pixel_values,
        "labels":             labels_masked,
        "decoder_input_ids":  decoder_input_ids,
    }

train_ds = CaptionDataset(TRAIN_CAPTIONS_JSON, IMAGE_ROOT, feature_extractor, tokenizer, MAX_TARGET_LEN)
val_ds   = CaptionDataset(VAL_CAPTIONS_JSON, IMAGE_ROOT, feature_extractor, tokenizer, MAX_TARGET_LEN)

# 8. Metrics
bleu_metric  = evaluate.load("bleu")
rouge_metric = evaluate.load("rouge")
cider_scorer = Cider()
spice_scorer = Spice()
import numpy as np

def compute_metrics(eval_pred):
    preds, labels = eval_pred

    # --- decode predictions & labels ---
    decoded_preds  = tokenizer.batch_decode(preds, skip_special_tokens=True)
    labels_clean   = np.where(labels == -100, tokenizer.pad_token_id, labels)
    decoded_labels = tokenizer.batch_decode(labels_clean, skip_special_tokens=True)

    # --- BLEU ---
    bleu = bleu_metric.compute(
        predictions=decoded_preds,
        references=[[ref] for ref in decoded_labels]
    )["bleu"]

    # --- ROUGE (grab floats directly) ---
    rouge_res = rouge_metric.compute(
        predictions=decoded_preds,
        references=decoded_labels
    )
    rouge1 = rouge_res["rouge1"]
    rougeL = rouge_res["rougeL"]

    # --- CIDEr & SPICE (unchanged) ---
    refs_dict = {i: [decoded_labels[i]] for i in range(len(decoded_labels))}
    hyps_dict = {i: [decoded_preds[i]]  for i in range(len(decoded_preds))}
    cider_score, _ = cider_scorer.compute_score(refs_dict, hyps_dict)
    spice_score, _ = spice_scorer.compute_score(refs_dict, hyps_dict)

    return {
        "bleu":   bleu,
        "rouge1": rouge1,
        "rougeL": rougeL,
        "cider":  cider_score,
        "spice":  spice_score,
    }

# 9. TrainingArguments & Trainer
# --- 6. TrainingArguments & Trainer ---
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
# where to save checkpoints/logs locally
OUTPUT_DIR = extract_path / "outputs" / "models"
tensorboard_dir = extract_path/ "outputs" / "tensorboard_logs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
tensorboard_dir.mkdir(parents=True, exist_ok=True)

class CleanSeq2SeqTrainer(Seq2SeqTrainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # pop Trainer-only args so they don't get forwarded into your model
        inputs.pop("num_items_in_batch", None)
        # now call the parent (it expects just model, inputs, return_outputs)
        return super().compute_loss(model, inputs, return_outputs)

training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=TRAIN_BATCH_SIZE,
    per_device_eval_batch_size=EVAL_BATCH_SIZE,
    predict_with_generate=True,
    eval_strategy="epoch",     # run eval (and log eval_loss) at end of each epoch
    logging_strategy="epoch",        # log train loss every epoch             
    save_strategy="epoch",
    save_total_limit=2,
    weight_decay=0.01,            # L2 regularization on all weights
    label_smoothing_factor=0.1,   # soften the training targets
    num_train_epochs=NUM_EPOCHS,
    learning_rate=LR,
    fp16=torch.cuda.is_available(),
    load_best_model_at_end=True,  # pick the checkpoint with highest cider
    metric_for_best_model="cider",
    greater_is_better=True,
    report_to=["tensorboard"],       # enable TensorBoard
    logging_dir=tensorboard_dir
)

trainer = CleanSeq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    data_collator=collate_fn,
    compute_metrics=compute_metrics
)

# 10. Train
trainer.train()

# 11. Plot loss curve and log to ClearML
history = trainer.state.log_history
epochs = [h['epoch'] for h in history if 'loss' in h]
losses = [h['loss'] for h in history if 'loss' in h]
f, ax = plt.subplots()
ax.plot(epochs, losses, marker='o')
ax.set_xlabel('Epoch')
ax.set_ylabel('Training Loss')
ax.set_title('Loss Curve')
logger.report_matplotlib("training_plot", "loss_curve", f)

# 12. Save best model and artifacts
best_ckpt = trainer.state.best_model_checkpoint
task.get_logger().report_text(f"Best checkpoint at: {best_ckpt}")
best_model = VisionEncoderDecoderModel.from_pretrained(best_ckpt)
best_dir = extract_path / "outputs"/ "best_model"
best_dir.mkdir(parents=True, exist_ok=True)
best_model.save_pretrained(best_dir)
tokenizer.save_pretrained(best_dir)
feature_extractor.save_pretrained(best_dir)
task.upload_artifact(name="best_model", artifact_object=best_dir)

# 13. Final evaluation on test set
test_ds = CaptionDataset(TEST_CAPTIONS_JSON, IMAGE_ROOT, feature_extractor, tokenizer, MAX_TARGET_LEN)
res = trainer.evaluate(eval_dataset=test_ds)
for k, v in res.items():
    logger.report_scalar("test_metrics", k, iteration=0, value=v)
    logging.info(f"Test {k}: {v}")

logging.info("Student training on ClearML complete.")
