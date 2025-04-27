from clearml import Task, Dataset
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

# 1. Initialize ClearML Task
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
task = Task.init(
    project_name="Description",
    task_name="step4_desc_model_training",
    task_type=Task.TaskTypes.training
)
logger = task.get_logger()
task.add_requirements_file("requirements.txt")
# 2. Fetch split JSON dataset from "Desc_final_dataset" under "Description" project
split_ds = Dataset.get(dataset_id="41511324658b4cc0a49d3e1c771415f4", only_completed=True, alias="split_data")
splits_path = Path(split_ds.get_local_copy())
TRAIN_CAPTIONS_JSON = splits_path / "train.json"
VAL_CAPTIONS_JSON = splits_path / "val.json"
TEST_CAPTIONS_JSON = splits_path / "test.json"
logging.info(f"Split JSONs located at: {splits_path}")

# 3. Fetch images ZIP from "base_dataset_zip" under "Detection" project
img_ds = Dataset.get(dataset_project="Detection", dataset_name="base_dataset_zip", only_completed=True, alias="image_data")
raw_img = Path(img_ds.get_local_copy())
# if it's a directory containing a zip, find inner zip
if raw_img.is_dir():
    inner = list(raw_img.glob("*.zip"))
    if inner:
        raw_img = inner[0]
# unzip full archive
if raw_img.is_file() and raw_img.suffix.lower() == ".zip":
    img_root = raw_img.parent / raw_img.stem
    img_root.mkdir(exist_ok=True)
    with zipfile.ZipFile(raw_img, "r") as zp:
        zp.extractall(path=img_root)
else:
    img_root = raw_img
IMAGE_ROOT = img_root
logging.info(f"Images located at: {IMAGE_ROOT}")

# 4. Student & training config
STUDENT_CONFIG = {"encoder": "google/vit-base-patch16-224-in21k", "decoder": "distilgpt2"}
TRAIN_BATCH_SIZE = 16
EVAL_BATCH_SIZE = 16
NUM_EPOCHS = 30
LR = 3.5e-5
MAX_TARGET_LEN = 64
BEAM_SIZE = 4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 5. Dataset class
def shift_tokens_right(input_ids, pad_token_id, decoder_start_token_id):
    from transformers.models.bart.modeling_bart import shift_tokens_right as _shift
    return _shift(input_ids, pad_token_id, decoder_start_token_id)

class CaptionDataset(TorchDataset):
    def __init__(self, captions_json, image_root, feature_extractor, tokenizer, max_len):
        raw = json.load(open(captions_json))
        if isinstance(raw, dict) and all(isinstance(v, str) for v in raw.values()):
            self.data = [{"image": k, "caption": v} for k, v in raw.items()]
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
        img = Image.open(self.image_root / item["image"]).convert("RGB")
        pixel = self.fe(images=img, return_tensors="pt").pixel_values.squeeze()
        toks = self.tk(
            item["caption"], padding="max_length", truncation=True,
            max_length=self.max_len, return_tensors="pt"
        )
        labels = toks.input_ids.squeeze()
        labels[labels == self.tk.pad_token_id] = -100
        decoder_input_ids = shift_tokens_right(
            toks.input_ids.squeeze(), self.tk.pad_token_id,  self.tk.bos_token_id
        )
        return {"pixel_values": pixel, "labels": labels, "decoder_input_ids": decoder_input_ids}

# 6. Load model and preprocessors
feature_extractor = ViTFeatureExtractor.from_pretrained(STUDENT_CONFIG["encoder"])
tokenizer = AutoTokenizer.from_pretrained(STUDENT_CONFIG["decoder"])
model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
    STUDENT_CONFIG["encoder"], STUDENT_CONFIG["decoder"]
)
tokenizer.add_special_tokens({"pad_token": "[PAD]"})
model.decoder.resize_token_embeddings(len(tokenizer))
model.config.pad_token_id = tokenizer.pad_token_id
model.config.decoder_start_token_id = tokenizer.bos_token_id
model.config.eos_token_id = tokenizer.eos_token_id
model.config.max_length = MAX_TARGET_LEN
model.config.no_repeat_ngram_size = 3
model.config.num_beams = BEAM_SIZE
model.config.length_penalty = 2.0
model.to(device)

# 7. Prepare datasets & dataloaders
def collate_fn(batch):
    pv = torch.stack([b['pixel_values'] for b in batch])
    labs = torch.stack([b['labels'] for b in batch])
    decoder_ids = torch.stack([b['decoder_input_ids'] for b in batch])
    return {'pixel_values': pv, 'labels': labs, 'decoder_input_ids': decoder_ids}

train_ds = CaptionDataset(TRAIN_CAPTIONS_JSON, IMAGE_ROOT, feature_extractor, tokenizer, MAX_TARGET_LEN)
val_ds = CaptionDataset(VAL_CAPTIONS_JSON, IMAGE_ROOT, feature_extractor, tokenizer, MAX_TARGET_LEN)

# 8. Metrics
bleu = evaluate.load("bleu")
rouge = evaluate.load("rouge")
cider = Cider()
spice = Spice()

def compute_metrics(eval_pred):
    preds, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    labels_mask = np.where(labels == -100, tokenizer.pad_token_id, labels)
    decoded_labels = tokenizer.batch_decode(labels_mask, skip_special_tokens=True)
    bleu_score = bleu.compute(predictions=decoded_preds, references=[[r] for r in decoded_labels])['bleu']
    rouge_res = rouge.compute(predictions=decoded_preds, references=decoded_labels)
    cider_score, _ = cider.compute_score({i:[decoded_labels[i]] for i in range(len(decoded_labels))},
                                          {i:[decoded_preds[i]] for i in range(len(decoded_preds))})
    spice_score, _ = spice.compute_score({i:[decoded_labels[i]] for i in range(len(decoded_labels))},
                                          {i:[decoded_preds[i]] for i in range(len(decoded_preds))})
    metrics = {'bleu': bleu_score, 'rouge1': rouge_res['rouge1'], 'rougeL': rouge_res['rougeL'],
               'cider': cider_score, 'spice': spice_score}
    logger.report_scalar("eval_metrics", "bleu", iteration=trainer.state.epoch, value=metrics['bleu'])
    logger.report_scalar("eval_metrics", "cider", iteration=trainer.state.epoch, value=metrics['cider'])
    return metrics

# 9. TrainingArguments & Trainer
args = Seq2SeqTrainingArguments(
    output_dir="./outputs",
    per_device_train_batch_size=TRAIN_BATCH_SIZE,
    per_device_eval_batch_size=EVAL_BATCH_SIZE,
    num_train_epochs=NUM_EPOCHS,
    learning_rate=LR,
    predict_with_generate=True,
    evaluation_strategy="epoch",
    logging_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="cider",
    greater_is_better=True,
    report_to=["tensorboard"],
    logging_dir="./tensorboard_logs"
)
trainer = Seq2SeqTrainer(
    model=model,
    args=args,
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
best_dir = "./best_model"
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
