import torch
from torch.nn.utils.rnn import pad_sequence
from transformers.models.bart.modeling_bart import shift_tokens_right
from torch.utils.data import Dataset
import json
import numpy as np
import os
from PIL import Image
from transformers import Seq2SeqTrainer
import evaluate
from pycocoevalcap.cider.cider import Cider
#from pycocoevalcap.spice.spice import Spice

"""
Dataset class
"""
class CaptionDataset(Dataset):
    def __init__(self, captions_json, image_root, feature_extractor, tokenizer, max_len):
        """
        captions_json: list of {"image": "path.jpg", "caption": "…"}
        """
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

"""
Custom Seq2Seq Trainer
"""
class CleanSeq2SeqTrainer(Seq2SeqTrainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # pop Trainer-only args so they don't get forwarded into your model
        inputs.pop("num_items_in_batch", None)
        # now call the parent (it expects just model, inputs, return_outputs)
        return super().compute_loss(model, inputs, return_outputs)
"""
Custom Data Collator
"""
class CustomDataCollator:
    def __init__(self, model, tokenizer, device):
        self.model     = model
        self.tokenizer = tokenizer
        self.device    = device
    def __call__(self, batch):
        # 1) Stack the images
        pixel_values = torch.stack([ex["pixel_values"] for ex in batch])
        # 2) Pad & mask the labels
        label_seqs = [ex["labels"] for ex in batch]
        labels     = pad_sequence(label_seqs, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels_mask = labels.clone()
        labels_mask[labels_mask == self.tokenizer.pad_token_id] = -100
        # 3) Build decoder_input_ids using the model’s config
        decoder_input_ids = shift_tokens_right(labels, pad_token_id=self.tokenizer.pad_token_id, decoder_start_token_id=self.model.config.decoder_start_token_id)
        return {
            "pixel_values":      pixel_values.to(self.device),
            "labels":            labels_mask.to(self.device),
            "decoder_input_ids": decoder_input_ids.to(self.device),
        }

"""
Metrics for evaluation
"""
class ComputeMetrics:
    def __init__(self, tokenizer):
        self.tokenizer   = tokenizer
        self.bleu_metric = evaluate.load("bleu")
        self.rouge_metric= evaluate.load("rouge")
        self.cider       = Cider()
        #self.spice       = Spice()
    def __call__(self, eval_pred):
        preds, labels = eval_pred
        tk = self.tokenizer
        decoded_preds  = tk.batch_decode(preds, skip_special_tokens=True)
        labels_clean   = np.where(labels == -100, tk.pad_token_id, labels)
        decoded_labels = tk.batch_decode(labels_clean, skip_special_tokens=True)

        bleu = self.bleu_metric.compute(predictions=decoded_preds, references=[[ref] for ref in decoded_labels])["bleu"]
        rouge_res = self.rouge_metric.compute(predictions=decoded_preds, references=decoded_labels)

        refs_dict = {i: [decoded_labels[i]] for i in range(len(decoded_labels))}
        hyps_dict = {i: [decoded_preds[i]]   for i in range(len(decoded_preds))}
        cider_score, _ = self.cider.compute_score(refs_dict, hyps_dict)
        #spice_score, _ = self.spice.compute_score(refs_dict, hyps_dict)

        return {
            "bleu":   bleu,
            "rouge1": rouge_res["rouge1"],
            "rougeL": rouge_res["rougeL"],
            "cider":  cider_score,
            #"spice":  spice_score,
        }