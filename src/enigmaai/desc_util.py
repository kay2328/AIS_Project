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
from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer
from enigmaai import util

"""
Load Model components for training
"""
class StudentModelLoader:
    """
    Loads a vision‐encoder‐decoder student model along with its
    feature extractor and tokenizer, using a standard config.
    """

    DEFAULT_CONFIG = {
        "encoder": "google/vit-base-patch16-224-in21k",
        "decoder": "distilgpt2",
        "max_target_len": 64,
        "beam_size": 4,
        "length_penalty": 0.7,
        "no_repeat_ngram_size": 3,
        "early_stopping": False,
    }

    def __init__(self, config: dict = None, device: torch.device = None):
        cfg = dict(self.DEFAULT_CONFIG)
        if config:
            cfg.update(config)
        self.encoder_name = cfg["encoder"]
        self.decoder_name = cfg["decoder"]
        self.max_target_len       = cfg["max_target_len"]
        self.num_beams            = cfg["beam_size"]
        self.length_penalty       = cfg["length_penalty"]
        self.no_repeat_ngram_size = cfg["no_repeat_ngram_size"]
        self.early_stopping       = cfg["early_stopping"]

        self.device = util.get_device_name()
        self._load()

    def _load(self):
        # 1) load preprocessors
        self.feature_extractor = ViTFeatureExtractor.from_pretrained(self.encoder_name)
        self.tokenizer         = AutoTokenizer.from_pretrained(self.decoder_name)
        # 2) build model 
        self.model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
            self.encoder_name,
            self.decoder_name)
        # 3) special tokens & config tweaks
        self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        self.model.decoder.resize_token_embeddings(len(self.tokenizer))

        cfg = self.model.config
        cfg.pad_token_id           = self.tokenizer.pad_token_id
        cfg.decoder_start_token_id = self.tokenizer.bos_token_id
        cfg.eos_token_id           = self.tokenizer.eos_token_id
        cfg.vocab_size             = cfg.decoder.vocab_size
        cfg.max_length             = self.max_target_len
        cfg.num_beams              = self.num_beams
        cfg.length_penalty         = self.length_penalty
        cfg.no_repeat_ngram_size   = self.no_repeat_ngram_size
        cfg.early_stopping         = self.early_stopping

        # 4) move to device
        self.model.to(self.device)

    def get_components(self):
        """
        Returns
        -------
        model : VisionEncoderDecoderModel
        feature_extractor : ViTFeatureExtractor
        tokenizer : AutoTokenizer
        """
        return self.model, self.feature_extractor, self.tokenizer

"""
Dataset class
"""
class CaptionDataset(Dataset):
    def __init__(self, captions_json, image_root, feature_extractor, tokenizer, max_len=64):
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
    def __init__(self, model, tokenizer):
        self.model     = model
        self.tokenizer = tokenizer
        self.device    = util.get_device_name()
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