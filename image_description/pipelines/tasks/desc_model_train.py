from clearml import Task, Dataset, OutputModel
#Task.add_requirements("requirements.txt")
import os
import logging
import zipfile
from pathlib import Path
import torch
from transformers import (
    VisionEncoderDecoderModel,
    ViTFeatureExtractor,
    AutoTokenizer,
    Seq2SeqTrainingArguments)
import matplotlib.pyplot as plt
import torch
import sys
import tempfile
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../src')))
from enigmaai.config import Project, ConfigFactory
from enigmaai.desc_util import CaptionDataset, ComputeMetrics, CustomDataCollator, CleanSeq2SeqTrainer
from enigmaai.desc_prep_util import find_dir_with_files
import subprocess
# Install absl-py on the fly so evaluate.load("rouge") can import it
subprocess.check_call([sys.executable, "-m", "pip", "install", "absl-py"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "rouge-score"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "tensorboardX"])

"""
Initial configurations
"""
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
# get project configurations
project = ConfigFactory.get_config(Project.SCENE_DESCRIPTION)
project_name = project.get('project-name')

"""
Initialize task for model training
"""
# 1. Initialize ClearML Task
task = Task.init(
    project_name=project_name,
    task_name="step6_desc_model_training",
    task_type=Task.TaskTypes.training
)
logger = task.get_logger()
params = {
    'split_dataset_id': '',                # specific version of the dataset
    'split_dataset_name': 'Desc_Split_dataset',              # latest registered dataset
    'base_dataset_id': '', #'26083b24ab0c47219a5e4f3fe026b085',#'2231b5b121924ed684d6560cf6839619',     # specific version of the dataset
    'base_dataset_name': 'base_dataset_zip'
}
task.connect(params)
task.execute_remotely(queue_name="desc_preparation")

dataset_id = params['split_dataset_id']
dataset_name = params['split_dataset_name']
img_dataset_id = params['base_dataset_id']
img_dataset_name = params['base_dataset_name']
# validate task input params
if not dataset_id and not dataset_name:
    task.mark_completed(status_message="No dataset provided. Nothing to train on. Ensure to execute task 5")
    exit(0)
if not img_dataset_id and not img_dataset_name:
    task.mark_completed(status_message="No image dataset provided. Nothing to train on.")
    exit(0)

"""
Fetching Split Annotation Dataset for training
"""
# 2. Fetch split JSON dataset from "Desc_Split_dataset" under "Description" project
try: 
    # download the latest registered dataset
    server_dataset = Dataset.get(dataset_id=dataset_id, only_completed=True, alias="desc_split_data")
except ValueError:
    # download the latest registered dataset
    server_dataset = Dataset.get(dataset_name=dataset_name, dataset_project=project_name, only_completed=True, alias="desc_split_data")

extract_path = server_dataset.get_local_copy()          
print(f"Downloaded dataset name: {server_dataset.name} id: ({server_dataset.id}) to: {extract_path}")
extract_path = Path(extract_path)
train_json = extract_path / "train.json"
val_json = extract_path / "val.json"
logging.info(f"Split JSONs located at: {extract_path}")
if not train_json.exists() or not val_json.exists():
    # print out what _is_ in that folder to see where JSON landed
    logging.error(f"Expected JSON not found! Contents are:\n{list(extract_path.iterdir())}")
    raise FileNotFoundError(f"JSON File with reference descriptions does not exist")

"""
Fetching image dataset for training
"""
# 3. Fetch images ZIP from "base_dataset_zip" under "Detection" project
try: 
    # download the latest registered dataset
    server_dataset = Dataset.get(dataset_id=img_dataset_id, only_completed=True, alias="base_dataset")
except ValueError:
    # download the latest registered dataset
    server_dataset = Dataset.get(dataset_name=img_dataset_name, dataset_project="Detection", only_completed=True, alias="base_dataset")
extract_path = server_dataset.get_local_copy()          
print(f"Downloaded base dataset name: {server_dataset.name} id: ({server_dataset.id}) to: {extract_path}")

raw_path = Path(extract_path)
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
images_dir = find_dir_with_files(extract_path, "images")
logging.info(f"Images downloaded to: {images_dir}")

"""
Student model training configuration and set up
"""
# 4. Student & training config
STUDENT_CONFIG = {"encoder": "google/vit-base-patch16-224-in21k", "decoder": "distilgpt2"}
train_batch_size = 16
eval_batch_size = 16
num_epochs = 1 
lr = 1e-4
weight_decay=0.01
max_target_len = 64
beam_size = 4
encoder = STUDENT_CONFIG["encoder"]
decoder = STUDENT_CONFIG["decoder"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(encoder, decoder):
    feature_extractor = ViTFeatureExtractor.from_pretrained(encoder)
    tokenizer = AutoTokenizer.from_pretrained(decoder)
    model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(encoder, decoder)
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    # Resize the decoder’s embeddings to match new vocab size
    model.decoder.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id
    # Tie special tokens
    model.config.decoder_start_token_id = tokenizer.bos_token_id
    model.config.eos_token_id           = tokenizer.eos_token_id
    model.config.vocab_size             = model.config.decoder.vocab_size
    model.config.max_length             = max_target_len
    model.config.early_stopping         = True
    model.config.no_repeat_ngram_size   = 3
    model.config.num_beams              = beam_size
    model.config.length_penalty         = 2.0
    model.to(device)
    return model, feature_extractor, tokenizer

# 6. Load model and preprocessors
model, feature_extractor, tokenizer = load_model(encoder, decoder)
train_ds = CaptionDataset(train_json, images_dir, feature_extractor, tokenizer, max_target_len)
val_ds   = CaptionDataset(val_json, images_dir, feature_extractor, tokenizer, max_target_len)
collator_fn = CustomDataCollator(model, tokenizer, device)
compute_metrics_fn = ComputeMetrics(tokenizer)

"""
Model Training
"""
working_dir = Path(tempfile.mkdtemp()) / project_name
working_dir.mkdir(parents=True, exist_ok=True)    
print("Working temp directory at:", working_dir)
trainout_dir = working_dir / "outputs" / "models"
tensorboard_dir = working_dir/ "outputs" / "tensorboard_logs"
trainout_dir.mkdir(parents=True, exist_ok=True)
tensorboard_dir.mkdir(parents=True, exist_ok=True)
# TrainingArguments & Trainer
training_args = Seq2SeqTrainingArguments(
    output_dir=trainout_dir,
    per_device_train_batch_size=train_batch_size,
    per_device_eval_batch_size=eval_batch_size,
    num_train_epochs=num_epochs,
    learning_rate=lr,
    weight_decay=weight_decay, # L2 regularization on all weights
    predict_with_generate=True,
    eval_strategy="epoch",     # run eval (and log eval_loss) at end of each epoch
    logging_strategy="epoch",  # log train loss every epoch             
    save_strategy="epoch",
    save_total_limit=2,  
    label_smoothing_factor=0.1,   # soften the training targets
    fp16=torch.cuda.is_available(),
    load_best_model_at_end=True,  
    metric_for_best_model="cider", # pick the checkpoint with highest cider
    greater_is_better=True,
    report_to=["tensorboard"],    # enable TensorBoard
    logging_dir=tensorboard_dir
)
trainer = CleanSeq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    data_collator=collator_fn,
    compute_metrics=compute_metrics_fn
)

# Train
results = trainer.train()

"""
Plotting loss curve graphs
"""
# Plot loss curve and log to ClearML
history = trainer.state.log_history
epochs = [h['epoch'] for h in history if 'loss' in h]
losses = [h['loss'] for h in history if 'loss' in h]
f, ax = plt.subplots()
ax.plot(epochs, losses, marker='o')
ax.set_xlabel('Epoch')
ax.set_ylabel('Training Loss')
ax.set_title('Loss Curve')
logger.report_matplotlib_figure("training_plot", "loss_curve", f)

"""
Saving best model after training and uploading artifacts
"""
# Save best model and artifacts
best_ckpt = trainer.state.best_model_checkpoint
task.get_logger().report_text(f"Best checkpoint at: {best_ckpt}")
best_model = VisionEncoderDecoderModel.from_pretrained(best_ckpt)
best_dir = working_dir / "outputs"/ "best_model"
best_dir.mkdir(parents=True, exist_ok=True)
best_model.save_pretrained(best_dir)
tokenizer.save_pretrained(best_dir)
feature_extractor.save_pretrained(best_dir)
task.upload_artifact(name="best_model", artifact_object=best_dir)
result_file = working_dir / "outputs" / "results.csv"
task.upload_artifact(name="results", artifact_object=result_file)

import shutil
zip_path = shutil.make_archive(str(best_dir), 'zip', root_dir=str(best_dir))

# Register best model as an OutputModel
task = Task.current_task()
output_model = OutputModel(
    task=task,
    name="student_desc_model",    
    framework="pytorch"
)
# Upload the ZIP as the model weights
output_model.update_weights(weights_filename=zip_path)
print("Registered model id:", output_model.id)
logging.info("Student training on ClearML complete.")