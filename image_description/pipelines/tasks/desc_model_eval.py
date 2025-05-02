import sys
import os
from clearml import Task, Dataset, Model
#Task.add_requirements("requirements.txt")
from pathlib import Path
import logging
import torch
from transformers import (
    VisionEncoderDecoderModel,
    ViTFeatureExtractor,
    AutoTokenizer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)
from PIL import Image
import tempfile, zipfile
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../src')))
from enigmaai import util
from enigmaai.config import Project, ConfigFactory
from enigmaai.desc_prep_util import find_dir_with_files
from enigmaai.desc_util import CaptionDataset, ComputeMetrics, CustomDataCollator
import subprocess
# Install absl-py on the fly so evaluate.load("rouge") can import it
subprocess.check_call([sys.executable, "-m", "pip", "install", "absl-py"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "rouge-score"])

# get project configurations
project = ConfigFactory.get_config(Project.SCENE_DESCRIPTION)
project_name = project.get('project-name')
working_dir = Path(tempfile.mkdtemp()) / project_name
working_dir.mkdir(parents=True, exist_ok=True)    
print("Working temp directory at:", working_dir)

"""
Initialize task for model evaluation
"""
# Initialize clearl task
task = Task.init(project_name=project_name, 
                task_name="step7_desc_model_evaluation", 
                task_type=Task.TaskTypes.qc)
params = {
    'dataset_id': '', #'f6865cde77d843eb93829a268b2adeaf',                # specific version of the eval caption dataset
    'dataset_name': 'Desc_Caption_EvalDataset',              # latest registered dataset
    'eval_dataset_id': '', #'e19da140dd6a479c864dd7bdf930918d',#'2231b5b121924ed684d6560cf6839619',     # specific version of the dataset
    'eval_dataset_name': 'eval_dataset_zip',
    'desc_draft_model_id': '6d7b1be6082543efaade2fb24df63ae3', #'36939d5f9c7a41a2b75ee2110e155144',       # the unpublished model to evaluate 
    'desc_pub_model_name': 'student_desc_model',       # the published model name for comparison
}
task.connect(params)
task.execute_remotely(queue_name="desc_preparation")
task_params = task.get_parameters()
logging.info("model_eval params=", task_params)

dataset_id = task.get_parameters()['General/dataset_id']
dataset_name = task.get_parameters()['General/dataset_name']
img_dataset_id = task.get_parameters()['General/eval_dataset_id']
img_dataset_name = task.get_parameters()['General/eval_dataset_name']
draft_model_id = task.get_parameters()['General/desc_draft_model_id']
pub_model_name = task.get_parameters()['General/desc_pub_model_name']

# validate task input params
if not dataset_id and not dataset_name:
    task.mark_completed(status_message="No dataset provided. Nothing to evaluate on. Ensure to execute task 4")
    exit(0)
if not img_dataset_id and not img_dataset_name:
    task.mark_completed(status_message="No image dataset provided. Nothing to evaluate.")
    exit(0)

"""
Reference description/caption Dataset for evaluation - desc_caption_testdataset.json
"""
# 2. Fetch JSON dataset from "Desc_Caption_EvalDataset" under "Description" project
try: 
    # download the latest registered caption eval dataset
    server_dataset = Dataset.get(dataset_id=dataset_id, only_completed=True, alias="eval_cap_dataset")
except ValueError:
    # download the latest registered dataset
    server_dataset = Dataset.get(dataset_name=dataset_name, dataset_project=project_name, only_completed=True, alias="eval_cap_dataset")
eval_cap_path = server_dataset.get_local_copy()          
print(f"Downloaded dataset name: {server_dataset.name} id: ({server_dataset.id}) to: {eval_cap_path}")
eval_cap_path = Path(eval_cap_path)
test_json = eval_cap_path / "desc_caption_testdataset.json"
logging.info(f"Split JSONs located at: {eval_cap_path}")

"""
Fetching image dataset for evaluation
"""
# Fetch images ZIP from "eval_dataset_zip" under "Detection" project
try: 
    # download the latest registered dataset
    server_dataset = Dataset.get(dataset_id=img_dataset_id, only_completed=True, alias="eval_img_dataset")
except ValueError:
    server_dataset = Dataset.get(dataset_name=img_dataset_name, dataset_project="Detection", only_completed=True, alias="eval_img_dataset")
extract_path = server_dataset.get_local_copy()          
print(f"Downloaded eval image dataset name: {server_dataset.name} id: ({server_dataset.id}) to: {extract_path}")

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
Model evaluation
"""
max_target_len = 64
eval_batch_size = 16
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
out_dir = working_dir / "outputs_eval" 

# no eval caption and image dataset provided
if not test_json or not images_dir:
    task.mark_completed(status_message="No dataset provided for evaluation.")
    exit(0)
# no model provided for evaluation
if not draft_model_id:
    raise ValueError("Missing new/draft model. Please provide draft_model_id.")
# Mandatory input param
if not pub_model_name:
    raise ValueError("Missing model. Please provide pub_model_name.")

# fetch the draft model path for evaluation    
draft_model = Model(model_id=draft_model_id)    
print(f"Found draft model name:{draft_model.name} id:{draft_model.id}")
draft_model_path = draft_model.get_local_copy(raise_on_error=True)
print(f"Downloaded published model name: {draft_model.name} id:{draft_model.id} to: {draft_model_path}")
  
# fetch the published best model path
server_models = Model.query_models(model_name=pub_model_name, only_published=True)
if not server_models:     
    best_model = draft_model # best published model not found, use first draft as best
    print(f"No published model found, use draft as the best model name:{draft_model.name} id:{draft_model.id}")
    pub_model_path = best_model.get_local_copy(raise_on_error=True)
    print(f"Downloaded draft model name: {best_model.name} id:{best_model.id} to: {pub_model_path}")
else:    
    # best published model found
    pub_model = server_models[0] # get the most recent one
    task.set_parameter("pub_model_id", pub_model.id)
    print(f"Found published model name:{pub_model.name} id:{pub_model.id}")
    pub_model_path = pub_model.get_local_copy(raise_on_error=True)
    print(f"Downloaded published model name: {pub_model.name} id:{pub_model.id} to: {pub_model_path}")
  
def load_model(model_path):
    tmp_dir = Path(tempfile.mkdtemp())
    # Unzip all files there
    with zipfile.ZipFile(model_path, 'r') as zf:
        zf.extractall(tmp_dir)
    model = VisionEncoderDecoderModel.from_pretrained(tmp_dir)
    tokenizer = AutoTokenizer.from_pretrained(tmp_dir)
    feature_extractor = ViTFeatureExtractor.from_pretrained(tmp_dir)
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    model.decoder.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id            = tokenizer.pad_token_id
    model.config.decoder_start_token_id  = tokenizer.bos_token_id
    model.config.eos_token_id            = tokenizer.eos_token_id
    model.config.max_length              = max_target_len
    model.config.num_beams               = 4
    model.config.length_penalty          = 2.0
    model.config.early_stopping          = True
    model.to(device)
    return model, feature_extractor, tokenizer

def generate_caption(image_path, model, fe, tk, max_new_tokens=40, num_beams=4, decode=True):
    model.eval()
    img = Image.open(image_path).convert("RGB")
    inputs = fe(images=img, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            pixel_values=inputs.pixel_values,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            pad_token_id=tk.pad_token_id,
            eos_token_id=tk.eos_token_id,
        )
    return tk.decode(output_ids[0], skip_special_tokens=True) if decode else output_ids[0]

# Evaluation Setup 
eval_args = Seq2SeqTrainingArguments(
    output_dir=out_dir,
    run_name="test_student_model",# temp directory
    per_device_eval_batch_size=eval_batch_size,
    predict_with_generate=True,
    do_train=False,
    do_eval=True,
    dataloader_pin_memory=False,
    report_to=[]
)

# function to run evaluation 
def run_eval(split_name, captions_json, model, feature_extractor, tokenizer):
    ds = CaptionDataset(captions_json, images_dir, feature_extractor, tokenizer, max_target_len)
    collator_fn = CustomDataCollator(model, tokenizer, device)
    compute_metrics_fn = ComputeMetrics(tokenizer)
    trainer = Seq2SeqTrainer(
        model=model,
        args=eval_args,
        data_collator=collator_fn,
        compute_metrics=compute_metrics_fn,
        eval_dataset=ds)
    metrics = trainer.evaluate()
    print(f"\n=== {split_name} metrics ===")
    for k,v in metrics.items():
        print(f"{k}: {v:.4f}")
    return metrics

"""
Evaluation and comparison between draft and published best model
"""
# evaluate the draft model    
draft_model_hf, draft_feature_extractor, draft_tokenizer = load_model(draft_model_path)
draft_metrics = run_eval("Test", test_json, draft_model_hf, draft_feature_extractor, draft_tokenizer)
# evaluate the published best model
pub_model_hf, pub_feature_extractor, pub_tokenizer = load_model(pub_model_path)
pub_metrics = run_eval("Test", test_json, pub_model_hf, pub_feature_extractor, pub_tokenizer)    
# show metrics for comparision
print("keys=", draft_metrics.keys)
print("draft_metrics=", draft_metrics)
print("pub_metrics=", pub_metrics)

# compare and select the best model
best_model = pub_model if pub_metrics["eval_cider"] > draft_metrics["eval_cider"] else draft_model    
# upload results reference for report analysis 
#task.upload_artifact(name=f"{pub_model_name}_eval_config", artifact_object=eval_args)
task.upload_artifact(name=f"{pub_model_name}_draft_metrics", artifact_object=draft_metrics)
task.upload_artifact(name=f"{pub_model_name}_pub_metrics", artifact_object=pub_metrics)
    
# check the best model
if best_model.id == draft_model.id: 
    print(f"Draft model is the best model name:{best_model.name} id:{best_model.id}")
else: # new model not better, nothing to publish
    print(f"Existing published model is the best name:{best_model.name} id:{best_model.id}.")

# show output    
print("best_model_project:", project_name)
print("best_model_id:", best_model.id)
print("best_model_name:", best_model.name)
print("best_model_variant:", best_model.name)

# task output info
task.set_parameter("best_model_project", project_name)
task.set_parameter("best_model_id", best_model.id)
task.set_parameter("best_model_name", best_model.name)