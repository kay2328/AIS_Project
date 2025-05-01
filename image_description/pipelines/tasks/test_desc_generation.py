"""
This script generates pseudo captions for a dataset using a teacher model (e.g., Llava v1.5)
as part of a knowledge distillation pipeline. It leverages object detection annotations
from a JSON file and a predefined class mapping to build descriptive prompts.

The expected JSON annotation format (generated from your data preparation) is:
{
    "frame_IMG_4318_00002.jpg": [
        {
            "class_label": [2],
            "additional_values": [[0.5, 0.453, 1.0, 0.369125]]
        },
        ...
    ],
    ...
}
The class mapping is defined as:
    {0: "hole", 1: "pole", 2: "stairs", 3: "bottle/glass", 4: "rock", 5: "no objects"}
For each image:
  - If all annotations are “no objects” (class 5), then the prompt will be "Describe the scene."
  - Otherwise, the prompt is built from the object names (ignoring “no objects”).
The teacher model then uses the image along with the text prompt to generate a caption.
The resulting pseudo captions are saved to an output JSON file.
"""
import os
import json
import logging
from clearml import Task, Dataset, Model
from pathlib import Path
import torch
import sys
import zipfile
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../src')))
from enigmaai.config import Project, ConfigFactory
from enigmaai.desc_prep_util import find_dir_with_files
from enigmaai.desc_caption_util import desc_generation

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
# get project configurations
project = ConfigFactory.get_config(Project.SCENE_DESCRIPTION)
project_name = project.get('project-name')

"""
Task initiation
"""
# Initiate the task 4 to generate mapping of image name and reference description for student model to learn later in the pipeline for train set
task = Task.init(project_name=project_name, 
                task_name="step4_desc_evalcaption_generation",
                task_type=Task.TaskTypes.data_processing)
params = {
    'dataset_id': '',                # specific version of the dataset
    'dataset_name': 'Desc_Eval_Dataset',              # latest registered dataset
    'eval_dataset_id': '', #'e19da140dd6a479c864dd7bdf930918d',#'2231b5b121924ed684d6560cf6839619',     # specific version of the dataset
    'eval_dataset_name': 'eval_dataset_zip'
}

logger = task.get_logger()
task.connect(params)
task.execute_remotely(queue_name="desc_preparation")

dataset_id = params['dataset_id']
dataset_name = params['dataset_name']
img_dataset_id = params['eval_dataset_id']
img_dataset_name = params['eval_dataset_name']
# validate task input params
if not dataset_id and not dataset_name:
    task.mark_completed(status_message="No mapping dataset provided. Nothing to evaluate on. Ensure to execute task 2")
    exit(0)
if not img_dataset_id and not img_dataset_name:
    task.mark_completed(status_message="No image dataset provided. Nothing to evaluate on.")
    exit(0)

"""
Fetch images data to generate descriptions for evaluation
"""
# get the image dataset from "Detection project - base_dataset_zip"
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
Fetch mapping json from task 2
"""
if dataset_id: 
    # download the latest registered dataset
    server_dataset = Dataset.get(dataset_id=dataset_id, only_completed=True, alias="desc_testprep_mapping")
elif dataset_name: 
    # download the latest registered dataset
    server_dataset = Dataset.get(dataset_name=dataset_name, dataset_project=project_name, only_completed=True, alias="desc_testprep_mapping")

extract_path = server_dataset.get_local_copy()          
print(f"Downloaded dataset name: {server_dataset.name} id: ({server_dataset.id}) to: {extract_path}")
extract_path = Path(extract_path)
annot_file = extract_path / "desc_prep_test_dataset.json"

if not annot_file.exists():
    # print out what _is_ in that folder to see where JSON landed
    logging.error(f"Expected JSON not found! Contents are:\n{list(extract_path.iterdir())}")
    raise FileNotFoundError(f"{annot_file} does not exist")
with annot_file.open("r") as f:
    mapping = json.load(f)
logging.info(f"Loaded {len(mapping)} image→annotation entries")

"""
Preparing the Caption/Description mapping data to evaluate student model
"""
# build a Path to the JSON file under a subfolder "Desc_Caption_BaseDataset"
out_dir = extract_path / project_name / "Desc_Caption_EvalDataset"
# ensure the output directory exists
out_dir.mkdir(parents=True, exist_ok=True)

out_file = out_dir / "desc_caption_testdataset.json"
# if an old JSON exists, delete it
if out_file.exists():
    logging.info(f"Removing old mapping at {out_file}")
    out_file.unlink()

model_name = "llava-hf/llava-1.5-7b-hf"
device = "cuda" if torch.cuda.is_available() else "cpu"
# Define the class mapping.
class_mapping = {
    0: "hole",
    1: "pole",
    2: "stairs",
    3: "bottle/glass",
    4: "rock",
    5: "no objects"
}

desc_generation(model_name, device, class_mapping, annot_file, images_dir, out_file)
# upload prepared dataset to ClearML server
test_cap_dataset = Dataset.create(dataset_project=project_name, dataset_name="Desc_Caption_EvalDataset")
test_cap_dataset.add_files(path=str(out_file))
logging.info('Uploading dataset in the background')
test_cap_dataset.upload()
test_cap_dataset.finalize()

task.set_parameter("output_dataset_project", test_cap_dataset.project)
task.set_parameter("output_dataset_id", test_cap_dataset.id)
task.set_parameter("output_dataset_name", test_cap_dataset.name)