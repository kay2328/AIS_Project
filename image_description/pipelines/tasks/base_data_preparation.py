from clearml import Task, Dataset
import logging, zipfile
import os, sys
from pathlib import Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../src')))
from enigmaai.config import Project, ConfigFactory
from enigmaai.desc_prep_util import create_mapping, find_dir_with_files
"""
Map the images from latest  dataset stored on ClearML server to their corresponding annotation/labels files.
Each annotation file (stored in a separate labels folder) may contain one or more lines,
each with the following format:
    <class_label> <val1> <val2> <val3> <val4>
For example:
    0 0.705242 0.791633 0.058828 0.075641
    0 0.445586 0.652133 0.097484 0.156297
If a label file is empty (i.e. no lines are present), this script will instead assign a
default annotation indicating no objects detected. In that case, the default annotation is:
    "class_label": [5], "additional_values": []
The final output is a JSON file mapping each image filename to a list of annotation dictionaries.
"""

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
# get project configurations
project = ConfigFactory.get_config(Project.SCENE_DESCRIPTION)
project_name = project.get('project-name')

task = Task.init(project_name=project_name, 
                task_name="step1_desc_basedata_preparation",
                task_type=Task.TaskTypes.data_processing)
params = {
    'base_dataset_id': '', #'26083b24ab0c47219a5e4f3fe026b085',#'2231b5b121924ed684d6560cf6839619',     # specific version of the dataset
    'base_dataset_name': 'base_dataset_zip',               # latest registered dataset
}

logger = task.get_logger()
task.connect(params)
#task.execute_remotely(queue_name="desc_preparation")
dataset_id = task.get_parameters()['General/base_dataset_id']
dataset_name = task.get_parameters()['General/base_dataset_name']

# validate task input params
if not dataset_id and not dataset_name:
    task.mark_completed(status_message="No dataset provided. Nothing to train on.")
    exit(0)
if dataset_id: 
    # download the latest registered dataset
    server_dataset = Dataset.get(dataset_id=dataset_id, only_completed=True, alias="base_dataset")
elif dataset_name: 
    # download the latest registered dataset
    server_dataset = Dataset.get(dataset_name=dataset_name, dataset_project="Detection", only_completed=True, alias="base_dataset")

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

"""
Prepare dataset for training.
"""
# get image file prefix that has corresponding labels
images_dir = find_dir_with_files(extract_path, "images")
labels_dir = find_dir_with_files(extract_path, "labels")
logging.info(f"Images located at: {images_dir}")
logging.info(f"Labels located at: {labels_dir}")

# build a Path to the JSON file under a subfolder "Desc_Dataset"
out_dir  = extract_path / project_name / "Desc_Base_Dataset"
# ensure the output directory exists
out_dir.mkdir(parents=True, exist_ok=True)
out_file = out_dir / "desc_prep_base_dataset.json"
# if an old JSON exists, delete it
if out_file.exists():
    logging.info(f"Removing old mapping at {out_file}")
    out_file.unlink()

create_mapping(images_dir, labels_dir, out_file)
# upload prepared dataset to ClearML server
base_dataset = Dataset.create(
    dataset_project=project_name, dataset_name="Desc_Base_Dataset"
)
base_dataset.add_files(path=str(out_file))
print('Uploading dataset in the background')
base_dataset.upload()
base_dataset.finalize()

task.set_parameter("output_dataset_project", base_dataset.project)
task.set_parameter("output_dataset_id", base_dataset.id)
task.set_parameter("output_dataset_name", base_dataset.name)
