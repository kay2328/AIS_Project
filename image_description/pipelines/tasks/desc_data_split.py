import os
from pathlib import Path
import logging
import json
import sys
from sklearn.model_selection import train_test_split
from clearml import Task, Dataset
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../src')))
from enigmaai.config import Project, ConfigFactory

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
# get project configurations
project = ConfigFactory.get_config(Project.SCENE_DESCRIPTION)
project_name = project.get('project-name')

task = Task.init(project_name=project_name, 
                task_name="step5_desc_split_data", 
                task_type=Task.TaskTypes.data_processing)
params = {
    'cap_dataset_id': '',
    'cap_dataset_name': 'Desc_Caption_BaseDataset',
    'random_state': 42,
    'val_size': 0.2,
}
logger = task.get_logger()
task.connect(params)
#task.execute_remotely(queue_name="desc_preparation")

"""
Fetching captions dataset from task 3
"""
# download dataset 
dataset_id = task.get_parameters()['General/cap_dataset_id']
dataset_name = task.get_parameters()['General/cap_dataset_name']

# validate task input params
if not dataset_id and not dataset_name:
    task.mark_completed(status_message="No annotation dataset provided. Nothing to train on.")
    exit(0)
try: 
    # download the latest registered dataset
    server_dataset = Dataset.get(dataset_id=dataset_id, only_completed=True, alias="desc_cap_data")
except ValueError:
    # download the latest registered dataset
    server_dataset = Dataset.get(dataset_name=dataset_name, dataset_project=project_name, only_completed=True, alias="desc_cap_data")

extract_path = server_dataset.get_local_copy()          
print(f"Downloaded dataset name: {server_dataset.name} id: ({server_dataset.id}) to: {extract_path}")
extract_path = Path(extract_path)
caption_file = extract_path / "desc_caption_basedataset.json"
mapping = json.loads(open(caption_file, "r").read())
logging.info(f"Loaded {len(mapping)} captions from {caption_file}")

"""
Split dataset after including reference descriptions, downloaded from ClearML server into train and val
"""
# Prepare list of stems
all_stems = [Path(fn).stem for fn in mapping.keys()]
# split sizes
val_size = task.get_parameter()['General/val_size']
random_state = task.get_parameter()['General/random_state']
# split train, val, test sets according task params
train_stems, val_stems = train_test_split(all_stems, test_size = val_size, random_state=random_state)
# Build split mappings
train_map = {f"{s}.jpg": mapping[f"{s}.jpg"] for s in train_stems}
val_map   = {f"{s}.jpg": mapping[f"{s}.jpg"] for s in val_stems}

"""
Move files to train, val folders
"""
# destination path for split dataset
dest_path = Path(extract_path / project_name/ "Desc_Split_dataset")        
os.makedirs(dest_path, exist_ok=True)
for split, split_map in [("train", train_map), ("val", val_map)]:
    dst = dest_path / f"{split}.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(dst, "w") as f:
        json.dump(split_map, f, indent=2)
    logging.info(f"Wrote {len(split_map)} entries to {dst}")

# (Optional) Copy images into split folders
# for split, stems in [("train", train_stems), ("val", val_stems), ("test", test_stems)]:
#     img_out = dest_path / split / "images"
#     img_out.mkdir(parents=True, exist_ok=True)
#     for s in stems:
#         src = images_root / f"{s}.jpg"
#         if src.exists():
#             shutil.copy(src, img_out / src.name)

# Upload the splits as a new ClearML dataset
split_data = Dataset.create(
    dataset_project=project_name,
    dataset_name="Desc_Split_dataset")
split_data.add_files(str(dest_path))
split_data.upload()
split_data.finalize()
logging.info(f"Uploaded split dataset id={split_data.id}")

# Save outputs
task.set_parameter("output_dataset_project", split_data.project)
task.set_parameter("output_dataset_id", split_data.id)
task.set_parameter("output_dataset_name", split_data.name)