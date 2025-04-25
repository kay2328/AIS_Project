import os
#import shutil
from pathlib import Path
import logging
import json
from sklearn.model_selection import train_test_split
from clearml import Task, Dataset, StorageManager


# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
"""
Split dataset after including reference descriptions, downloaded from ClearML server into train, val and test
"""
project_name="Description"
task = Task.init(project_name=project_name, 
                task_name="step3_desc_split_data", 
                task_type=Task.TaskTypes.data_processing)

params = {
    'cap_dataset_id': '',
    'cap_dataset_name': 'Desc_Caption_Dataset',
    'random_state': 42,
    'val_size': 0.15,
    'test_size': 0.15,
}

# logger = task.get_logger()
task.connect(params)
task.execute_remotely(queue_name="desc_preparation")

# download dataset 
dataset_id = params['cap_dataset_id']
dataset_name = params['cap_dataset_name']

# validate task input params
if not dataset_id and not dataset_name:
    task.mark_completed(status_message="No dataset provided. Nothing to train on.")
    exit(0)
if dataset_name: 
    # download the latest registered dataset
    server_dataset = Dataset.get(dataset_name=dataset_name, dataset_project=project_name, only_completed=True, alias="desc_split data")

extract_path = server_dataset.get_local_copy()          
print(f"Downloaded dataset name: {server_dataset.name} id: ({server_dataset.id}) to: {extract_path}")
extract_path = Path(extract_path)
caption_file = extract_path / "desc_caption_dataset.json"
mapping = json.loads(open(caption_file, "r").read())
logging.info(f"Loaded {len(mapping)} captions from {caption_file}")

#get the image dataset from "Detection project- base_dataset"
images_data = Dataset.get(
    dataset_name="base_dataset",
    dataset_project="Detection",
    only_completed=True,
    alias="base_images"  
)
images_root = Path(images_data.get_local_copy())
images_dir  = images_root / "images"
logging.info(f"Images downloaded to: {images_dir}")

# Prepare list of stems
all_stems = [Path(fn).stem for fn in mapping.keys()]

"""
Split dataset to train, val, test
"""

# split sizes
val_size = params['val_size']
test_size = params['test_size']
random_state = params['random_state']

# split train, val, test sets according task params
train_stems, remaining_stems = train_test_split(all_stems, 
                                         test_size = val_size + test_size, 
                                         random_state=random_state)

val_stems, test_stems = train_test_split(remaining_stems, 
                                         test_size=test_size/(val_size + test_size), 
                                         random_state=random_state) 
# Build split mappings
train_map = {f"{s}.jpg": mapping[f"{s}.jpg"] for s in train_stems}
val_map   = {f"{s}.jpg": mapping[f"{s}.jpg"] for s in val_stems}
test_map  = {f"{s}.jpg": mapping[f"{s}.jpg"] for s in test_stems}

"""
Move files to train, val, test folders
"""
# destination path for split dataset
dest_path = Path(extract_path / "split_dataset")        
os.makedirs(dest_path, exist_ok=True)

for split, split_map in [("train", train_map), ("val", val_map), ("test", test_map)]:
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
    dataset_name="Desc_final_dataset"
)
split_data.add_files(str(dest_path))
split_data.upload()
split_data.finalize()
logging.info(f"Uploaded split dataset id={split_data.id}")

# 11) Save outputs
task.set_parameter("output_dataset_project", split_data.project)
task.set_parameter("output_dataset_id", split_data.id)
task.set_parameter("output_dataset_name", split_data.name)