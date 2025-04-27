from clearml import Task, Dataset, Model
import json
import logging, zipfile
from typing import List, Optional, Dict, Any
from pathlib import Path
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
THe dataset has the following structure:
data.yaml
"""

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
project_name="Description"
task = Task.init(project_name=project_name, 
                task_name="step1_desc_data_preparation",
                task_type=Task.TaskTypes.data_processing)

params = {
    'dataset_id': 'd8316762cb3844569f4c1fbe643ed7f4',     #'2231b5b121924ed684d6560cf6839619', specific version of the dataset
    'dataset_name': 'base_dataset_zip',               # latest registered dataset
    "dataset_project": "Detection",
}

# logger = task.get_logger()
task.connect(params)
task.execute_remotely(queue_name="desc_preparation")

dataset_id = params['dataset_id']
dataset_name = params['dataset_name']

# validate task input params
if not dataset_id and not dataset_name:
    task.mark_completed(status_message="No dataset provided. Nothing to train on.")
    exit(0)
if dataset_id: 
    # download the latest registered dataset
    server_dataset = Dataset.get(dataset_id=dataset_id, only_completed=True, alias="base_dataset_full")
elif dataset_name: 
    # download the latest registered dataset
    server_dataset = Dataset.get(dataset_name=dataset_name, dataset_project="Detection", only_completed=True, alias="base_dataset_full")

raw_path = Path(server_dataset.get_local_copy())          
print(f"Downloaded dataset name: {server_dataset.name} id: ({server_dataset.id}) to: {raw_path}")

"""
Prepare dataset.
"""
if raw_path.is_dir():
    inner_zips = list(raw_path.glob("*.zip"))
    if inner_zips:
        zip_path = inner_zips[0]
        logging.info(f"Found inner zip: {zip_path.name}, will extract that")
        raw_path = zip_path
# ─── UNZIP ALL CONTENTS ──────────────────────────────────────────────────────────
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

images_dir = find_dir_with_most_files(extract_path, "images")
labels_dir = find_dir_with_most_files(extract_path, "labels")
logging.info(f"Using images_dir = {images_dir} ({len(list(images_dir.iterdir()))} files)")
logging.info(f"Using labels_dir = {labels_dir} ({len(list(labels_dir.iterdir()))} files)")

# build a Path to the JSON file under a subfolder "Desc_Dataset"
out_dir  = extract_path / project_name / "Desc_Dataset"
out_file = out_dir / "desc_prep_dataset.json"
# ensure the output directory exists
out_dir.mkdir(parents=True, exist_ok=True)

# if an old JSON exists, delete it
if out_file.exists():
    logging.info(f"Removing old mapping at {out_file}")
    out_file.unlink()


def parse_label_file(label_file_path: str) -> Optional[List[Dict[str, Any]]]:
    """
    Reads a label file and returns a list of annotation dictionaries.
    Each annotation dictionary has the following keys:
      - "class_label": a list containing the integer value (or values) for the class label.
      - "additional_values": a list with additional numeric values (e.g., bounding box coordinates).
    If the label file is empty (indicating no objects detected), a default annotation is returned:
      [{"class_label": [5], "additional_values": []}]
    Parameters:
        label_file_path (str): Path to the annotation text file.   
    Returns:
        List[dict]: List of annotation dictionaries.
    """
    annotations = []
    try:
        with open(label_file_path, "r") as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()
            if not line:
                continue  # Skip empty lines.
            parts = line.split()
            try:
                # Wrap the class label in a list.
                label = int(parts[0])
                additional_values = [float(val) for val in parts[1:]] if len(parts) > 1 else []
                annotations.append({"class_label": [label], "additional_values": additional_values})
            except ValueError as ve:
                logging.error(f"Error parsing line in {label_file_path}: {ve}")
                continue

        # If no valid annotations were found, return the default annotation.
        if not annotations:
            logging.info(f"Label file {label_file_path} is empty. Assigning default annotation.")
            return [{"class_label": [5], "additional_values": []}]
        return annotations

    except Exception as e:
        logging.error(f"Error reading label file {label_file_path}: {e}")
        # In case of an error reading the file, also return the default annotation.
        return [{"class_label": [5], "additional_values": []}]


def create_mapping(images_dir: str, labels_dir: str, output_file: str) -> None:
    """
    Creates and saves a mapping from image filenames to their corresponding annotations.
    For each image file in the images directory, a corresponding label file (with the same base name
    and a .txt extension) is read from the labels directory. The annotations are parsed and stored
    in a JSON mapping.
    Parameters:
        images_dir (str): Directory containing the image files.
        labels_dir (str): Directory containing the label .txt files.
        output_file (str): Path to the output JSON file.
    """
    mapping = {}
    # Iterate over image files.
    for img in images_dir.iterdir():
        if img.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            lbl = labels_dir / (img.stem + ".txt")
            if lbl:
                anns = parse_label_file(str(lbl))
                if anns is not None:
                    mapping[img.name] = anns
                else:
                    logging.warning(f"Annotation parsing failed for {lbl}")
            else:
                logging.warning(f"No label file found for image: {img.name} in {labels_dir}")
    # Save the mapping as a JSON file.
    try:
        with output_file.open("w") as f:
            json.dump(mapping, f, indent=4)
        logging.info(f"Image-label mapping saved successfully to {output_file}")
    except Exception as e:
        logging.error(f"Error writing JSON mapping to {output_file}: {e}") 

create_mapping(images_dir, labels_dir, out_file)
# upload prepared dataset to ClearML server
dataset = Dataset.create(
    dataset_project=project_name, dataset_name="Desc_Dataset"
)
dataset.add_files(path=str(out_file))
print('Uploading dataset in the background')
dataset.upload()
dataset.finalize()

task.set_parameter("output_dataset_project", dataset.project)
task.set_parameter("output_dataset_id", dataset.id)
task.set_parameter("output_dataset_name", dataset.name)
