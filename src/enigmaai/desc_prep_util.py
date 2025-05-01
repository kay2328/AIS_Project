import json
import logging
from typing import List, Optional, Dict, Any
from pathlib import Path

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

def find_dir_with_files(root: Path, name: str) -> Path:
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
