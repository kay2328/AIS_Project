"""
This script maps images to their corresponding annotation files.
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

import os
import json
import logging
from typing import List, Optional, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")


images_dir = "C:\\Users\\kamat\\OneDrive\\Documents\\GitHub\\AIS\\EnigmaAI\\image_description\\dataset\\data_subset\\images" 
labels_dir = "C:\\Users\\kamat\\OneDrive\\Documents\\GitHub\\AIS\\EnigmaAI\\image_description\\dataset\\data_subset\\labels"
data_prep_file = "C:\\Users\\kamat\\OneDrive\\Documents\\GitHub\\AIS\\EnigmaAI\\image_description\\dataset\\prepared\\img_label.json"

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
    image_extensions = (".jpg", ".jpeg", ".png")

    # Iterate over image files.
    for filename in os.listdir(images_dir):
        if filename.lower().endswith(image_extensions):
            base_name, _ = os.path.splitext(filename)
            label_file_path = os.path.join(labels_dir, base_name + ".txt")
            if os.path.exists(label_file_path):
                annotations = parse_label_file(label_file_path)
                if annotations is not None:
                    mapping[filename] = annotations
                else:
                    logging.warning(f"Annotation parsing failed for {label_file_path}")
            else:
                logging.warning(f"No label file found for image: {filename} in {labels_dir}")

    # Save the mapping as a JSON file.
    try:
        with open(output_file, "w") as f:
            json.dump(mapping, f, indent=4)
        logging.info(f"Image-label mapping saved successfully to {output_file}")
    except Exception as e:
        logging.error(f"Error writing JSON mapping to {output_file}: {e}")

if __name__ == "__main__":
    create_mapping(images_dir, labels_dir, data_prep_file)
