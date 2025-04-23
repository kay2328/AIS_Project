"""
To preprocess the VizWiz image captioning dataset for further fine-tuning.
  - Loaded the merged csv dataset (which contains information of "images" and "annotations").
  - Caption cleaning: Converted to lowercase, punctuations removed
  - Special tokens additions will be taken care automatically when using pretrained model tokenizer before finetuning.
  - Image processing is handled in model transforming pipeline
  - The preprocessed caption data is saved under dataset/preprocessed_data folder.
"""

import os
import logging
import re
import pandas as pd
from data_exploration import load_data, load_config


# Load configuration of path variables
config = load_config()
TRAIN_JSON = config["data"]["train_json"]
VALID_JSON = config["data"]["valid_json"]
PREPROCESSED_OUTPUT = config["output"]["preprocessed"]

def setup_logging():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def clean_caption(caption: str) -> str:
    caption = caption.lower()
    caption = re.sub(r"[^\w\s]", "", caption)
    return caption

def preprocess_annotations(data) -> dict:
    """
    Preprocess each annotation by cleaning its caption.
    """
    annotations = data.get("annotations", [])
    for annotation in annotations:
        caption = annotation.get("caption", "")
        cleaned = clean_caption(caption)
        annotation["caption"] = cleaned
    return data

def save_preprocessed_data(data, output_path, type):
    image_df = pd.DataFrame(data["images"])
    annot_df = pd.DataFrame(data["annotations"]) 
    if annot_df.empty or image_df.empty:
        logging.error("One of the datasets is empty. Check your JSON files.")
        raise ValueError("Images or annotations is empty.")
    else:
        logging.info(f"Total image samples in {type} data: {len(image_df)}")
        logging.info(f"Total annotations/captions samples in {type} data: {len(annot_df)}")
    df = pd.merge(annot_df, image_df, left_on="image_id", right_on="id", suffixes=('_annot', '_img'))
    logging.info(f"Merged Images and Annotations DataFrame shape after preprcoessing: {df.shape}")
    df.to_csv(output_path, index=False)
    logging.info(f"Preprocessed data saved to {output_path}")

def main():
    setup_logging()
    os.makedirs(PREPROCESSED_OUTPUT, exist_ok=True)
    for type, json_file in zip(["train" ,"valid"], [TRAIN_JSON, VALID_JSON]):
        data = load_data(json_file)
        logging.info(f"Preprocessing {type} data")
        data = preprocess_annotations(data)
        preprocess_output_file = os.path.join(PREPROCESSED_OUTPUT, f"{type}.csv")
        save_preprocessed_data(data, preprocess_output_file, type)
        logging.info("Preprocessing completed.")

if __name__ == "__main__":
    main()
