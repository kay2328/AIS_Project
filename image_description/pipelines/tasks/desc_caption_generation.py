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
from PIL import Image
import torch
from transformers import AutoProcessor,LlavaForConditionalGeneration, AutoTokenizer
from clearml import Task, Dataset, Model
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
project_name="Description"
MODEL_NAME = "llava-hf/llava-1.5-7b-hf"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

#get the image dataset from "Detection project- base_dataset"
images_data = Dataset.get(
    dataset_id="2231b5b121924ed684d6560cf6839619",
    only_completed=True,
    alias="base_images"  
)
images_root = Path(images_data.get_local_copy())
images_dir  = images_root / "images"
logging.info(f"Images downloaded to: {images_dir}")

# Initiate the task 2 to generate mapping of image name and reference description for student model to learn later in the pipeline
task = Task.init(project_name=project_name, 
                task_name="step2_desc_caption_generation",
                task_type=Task.TaskTypes.data_processing)
params = {
    'dataset_id': '',                # specific version of the dataset
    'dataset_name': 'Desc_Dataset'               # latest registered dataset
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
if dataset_name: 
    # download the latest registered dataset
    server_dataset = Dataset.get(dataset_name=dataset_name, dataset_project=project_name, only_completed=True, alias="desc_prep_mapping")

extract_path = server_dataset.get_local_copy()          
print(f"Downloaded dataset name: {server_dataset.name} id: ({server_dataset.id}) to: {extract_path}")
extract_path = Path(extract_path)
annot_file = extract_path / "desc_prep_dataset.json"

if not annot_file.exists():
    # print out what _is_ in that folder so you can see where your JSON landed
    logging.error(f"Expected JSON not found! Contents are:\n{list(extract_path.iterdir())}")
    raise FileNotFoundError(f"{annot_file} does not exist")
with annot_file.open("r") as f:
    mapping = json.load(f)
logging.info(f"Loaded {len(mapping)} image→annotation entries")

# build a Path to the JSON file under a subfolder "Desc_Caption_Dataset"
out_dir = extract_path / project_name / "Desc_Caption_Dataset"
out_file = out_dir / "desc_caption_dataset.json"
# ensure the output directory exists
out_dir.mkdir(parents=True, exist_ok=True)
# if an old JSON exists, delete it
if out_file.exists():
    logging.info(f"Removing old mapping at {out_file}")
    out_file.unlink()

# Define the class mapping.
CLASS_MAPPING = {
    0: "hole",
    1: "pole",
    2: "stairs",
    3: "bottle/glass",
    4: "rock",
    5: "no objects"
}


def load_model_and_processor():
    logging.info(f"Loading LLAVA teacher '{MODEL_NAME}' on {DEVICE}")
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    model = LlavaForConditionalGeneration.from_pretrained(MODEL_NAME).to(DEVICE)
    model.eval()
    return processor, model


def build_prompt(annotations):
    # collect all non‑"no objects" labels
    labels = []
    for ann in annotations:
        for c in ann.get("class_label", []):
            name = CLASS_MAPPING.get(c, "")
            if name and name != "no objects":
                labels.append(name)
    if labels:
        objs = ", ".join(set(labels))
        return (
            f"Detected objects: {objs}. "
            "Generate a crisp, complete paragraph describing the scene in detail for a visually impaired person. "
            "Include the count, shape, approximate distance, and relative position of the given detected objects. End the description with a complete sentence."
            "DO NOT HALLUCINATE."

        )
    else:
        return (
            "Generate a crisp, complete paragraph describing the scene in detail for a visually impaired person. "
            "Include count, shape, approximate distance and relative position of any objects in the scene. End the description with a complete sentence."
            "DO NOT HALLUCINATE."
        )

def generate_caption_for_image(image_path: str, prompt: str, processor, model, device: str) -> str:
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        logging.error(f"Error opening image '{image_path}': {e}")
        return ""
    # Prepare vision-language conversation format
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt}
            ],
        }
    ]
    # Apply chat template to embed vision tokens into the prompt
    text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    # Tokenize inputs
    inputs = processor(
        text=[text_prompt],
        images=[image],
        padding=True,
        return_tensors="pt"
    ).to(device)
    # Generate caption
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=64,                        
            temperature=0.7,                           # ↑ Slight creativity
            top_p=0.9,                                 # ↑ Nucleus sampling
            repetition_penalty=1.1,                    # ↓ Less repetition
            do_sample=True,                            # ↑ Sampling over greedy
            eos_token_id=processor.tokenizer.eos_token_id,
            pad_token_id=processor.tokenizer.pad_token_id
        )
    # remove the prompt tokens from the front
    generated_ids = output_ids[:, inputs.input_ids.shape[-1]:]
    # Decode full output (no slicing needed)
    captions = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True
    )
    
    # Log token length for debugging
    logging.debug(f"Caption length: {len(captions[0].split())} tokens")
    return captions[0].strip()


def desc_generation():
    proc, mdl = load_model_and_processor()
    # load annotations mapping
    with open(annot_file, "r") as f:
        ann_map = json.load(f)

    pseudo = {}
    for img_name, anns in ann_map.items():
        path = os.path.join(images_dir, img_name)
        if not os.path.exists(path):
            logging.warning(f"Image missing: {path}, skipping")
            continue

        prompt = build_prompt(anns)
        logging.info(f"Prompt for '{img_name}': {prompt}")
        cap = generate_caption_for_image(path, prompt, proc, mdl, DEVICE)
        if cap:
            pseudo[img_name] = cap
            logging.info(f"Caption: {cap}")
        else:
            logging.warning(f"No caption for '{img_name}'")

    # write out pseudo captions
    with open(out_file, "w") as out:
        json.dump(pseudo, out, indent=2)
    logging.info(f"Wrote pseudo captions to {out_file}")

desc_generation()
# upload prepared dataset to ClearML server
dataset = Dataset.create(
    dataset_project=project_name, dataset_name="Desc_Caption_Dataset"
)
dataset.add_files(path=str(out_file))
print('Uploading dataset in the background')
dataset.upload()
dataset.finalize()

task.set_parameter("output_dataset_project", dataset.project)
task.set_parameter("output_dataset_id", dataset.id)
task.set_parameter("output_dataset_name", dataset.name)