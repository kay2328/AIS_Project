from PIL import Image
import torch
import os
import json
from transformers import AutoProcessor,LlavaForConditionalGeneration, AutoTokenizer
import logging
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def load_model_and_processor(model_name, device):
    logging.info(f"Loading LLAVA teacher '{model_name}' on {device}")
    processor = AutoProcessor.from_pretrained(model_name)
    model = LlavaForConditionalGeneration.from_pretrained(model_name).to(device)
    model.eval()
    return processor, model

def build_prompt(class_map, annotations):
    # collect all non‑"no objects" labels
    labels = []
    for ann in annotations:
        for c in ann.get("class_label", []):
            name = class_map.get(c, "")
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


def desc_generation(model_name, device, class_map, annot_file, images_dir, out_file):
    proc, mdl = load_model_and_processor(model_name, device)
    # load annotations mapping
    with open(annot_file, "r") as f:
        ann_map = json.load(f)
    pseudo = {}
    for img_name, anns in ann_map.items():
        path = os.path.join(images_dir, img_name)
        if not os.path.exists(path):
            logging.warning(f"Image missing: {path}, skipping")
            continue

        prompt = build_prompt(class_map, anns)
        logging.info(f"Prompt for '{img_name}': {prompt}")
        cap = generate_caption_for_image(path, prompt, proc, mdl, device)
        if cap:
            pseudo[img_name] = cap
            logging.info(f"Caption: {cap}")
        else:
            logging.warning(f"No caption for '{img_name}'")

    # write out pseudo captions
    with open(out_file, "w") as out:
        json.dump(pseudo, out, indent=2)
    logging.info(f"Wrote pseudo captions to {out_file}")