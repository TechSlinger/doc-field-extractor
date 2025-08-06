import re
import torch
from qwen_vl_utils import process_vision_info

def get_message(image_paths, prompt):
    return [{
        "role": "user",
        "content": [
            *[{"type": "image", "image": img} for img in image_paths],
            {"type": "text", "text": prompt}
        ],
    }]

def get_output_text(messages, model, processor):
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, return_tensors="pt")
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=100)
    generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    return processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)

def get_data_from_output_text(output_text):
    text = output_text[0]
    data = {}
    for line in text.split('\n'):
        if ':' in line:
            raw_key, value = line.split(':', 1)
            value = value.strip()
            cleaned_key = re.sub(r'^[^a-zA-ZÀ-ÿ]+', '', raw_key)
            cleaned_key = re.sub(r'[^a-zA-ZÀ-ÿ\s]+$', '', cleaned_key).strip()
            if cleaned_key and value:
                data[cleaned_key] = value
    return data
