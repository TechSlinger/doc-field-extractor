import streamlit as st
from PIL import Image
import tempfile
import os

# Your model and utility imports
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
#from peft import PeftModel
from qwen_vl_utils import process_vision_info
import torch
import fitz
import numpy as np
import re

# Load model and processor
@st.cache_resource
def load_model():
    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "../QWEN2.5-VL-3B-INSTRUCT", torch_dtype="auto", device_map="cpu"
    )
    model = PeftModel.from_pretrained(base_model, "../lora-matricule")
    processor = AutoProcessor.from_pretrained("../QWEN2.5-VL-3B-INSTRUCT")
    return model, processor

model, processor = load_model()


def crop_non_white_area(img, threshold=240, margin=10):
    np_img = np.array(img)
    non_empty_coords = np.argwhere(np_img < threshold)
    if non_empty_coords.size == 0:
        return None
    top_left = non_empty_coords.min(axis=0)
    bottom_right = non_empty_coords.max(axis=0)
    left = max(top_left[1] - margin, 0)
    top = max(top_left[0] - margin, 0)
    right = min(bottom_right[1] + margin, img.width)
    bottom = min(bottom_right[0] + margin, img.height)
    return img.crop((left, top, right, bottom))


def load_and_process_files(uploaded_files):   
    images = []
    for uploaded_file in uploaded_files:
        if uploaded_file.size > MAX_FILE_SIZE_BYTES:
            st.error(f"❌ {uploaded_file.name} exceeds the {MAX_FILE_SIZE_MB}MB limit. Please upload a smaller file.")
            continue
        ext = os.path.splitext(uploaded_file.name)[1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        if ext == ".pdf":
            doc = fitz.open(tmp_path)
            for page in doc:
                pix = page.get_pixmap(dpi=150)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                cropped = img
                if cropped:
                    images.append(cropped.resize((600, 800)))
        else:
            img = Image.open(tmp_path).convert("RGB")
            cropped = img
            if cropped:
                images.append(cropped.resize((600, 800)))
    return images


def get_message(image_paths, prompt):
    return [{
        "role": "user",
        "content": [
            *[{"type": "image", "image": img} for img in image_paths],
            {"type": "text", "text": prompt}
        ],
    }]


def get_output_text(messages):
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, return_tensors="pt")
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=100)
    generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    return processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)

def get_data_from_output_text(output_text):
    text = output_text[0] if isinstance(output_text, list) else output_text
    data = {}
    for line in text.split('\n'):
        if ':' in line:
            raw_key, value = line.split(':', 1)
            value = value.strip()
            cleaned_key = re.sub(r'^[^a-zA-ZÀ-ÿ]+', '', raw_key)
            cleaned_key = re.sub(r'[^a-zA-ZÀ-ÿ\s]+$', '', cleaned_key).strip()
            cleaned_value = re.sub(r'^[^a-zA-ZÀ-ÿ]+', '', value)
            cleaned_value = re.sub(r'[^a-zA-ZÀ-ÿ\s]+$', '', value).strip()
            if cleaned_key and cleaned_value:
                data[cleaned_key] = cleaned_value
    return data

import re

def normalize_matricule(matricule):
    """
    Normalise un matricule pour comparaison :
    - Supprime les espaces
    - Supprime les zéros en trop dans les parties numériques
    - Met la lettre en majuscule
    Exemple : "47549 -B-07" => "47549-B-7"
    """
    if not matricule:
        return None

    # Supprimer espaces et uniformiser le format
    matricule = matricule.strip().replace(" ", "")

    # Pattern pour format : num-lettre-num (lettre peut être arabe ou latine)
    match = re.match(r"(\d+)-([A-Zا-ي])-?(\d+)", matricule)
    if match:
        part1 = str(int(match.group(1)))  # Ex: "007" => "7"
        letter = match.group(2).upper()   # "b" => "B"
        part3 = str(int(match.group(3)))  # Ex: "07" => "7"
        return f"{part1}-{letter}-{part3}"
    
    # Si le format ne matche pas, retourner brut en majuscule
    return matricule.upper()


st.title("🚗 Vehicle Document Processing App")
MAX_FILE_SIZE_MB = 5  # Max allowed size (5MB)
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024


# Initialiser les variables globalement
assur_data = None
grise_data = None
VT_data = None

# Step 1: Upload Assurance
assur_files = st.file_uploader(
    "📎 Upload Assurance document (PDF or image — max 5MB)",
    type=["pdf", "png", "jpg", "jpeg"],
    accept_multiple_files=True
)

if assur_files:
    assur_images = load_and_process_files(assur_files)
    assur_prompt = ("From this image, Extract these informations: Numéro d'immatriculation (matricule) , Période de garantie, Nom, Marque et type\n"
    "Important: The 'matricule' follows this format: a sequence of digits, followed by a single uppercase letter, enclosed between two hyphens (e.g., 123-A-456). You must use OCR to accurately extract all digits and the letter, even if some characters are thin, faint, or partially faded."
    )
    assur_output = get_output_text(get_message(assur_images, assur_prompt))
    st.write(assur_output)
    assur_data = get_data_from_output_text(assur_output)
    st.subheader("📋 Assurance Extracted Data")
    st.json(assur_data)


# Step 2: Upload Carte grise
grise_files = st.file_uploader(
    "📎 Upload Carte Grise document (PDF or image — max 5MB)",
    type=["pdf", "png", "jpg", "jpeg"],
    accept_multiple_files=True,
)

#(matricule antérieur etape2)
if grise_files:
    grise_images = load_and_process_files(grise_files)
    grise_prompt = ("Extract these informations from this image:\n\n"
    "Numéro d'immatriculation (matricule), propriétaire, Fin de validité, P.T.A.C"
    "Important: The 'matricule' contains digits, followed by an Arabic letter between two hyphens, then followed by one or more digits.\n"
    "The Arabic letter is part of the Arabic alphabet and not a number. It must be one of the following:\n"
    "أ, ب, ت, ث, ج, ح, خ, د, ذ, ر, ز, س, ش, ص, ض, ط, ظ, ع, غ, ف, ق, ك, ل, م, ن, ه, و, ي. \n")
    grise_output = get_output_text(get_message( grise_images, grise_prompt))
    st.write(grise_output)
    grise_data = get_data_from_output_text(grise_output)
    st.subheader("📋 Carte grise Extracted Data")
    st.json(grise_data)
    # Vérification du type (véhicule ou engin) selon le PTAC
    ptac_raw = grise_data.get("P.T.A.C")  # Attention au nom exact de la clé

    if ptac_raw:
        # Nettoyage pour extraire juste le nombre (ex: "2500 kg" → 2500)
        ptac_match = re.search(r"\d+", ptac_raw.replace(" ", ""))
        if ptac_match:
            ptac = int(ptac_match.group())
            if ptac < 3000:
                st.success(f"🚗 Ce document correspond à un **véhicule** (PTAC = {ptac} kg < 3000 kg).")
            else:
                st.info(f"🚜 Ce document correspond à un **engin** (PTAC = {ptac} kg ≥ 3000 kg).")
        

# Step 3: Upload V.T
VT_files = st.file_uploader(
    "📎 Upload Visite technique document (PDF or image — max 5MB)",
    type=["pdf", "png", "jpg", "jpeg"],
    accept_multiple_files=True,
)

#(vt doit etre moins de 10j de sa date d'expiration)
if VT_files:
    VT_images = load_and_process_files(VT_files)
    VT_prompt = ("Extract these informations from this image:\n\n"
    "DATE DU CONTROLE, vdate de validité , Numéro d'immatriculation (matricule), propriétaire"
        "Important: The 'matricule' contains digits, followed by an Arabic letter, then followed by one or more digits.\n"
    "The Arabic letter is part of the Arabic alphabet and not a number. It must be one of the following:\n"
    "ا, ب, ت, ث, ج, ح, خ, د, ذ, ر, ز, س, ش, ص, ض, ط, ظ, ع, غ, ف, ق, ك, ل, م, ن, ه, و, ي. and in the output instead of this arabic letter return the corresponding english letter. here is an example of a matricule: 47549-B-5 \n"
)
    VT_output = get_output_text(get_message(VT_images, VT_prompt))
    st.write(VT_output)
    VT_data = get_data_from_output_text(VT_output)
    st.write(VT_data)
    st.subheader("📋 Visite technique Extracted Data")
    st.json(VT_data)

'''if assur_data and grise_data and VT_data:
    matricule_assur = normalize_matricule(assur_data.get("matricule"))
    matricule_grise = normalize_matricule(grise_data.get("matricule"))
    matricule_vt = normalize_matricule(VT_data.get("matricule"))

    if matricule_assur and matricule_grise and matricule_vt:
        if (
            matricule_assur == matricule_grise == matricule_vt
        ):
            st.success(f"✅ Les trois documents appartiennent au même véhicule : {matricule_assur}")
        else:
            st.warning("⚠️ Les matricules ne correspondent pas entre les documents :")
            st.markdown(f"- Assurance : `{matricule_assur}`")
            st.markdown(f"- Carte grise : `{matricule_grise}`")
            st.markdown(f"- Visite technique : `{matricule_vt}`")
'''




vignette_files = st.file_uploader(
    "📎 Upload Carte Grise document (PDF or image — max 5MB)",
    type=["pdf", "png", "jpg", "jpeg"],
    accept_multiple_files=True,
)

#(matricule antérieur etape2)
if vignette_files:
    vignette_images = load_and_process_files(vignette_files)
    vignette_prompt = ("Extract these informations from this image:\n\n"
    "Numéro d'immatriculation (matricule), année de validité, date de paiement, date d'édition"
    "Important: The 'matricule' contains digits, followed by an Arabic letter between two hyphens, then followed by one or more digits.\n"
    "The Arabic letter is part of the Arabic alphabet and not a number. It must be one of the following:\n"
    "أ, ب, ت, ث, ج, ح, خ, د, ذ, ر, ز, س, ش, ص, ض, ط, ظ, ع, غ, ف, ق, ك, ل, م, ن, ه, و, ي. \n")
    vignette_output = get_output_text(get_message(vignette_images, vignette_prompt))
    st.write(vignette_output)
    vignette_data = get_data_from_output_text(vignette_output)
    st.subheader("📋 Vignette Extracted Data")
    st.json(vignette_data)


#vignette : si matriculé sinon on le controle pas 
#année de validité, matricule, crop scan , date de paiement, date d'édition 
#vignette.ma : site de controle de vignette : année, matricule, date de paiement, édition, 


