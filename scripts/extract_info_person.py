import streamlit as st
from PIL import Image
import tempfile
import os
import pandas as pd

# Your model and utility imports
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import fitz
import numpy as np
import re

# Load model and processor
@st.cache_resource
def load_model():
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "../QWEN2.5-VL-3B-INSTRUCT", torch_dtype="auto", device_map="cpu"
    )
    processor = AutoProcessor.from_pretrained("../QWEN2.5-VL-3B-INSTRUCT")
    return model, processor

model, processor = load_model()


def load_and_process_files(uploaded_files):   
    images = []
    for uploaded_file in uploaded_files:
        ext = os.path.splitext(uploaded_file.name)[1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        if ext == ".pdf":
            doc = fitz.open(tmp_path)
            for page in doc:
                pix = page.get_pixmap(dpi=150)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                if img:
                    images.append(img.resize((600, 800)))
        else:
            img = Image.open(tmp_path).convert("RGB")
            if img:
                images.append(img.resize((600, 800)))
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


st.header(" Extraction des informations à partir de la Fiche Anthropométrique et la Carte Nationale")

# Step 1: Upload FA
MAX_FILE_SIZE_MB = 5  # Max allowed size (5MB)
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024

fa_files = st.file_uploader(
    "📎 Téléversez la fiche anthropométrique (PDF ou image — max 5 Mo)",
    type=["pdf", "png", "jpg", "jpeg"],
    accept_multiple_files=True
)


if fa_files:
    fa_images = load_and_process_files(fa_files)
    # Step 1: Classify the FA
    classify_prompt = "Analyze the table in this image. Ignore the header row (which may contain words like 'NATURE DES CRIMES', 'DATE DES CONDAMNATIONS', etc.). Focus only on the table body, below the headers. If all rows contain only empty cells or placeholders, asterisks, respond with 0. else if any row contains real values like dates, words (not one letter), or numbers — respond with 1. Respond only o with 0 or 1"
  
    classification = get_output_text(get_message(fa_images, classify_prompt))[0]
    output = {"situation judiciaire class": classification}
    if classification == "1":
        st.warning("⚠️ Le fiche anthropométrique contient des antécédents judiciaires.")
    else:
        st.success("✅ La fiche anthropométrique ne contient aucun antécédent judiciaire.")
    
    fa_prompt = (
    "From this image, extract the following information:\n"
    "Période de validité\n"
    "CARTE D'IDENTITE NATIONALE NUMERO\n"
    "Prénom\n"
    "Nom")

    fa_output = get_output_text(get_message(fa_images, fa_prompt))
    fa_data = get_data_from_output_text(fa_output)
    st.subheader("📋 Informations extraites de la fiche anthropométrique")
    result_fa = {**fa_data, **output}
    st.json(result_fa)
    st.session_state["fa_data"] = fa_data  # Store FA data in session state
    
# Step 2: Ask for CIN only if FA is clean
cin_files = st.file_uploader("Téléversez la Carte Nationale (CIN) (PDF or images)", 
                                 type=["pdf", "png", "jpg", "jpeg"],
                                 accept_multiple_files=True)

if cin_files:
        cin_images = load_and_process_files(cin_files)
        cin_prompt = "From these images, Extract these informations : Prénom, Nom, date de naissance, date d'expiration de la carte, CIN, Adresse, Sexe(Homme ou Femme), Nationalité"
        cin_output = get_output_text(get_message(cin_images, cin_prompt))
        cin_data = get_data_from_output_text(cin_output)

        st.subheader("📋 Informations extraites de la Carte Nationale (CIN)")
        st.json(cin_data)
        st.session_state["cin_data"] = cin_data  # Store CIN data in session state
        
# Step 3: Compare FA and CIN
if "fa_data" in st.session_state and "cin_data" in st.session_state:
    if st.button("🔁 Vérifier si la fiche anthropométrique et la Carte Nationale correspondent"):
        # Retrieve data from session state
        cin_data = st.session_state["cin_data"]
        fa_data = st.session_state["fa_data"]
        # Step 3: Match and save
        if (
            (cin_data.get("Nom", "").strip().lower() != fa_data.get("Nom", "").strip().lower()
            and cin_data.get("Prénom", "").strip().lower() != fa_data.get("Prénom", "").strip().lower() )
            or cin_data.get("CIN", "").strip().upper() != fa_data.get("CARTE D'IDENTITE NATIONALE NUMERO", "").strip().upper()
        ):
            st.error("❌ la fiche anthropométrique et la carte nationale ne correspondent pas. Veuillez vérifier les informations.")
        else:

            st.success("✅ Les documents appartiennent à la même personne. Voici l’ensemble des informations demandées :")
            combined_data = {**cin_data, **fa_data}
            st.json(combined_data)
            df = pd.DataFrame([combined_data])
            df.to_excel("cin_FA_infos.xlsx", index=False)
            st.download_button("⬇️ Télécharger les données extraites (Excel)", data=df.to_csv(index=False), file_name="cin_FA_infos.csv", mime="text/csv")
