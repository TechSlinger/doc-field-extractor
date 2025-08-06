from flask import Flask, request, jsonify, Response, json
import pandas as pd
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from utils.file_utils import load_and_process_files
from utils.extraction_utils import get_message, get_output_text, get_data_from_output_text

app = Flask(__name__)

# Load model
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "model", torch_dtype="auto", device_map="cpu"
)
processor = AutoProcessor.from_pretrained("model")

@app.route("/extract_assurance", methods=["POST"])
def extract_assurance():
    files = request.files.getlist("assurance_files")
    if not files:
        return jsonify({"error": "No assurance files uploaded"}), 400

    images = load_and_process_files(files)
    prompt = """
    From these images, extract the following informations:
    Numéro d'immatriculation (matricule)
    Période de garantie
    Nom
    Marque et type
    Important: The 'matricule' follows this format: a sequence of digits, followed by a single uppercase letter, enclosed between two hyphens (e.g., 123-A-456).
    """
    output = get_output_text(get_message(images, prompt), model, processor)
    data = get_data_from_output_text(output)
    json_str = json.dumps({"assurance_data": data}, ensure_ascii=False)
    return Response(json_str, content_type='application/json; charset=utf-8')

@app.route("/extract_carte_grise", methods=["POST"])
def extract_carte_grise():
    files = request.getlist("carte_grise_files")
    if not files:
        return jsonify({"error": "No carte grise files uploaded"}), 400
    images = load_and_process_files(files)
    prompt = """Extract these informations from this image:
    Numéro d'immatriculation (matricule)
    Propriétaire
    Fin de validité
    P.T.A.C
    Important: The 'matricule' contains digits, followed by an Arabic letter between two hyphens, then followed by one or more digits.
    The Arabic letter is part of the Arabic alphabet and not a number. It must be one of the following:
    أ, ب, ت, ث, ج, ح, خ, د, ذ, ر, ز, س, ش, ص, ض, ط, ظ, ع, غ, ف, ق, ك, ل, م, ن, ه, و, ي
    """
    output = get_output_text(get_message(images, prompt), model, processor)
    data = get_data_from_output_text(output)
    json_str = json.dumps({"carte_grise_data": data}, ensure_ascii=False)
    return Response(json_str, content_type='application/json; charset=utf-8')

@app.route("/extract_vignette", methods=["POST"])
def extract_vignette():
    files = request.files.getlist("vignette_files")
    if not files:
        return jsonify({"error": "No vignette files uploaded"}), 400

    images = load_and_process_files(files)
    prompt = """
    From these images, extract the following information:
    - DATE DU CONTROLE
    - DATE DE VALIDITE
    - Numéro d'immatriculation
    - Propriétaire
    Important: The 'matricule' follows this format: a sequence of digits, followed by a single uppercase letter, enclosed between two hyphens (e.g., 123-A-456).
    The Arabic letter is part of the Arabic alphabet and not a number. It must be one of the following:
     أ, ب, ت, ث, ج, ح, خ, د, ذ, ر, ز, س, ش, ص, ض, ط, ظ, ع, غ, ف, ق, ك, ل, م, ن, ه, و, ي
    """
    output = get_output_text(get_message(images, prompt), model, processor)
    data = get_data_from_output_text(output)
    json_str = json.dumps({"vignette_data": data}, ensure_ascii=False)
    return Response(json_str, content_type='application/json; charset=utf-8')

if __name__ == "__main__":
    app.run(debug=True)