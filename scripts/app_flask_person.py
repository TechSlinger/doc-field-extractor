from flask import Flask, request, jsonify, Response, json
import pandas as pd
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from utils.file_utils import load_and_process_files
from utils.extraction_utils import get_message, get_output_text, get_data_from_output_text

app = Flask(__name__)

# Load model
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "model", torch_dtype="auto", device_map="cpu"
)
processor = AutoProcessor.from_pretrained("model")

@app.route("/extract_fa", methods=["POST"])
def extract_fa():
    files = request.files.getlist("fa_files")
    if not files:
        return jsonify({"error": "No FA files uploaded"}), 400

    images = load_and_process_files(files)
    prompt = """
    From this image, extract the following information:
    - Période de validité
    - CARTE D'IDENTITE NATIONALE NUMERO
    - Prénom
    - Nom
    - the rows and cells from the table body (excluding headers)
    """
    output = get_output_text(get_message(images, prompt), model, processor)
    data = get_data_from_output_text(output)
    json_str = json.dumps({"fa_data": data}, ensure_ascii=False)
    return Response(json_str, content_type='application/json; charset=utf-8')


@app.route("/extract_cin", methods=["POST"])
def extract_cin():
    files = request.files.getlist("cin_files")
    if not files:
        return jsonify({"error": "No CIN files uploaded"}), 400

    images = load_and_process_files(files)
    prompt = """
    From these images, extract the following informations:
    Prénom
    Nom
    date de naissance
    date d'expiration de la carte
    CIN
    Sexe(Homme ou Femme)
    Nationalité
    """
    output = get_output_text(get_message(images, prompt), model, processor)
    data = get_data_from_output_text(output)
    return jsonify({"cin_data": data})


@app.route("/compare", methods=["POST"])
def compare():
    data = request.get_json()
    fa_data = data.get("fa_data")
    cin_data = data.get("cin_data")

    if not fa_data or not cin_data:
        return jsonify({"error": "Missing FA or CIN data"}), 400

    match = (
        cin_data.get("Nom", "").strip().lower() == fa_data.get("Nom", "").strip().lower() or
        cin_data.get("Prénom", "").strip().lower() == fa_data.get("Prénom", "").strip().lower() and
        cin_data.get("CIN", "").strip().upper() == fa_data.get("CARTE D'IDENTITE NATIONALE NUMERO", "").strip().upper()
    )

    if match:
        combined_data = {**cin_data, **fa_data}
        df = pd.DataFrame([combined_data])
        df.to_csv("cin_FA_infos.csv", index=False)
        return jsonify({
            "match": True,
            "message": "✅ Les documents appartiennent à la même personne.",
            "combined_data": combined_data
        })
    else:
        return jsonify({
            "match": False,
            "message": "❌ Les documents ne correspondent pas."
        })


if __name__ == "__main__":
    app.run(debug=True)
