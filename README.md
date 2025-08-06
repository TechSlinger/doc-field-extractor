Model Setup
This project uses the Qwen2.5-VL-7B-Instruct model. You need to download it from Hugging Face.

Clone the model locally

git lfs install
git clone https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct model
Make sure the folder is named model and is placed at the root of the project.

Note: The project uses AutoProcessor and Qwen2_5_VLForConditionalGeneration to load this model. Ensure you have enough resources to run it (or switch to CPU as configured).

Configuration & Dependencies
Recommended Environment
Python 3.10+

CPU or GPU-enabled machine (GPU recommended for faster inference)

Install dependencies:
We recommend using a conda environment or a python virtual environment:

conda create -n qwen-local-ocr python=3.10 
conda activate qwen-local-ocr
pip install -r requirements.txt

How to Run
 Start the Flask Server
python app.py
 This will start the server at http://127.0.0.1:5000/.

API Endpoints
1. Extract FA Info
POST /extract_fa

Form-Data

1. fa_files: Upload image 

Response:

json

{
  "fa_data": {
    "Nom": "...",
    "Prénom": "...",
    ...
  }
}
2. Extract CIN Info
POST /extract_cin

Form-Data

cin_files: Upload one or multiple image files

Response

json

{
  "cin_data": {
    "Nom": "...",
    "Prénom": "...",
    ...
  }
}
3. Compare Documents
POST /compare

json body

{
  "fa_data": { ... },
  "cin_data": { ... }
}
Response

json

{
  "match": true,
  "message": " Les documents appartiennent à la même personne.",
  "combined_data": { ... }
}
If documents don't match:

json

{
  "match": false,
  "message": " Les documents ne correspondent pas."
}
