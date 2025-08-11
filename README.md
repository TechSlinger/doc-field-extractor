# 🤖 Model Setup & API Documentation

## 🔧 Model Setup

This project uses the **Qwen2.5-VL-7B-Instruct** model. You need to download it from Hugging Face.

### Download Model

Clone the model locally:

```bash
git lfs install
git clone https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct model
```

**Important**: Make sure the folder is named `model` and is placed at the root of the project.

> **Note**: The project uses `AutoProcessor` and `Qwen2_5_VLForConditionalGeneration` to load this model. Ensure you have enough resources to run it (or switch to CPU as configured).

## ⚙️ Configuration & Dependencies

### Recommended Environment

- **Python**: 3.10+
- **Hardware**: CPU or GPU-enabled machine (GPU recommended for faster inference)

### Install Dependencies

We recommend using a conda environment or a python virtual environment:

```bash
conda create -n qwen-local-ocr python=3.10
conda activate qwen-local-ocr
pip install -r requirements.txt
```

## 🚀 How to Run

### Start the Flask Server

```bash
python app.py
```

This will start the server at `http://127.0.0.1:5000/`.

## 🔌 API Endpoints

### 1. Extract "Fiche Anthropométrique" Info

**Endpoint**: `POST /extract_fa`

**Form-Data**:
- `fa_files`: Upload image

**Response**:
```json
{
  "fa_data": {
    "Nom": "...",
    "Prénom": "...",
    "..."
  }
}
```

### 2. Extract CIN Info

**Endpoint**: `POST /extract_cin`

**Form-Data**:
- `cin_files`: Upload one or multiple image files

**Response**:
```json
{
  "cin_data": {
    "Nom": "...",
    "Prénom": "...",
    "..."
  }
}
```

### 3. Compare Documents

**Endpoint**: `POST /compare`

**Request Body**:
```json
{
  "fa_data": { "..." },
  "cin_data": { "..." }
}
```

**Response** (Match):
```json
{
  "match": true,
  "message": "Les documents appartiennent à la même personne.",
  "combined_data": { "..." }
}
```

**Response** (No Match):
```json
{
  "match": false,
  "message": "Les documents ne correspondent pas."
}
```

## 📋 API Usage Examples

### Using cURL

```bash
# Extract FA Info
curl -X POST -F "fa_files=@path/to/fa_image.jpg" \
  http://127.0.0.1:5000/extract_fa

# Extract CIN Info
curl -X POST -F "cin_files=@path/to/cin_image.jpg" \
  http://127.0.0.1:5000/extract_cin

# Compare Documents
curl -X POST -H "Content-Type: application/json" \
  -d '{"fa_data": {...}, "cin_data": {...}}' \
  http://127.0.0.1:5000/compare
```

### Using Python

```python
import requests
import json

# Extract FA Info
with open('fa_image.jpg', 'rb') as f:
    response = requests.post(
        'http://127.0.0.1:5000/extract_fa',
        files={'fa_files': f}
    )
    fa_data = response.json()

# Extract CIN Info
with open('cin_image.jpg', 'rb') as f:
    response = requests.post(
        'http://127.0.0.1:5000/extract_cin',
        files={'cin_files': f}
    )
    cin_data = response.json()

# Compare Documents
compare_data = {
    "fa_data": fa_data["fa_data"],
    "cin_data": cin_data["cin_data"]
}

response = requests.post(
    'http://127.0.0.1:5000/compare',
    headers={'Content-Type': 'application/json'},
    data=json.dumps(compare_data)
)
result = response.json()
print(f"Match: {result['match']}")
print(f"Message: {result['message']}")
```

## 🔍 Response Status Codes

| Status Code | Description |
|-------------|-------------|
| `200` | Success |
| `400` | Bad Request - Invalid input |
| `404` | Endpoint not found |
| `500` | Internal Server Error |

## 📝 Notes

- **File Formats**: Supported image formats include JPG, PNG, JPEG
- **File Size**: Maximum file size limit is typically 10MB per image
- **Processing Time**: GPU inference is significantly faster than CPU
- **Memory Requirements**: Ensure sufficient RAM (8GB+ recommended) for model loading
