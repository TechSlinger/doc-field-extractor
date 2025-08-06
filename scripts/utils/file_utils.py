import tempfile, os
from PIL import Image
import fitz  # PyMuPDF

def load_and_process_files(uploaded_files):
    images = []
    for file in uploaded_files:
        filename = file.filename
        ext = os.path.splitext(filename)[1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp.write(file.read())
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
