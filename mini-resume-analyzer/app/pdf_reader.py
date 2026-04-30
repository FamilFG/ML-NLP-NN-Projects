import fitz
import pytesseract
from pdf2image import convert_from_path

POPPLER_PATH = r"C:\Program Files\poppler\poppler-25.12.0\Library\bin"

def extract_text_from_pdf(pdf_path, min_text_length=50):
    text = ""

    try:
        doc = fitz.open(pdf_path)
        for page in doc:
            text += page.get_text()
    except:
        text = ""

    text = text.strip()

    if len(text) >= min_text_length:
        return text

    try:
        images = convert_from_path(
            pdf_path,
            dpi=200,
            poppler_path=POPPLER_PATH,
            first_page=1,
            last_page=1
        )

        ocr_text = ""
        for img in images:
            ocr_text += pytesseract.image_to_string(img)

        return ocr_text.strip()

    except:
        return text