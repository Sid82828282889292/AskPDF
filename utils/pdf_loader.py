import fitz  # PyMuPDF
from typing import Union
from io import BytesIO

def load_pdf_text(uploaded_file: Union[BytesIO, str]) -> str:
    """Extracts and returns text from a PDF file."""
    if isinstance(uploaded_file, BytesIO):
        doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    else:
        doc = fitz.open(uploaded_file)

    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text
