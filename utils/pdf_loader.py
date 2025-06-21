import fitz  # PyMuPDF
from typing import Union, List
from io import BytesIO
from langchain.docstore.document import Document

def load_pdf_documents(uploaded_file: Union[BytesIO, str]) -> List[Document]:
    """Extracts text from PDF and returns a list of Document chunks with page metadata."""
    if isinstance(uploaded_file, BytesIO):
        doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    else:
        doc = fitz.open(uploaded_file)

    documents = []
    for page_num, page in enumerate(doc, start=1):
        text = page.get_text()
        if text.strip():
            documents.append(Document(page_content=text, metadata={"page": page_num}))
    doc.close()
    return documents
