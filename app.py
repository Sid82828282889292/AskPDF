# import streamlit as st
# import os
# from utils import auth, pdf_loader, chunk_embed, cache_utils, qa_chain
# from langchain_community.vectorstores import Chroma

# st.set_page_config(page_title="📄 Smart PDF Query", layout="wide")

# if not auth.check_auth():
#     st.stop()

# st.title("📄 Smart PDF Query App")

# # Session for PDF list
# if "pdfs" not in st.session_state:
#     st.session_state.pdfs = {}

# uploaded_file = st.file_uploader("📤 Upload a PDF", type="pdf")

# if uploaded_file:
#     file_bytes = uploaded_file.read()
#     file_hash = cache_utils.get_file_hash(file_bytes)
#     persist_path = os.path.join("cached", file_hash)

#     if file_hash not in st.session_state.pdfs:
#         save_path = os.path.join("cached", f"{file_hash}.pdf")
#         with open(save_path, "wb") as f:
#             f.write(file_bytes)

#         text = pdf_loader.extract_text_from_pdf(save_path)
#         chunk_embed.embed_and_store_chunks(text, persist_path)

#         st.session_state.pdfs[file_hash] = {
#             "name": uploaded_file.name,
#             "path": save_path
#         }

#     st.success(f"✅ Uploaded and indexed: {uploaded_file.name}")

# # PDF selector
# pdf_options = list(st.session_state.pdfs.items())
# if pdf_options:
#     selected_key, selected_pdf = st.selectbox(
#         "📂 Select a PDF to query",
#         options=pdf_options,
#         format_func=lambda x: x[1]["name"]
#     )

#     vectorstore = cache_utils.load_vectorstore(selected_key)
#     question = st.text_input("🔍 Ask a question about this PDF")

#     # Retrieval Mode
#     mode = st.radio("🧠 Choose Answer Mode:", ["Raw Chunk", "Polished Chunk", "AI Summarized"])

#     if question:
#         if mode == "Raw Chunk":
#             answer = qa_chain.query_pdf_with_chunks_only(vectorstore, question)
#         elif mode == "Polished Chunk":
#             raw = qa_chain.query_pdf_with_chunks_only(vectorstore, question)
#             answer = qa_chain.polish_with_llm(raw, question)
#         elif mode == "AI Summarized":
#             answer = qa_chain.query_pdf_with_llm(vectorstore, question)
#         else:
#             answer = "Invalid mode selected."

#         st.markdown("### ✅ Answer")
#         st.write(answer)

#         # PDF highlight
#         highlight = answer.split("\n")[0][:50].strip().replace(" ", "+")
#         viewer_url = f"static/pdfjs/viewer.html?file=../../../cached/{selected_key}.pdf&highlight={highlight}"
#         st.markdown(f"[📄 View PDF with Highlight]({viewer_url})", unsafe_allow_html=True)


import streamlit as st
import os
from utils import auth, pdf_loader, chunk_embed, cache_utils, qa_chain
from langchain_community.vectorstores import Chroma

st.set_page_config(page_title="📄 Smart PDF Query App", layout="wide")

if not auth.check_auth():
    st.stop()

st.title("📄 Smart PDF Query App (Local Mistral via Ollama)")

# Session state for multiple PDFs
if "pdfs" not in st.session_state:
    st.session_state.pdfs = {}

uploaded_file = st.file_uploader("📤 Upload a PDF", type="pdf")

if uploaded_file:
    file_bytes = uploaded_file.read()
    file_hash = cache_utils.get_file_hash(file_bytes)
    persist_path = os.path.join("cached", file_hash)

    if file_hash not in st.session_state.pdfs:
        save_path = os.path.join("cached", f"{file_hash}.pdf")
        with open(save_path, "wb") as f:
            f.write(file_bytes)

        text = pdf_loader.extract_text_from_pdf(save_path)
        chunk_embed.embed_and_store_chunks(text, persist_path)

        st.session_state.pdfs[file_hash] = {
            "name": uploaded_file.name,
            "path": save_path
        }

    st.success(f"✅ Uploaded and indexed: {uploaded_file.name}")

# PDF Selection
pdf_options = list(st.session_state.pdfs.items())
if pdf_options:
    selected_key, selected_pdf = st.selectbox(
        "📂 Select a PDF to query",
        options=pdf_options,
        format_func=lambda x: x[1]["name"]
    )

    vectorstore = cache_utils.load_vectorstore(selected_key)
    question = st.text_input("🔍 Ask a question about this PDF")

    mode = st.radio("🧠 Choose Answer Mode:", ["Raw Chunk", "Polished Chunk", "AI Summarized"])

    if question:
        if mode == "Raw Chunk":
            answer = qa_chain.query_pdf_with_chunks_only(vectorstore, question)
        elif mode == "Polished Chunk":
            raw = qa_chain.query_pdf_with_chunks_only(vectorstore, question)
            answer = qa_chain.polish_with_llm(raw, question)
        elif mode == "AI Summarized":
            answer = qa_chain.query_pdf_with_llm(vectorstore, question)
        else:
            answer = "Invalid mode."

        st.markdown("### ✅ Answer")
        st.write(answer)

        # Create PDF viewer link with highlight
        highlight = answer.split("\n")[0][:50].strip().replace(" ", "+")
        viewer_url = f"static/pdfjs/viewer.html?file=../../../cached/{selected_key}.pdf&highlight={highlight}"
        st.markdown(f"[📄 View PDF with Highlight]({viewer_url})", unsafe_allow_html=True)
