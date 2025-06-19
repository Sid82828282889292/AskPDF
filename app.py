import streamlit as st
import os
from dotenv import load_dotenv
from io import BytesIO
from utils.pdf_loader import load_pdf_text
from utils.chunk_embed import embed_and_store_chunks
from utils.qa_chain import create_qa_chain
from utils.cache_utils import get_file_hash, load_vectorstore, save_vectorstore

# Load .env
load_dotenv()

# Page config
st.set_page_config(page_title="PDFQueryAI", layout="wide")
st.title("📄 PDF Query AI — Ask Questions From PDFs")

# Session states
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

# Upload PDFs
uploaded_files = st.file_uploader("Upload one or more PDF files", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    all_text = ""
    combined_hash = ""

    for uploaded_file in uploaded_files:
        file_bytes = uploaded_file.read()
        uploaded_file.seek(0)  # reset pointer
        file_hash = get_file_hash(file_bytes)
        combined_hash += file_hash
        all_text += load_pdf_text(uploaded_file) + "\n"

    # Vector DB cache path
    cache_path = os.path.join("cached", combined_hash)

    # Load or create vectorstore
    vectordb = load_vectorstore(combined_hash)
    if vectordb:
        st.success("✅ Loaded cached vector DB")
    else:
        st.info("🔄 Embedding documents…")
        vectordb = embed_and_store_chunks(all_text, persist_path=cache_path)
        save_vectorstore(combined_hash, vectordb)
        st.success("✅ Vector DB created and cached!")

    # Create QA chain
    st.session_state.qa_chain = create_qa_chain(vectordb)

# Query input
if st.session_state.qa_chain:
    query = st.text_input("Ask a question about the documents:")

    if query:
        result = st.session_state.qa_chain.invoke({
            "question": query,
            "chat_history": st.session_state.chat_history
        })

        answer = result["answer"]
        sources = result.get("source_documents", [])

        st.markdown("### 🤖 Answer")
        st.markdown(answer)

        # Update chat history
        st.session_state.chat_history.append(("You", query))
        st.session_state.chat_history.append(("AI", answer))

        # Show source chunks
        if sources:
            st.markdown("## 📚 Source Chunks Used")
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### 📂 Uploaded Files")
                for f in uploaded_files:
                    st.markdown(f"- `{f.name}`")

            with col2:
                st.markdown("### 🧠 Relevant Chunks")
                for i, doc in enumerate(sources):
                    st.markdown(f"**Chunk {i+1}:**")
                    st.code(doc.page_content.strip(), language="text")

# Chat history viewer + download
if st.session_state.chat_history:
    with st.expander("🧠 Chat History"):
        for speaker, msg in st.session_state.chat_history:
            st.markdown(f"**{speaker}:** {msg}")

    def convert_to_txt():
        return "\n".join(f"{s}: {m}" for s, m in st.session_state.chat_history)

    chat_bytes = convert_to_txt().encode("utf-8")
    st.download_button("💾 Download Chat History", chat_bytes, file_name="chat_history.txt", mime="text/plain")
