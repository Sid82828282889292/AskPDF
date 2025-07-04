# from langchain.chains import RetrievalQA
# from langchain.llms import HuggingFacePipeline
# from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
# from langchain_community.vectorstores import Chroma

# LLM_PIPELINE = None

# def load_local_llm():
#     global LLM_PIPELINE
#     if LLM_PIPELINE is None:
#         model_id = "tiiuae/falcon-rw-1b"  # CPU-friendly
#         tokenizer = AutoTokenizer.from_pretrained(model_id)
#         model = AutoModelForCausalLM.from_pretrained(model_id)
#         pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=200)
#         LLM_PIPELINE = HuggingFacePipeline(pipeline=pipe)
#     return LLM_PIPELINE

# def query_pdf_with_llm(vectorstore: Chroma, question: str) -> str:
#     retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
#     llm = load_local_llm()
#     qa = RetrievalQA.from_chain_type(
#         llm=llm,
#         retriever=retriever,
#         return_source_documents=True
#     )
#     result = qa.invoke({"query": question})
#     return result["result"]

# def query_pdf_with_chunks_only(vectorstore: Chroma, question: str) -> str:
#     retriever = vectorstore.as_retriever(search_kwargs={"k": 1})  # only top result
#     docs = retriever.get_relevant_documents(question)
#     if docs:
#         return docs[0].page_content.strip()
#     else:
#         return "Sorry, I couldn't find anything relevant in the PDF."


from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_community.vectorstores import Chroma

LLM_PIPELINE = None

def load_local_llm():
    global LLM_PIPELINE
    if LLM_PIPELINE is None:
        model_id = "tiiuae/falcon-rw-1b"  # Small, CPU-friendly
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id)
        pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=200)
        LLM_PIPELINE = HuggingFacePipeline(pipeline=pipe)
    return LLM_PIPELINE

def query_pdf_with_llm(vectorstore: Chroma, question: str) -> str:
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm = load_local_llm()
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )
    result = qa.invoke({"query": question})
    return result["result"]

def query_pdf_with_chunks_only(vectorstore: Chroma, question: str) -> str:
    retriever = vectorstore.as_retriever(search_kwargs={"k": 1})  # Only top match
    docs = retriever.get_relevant_documents(question)
    if docs:
        return docs[0].page_content.strip()
    else:
        return "Sorry, I couldn't find anything relevant in the PDF."

def polish_with_llm(raw_text: str, question: str) -> str:
    llm = load_local_llm()
    prompt = f"Rephrase this content to answer the question '{question}' in a clear, confident, and polished tone for professional review:\n\n{raw_text}"
    return llm.invoke(prompt)
