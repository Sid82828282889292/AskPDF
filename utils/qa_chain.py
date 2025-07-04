from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_community.vectorstores import Chroma

def load_local_llm():
    model_id = "tiiuae/falcon-rw-1b"  # small model (~1.3B) suitable for CPU
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)

    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=200)
    llm = HuggingFacePipeline(pipeline=pipe)
    return llm

def query_pdf(vectorstore: Chroma, question: str) -> str:
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm = load_local_llm()

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )
    result = qa.invoke({"query": question})
    return result["result"]
