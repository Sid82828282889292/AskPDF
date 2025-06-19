from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from utils.custom_hf_llm import HuggingFaceLLM

def create_qa_chain(vectorstore):
    llm = HuggingFaceLLM()

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )

    return qa_chain
