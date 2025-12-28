from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama

from src.prompts import RAG_PROMPT


def create_vectorstore(persist_directory="chroma_db"):
    """
    Loads the existing Chroma vector store with embeddings
    """
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectordb = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )

    return vectordb


def answer_question(question, vectordb):
    """
    Retrieves relevant documents and generates answer using Ollama
    """
    retriever = vectordb.as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke(question)

    context = "\n\n".join([doc.page_content for doc in docs])

    llm = Ollama(
        model="llama3.2:1b",
        temperature=0
    )

    prompt = RAG_PROMPT.format(
        context=context,
        question=question
    )

    response = llm.invoke(prompt)
    return response
