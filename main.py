from src.load_and_chunk import load_and_chunk
from src.rag_pipeline import create_vectorstore, answer_question
from src.evaluate import EVAL_QUESTIONS

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings


def ingest_data():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    chunks = load_and_chunk("data/policy.txt")

    vectordb = Chroma.from_documents(
        chunks,
        embedding=embeddings,
        persist_directory="chroma_db"
    )

    vectordb.persist()


def run_evaluation(vectordb):
    for item in EVAL_QUESTIONS:
        print("\nQuestion:", item["question"])
        answer = answer_question(item["question"], vectordb)
        print("Answer:", answer)


def main():
    print("\nPolicy RAG System Ready")
    print("==============================")

    # Run only once if DB already exists
    # ingest_data()

    vectordb = create_vectorstore()
    run_evaluation(vectordb)


if __name__ == "__main__":
    main()
