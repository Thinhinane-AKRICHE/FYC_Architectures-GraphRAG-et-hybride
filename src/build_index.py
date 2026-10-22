from pathlib import Path

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from load_documents import load_pdfs


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHROMA_DIRECTORY = PROJECT_ROOT / "data" / "chroma_db"

COLLECTION_NAME = "projet_alpha"

EMBEDDING_MODEL_NAME = (
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)


def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=100,
        separators=["\n\n", "\n", ". ", " ", ""],
        add_start_index=True,
    )

    chunks = text_splitter.split_documents(documents)

    for chunk_number, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = chunk_number

    return chunks


def create_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def build_index():
    documents = load_pdfs()
    chunks = split_documents(documents)

    print("\nCréation des embeddings...")
    embeddings = create_embeddings()

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIRECTORY),
    )

    chunk_ids = [
        f"chunk-{index:04d}"
        for index in range(len(chunks))
    ]

    vector_store.add_documents(
        documents=chunks,
        ids=chunk_ids,
    )

    print("\nIndexation terminée")
    print(f"Nombre de pages : {len(documents)}")
    print(f"Nombre de chunks indexés : {len(chunks)}")
    print(f"Index enregistré dans : {CHROMA_DIRECTORY}")

    return vector_store


if __name__ == "__main__":
    build_index()