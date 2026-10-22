import argparse

from langchain_chroma import Chroma

from build_index import (
    CHROMA_DIRECTORY,
    COLLECTION_NAME,
    create_embeddings,
)


DEFAULT_QUESTION = (
    "Pourquoi le projet Alpha est-il en retard, "
    "quelles équipes sont directement touchées "
    "et quelle est la nouvelle date de lancement prévue ?"
)


def load_vector_store():
    if not CHROMA_DIRECTORY.exists():
        raise FileNotFoundError(
            "L'index Chroma n'existe pas. "
            "Lancez d'abord : python src/build_index.py"
        )

    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=create_embeddings(),
        persist_directory=str(CHROMA_DIRECTORY),
    )


def search_documents(question, top_k):
    vector_store = load_vector_store()

    results = vector_store.similarity_search_with_score(
        query=question,
        k=top_k,
    )

    print("\nQuestion :")
    print(question)

    print(f"\n{len(results)} passages récupérés avec top_k={top_k}")

    for rank, (document, distance) in enumerate(results, start=1):
        file_name = document.metadata.get("file_name", "Inconnu")
        page = document.metadata.get("page", 0) + 1

        print(f"\n{'=' * 70}")
        print(f"Rang : {rank}")
        print(f"Fichier : {file_name}")
        print(f"Page : {page}")
        print(f"Distance : {distance:.4f}")
        print("-" * 70)
        print(document.page_content)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Recherche vectorielle dans les PDF du projet Alpha"
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Nombre de chunks à récupérer",
    )

    parser.add_argument(
        "--question",
        type=str,
        default=DEFAULT_QUESTION,
        help="Question à rechercher",
    )

    args = parser.parse_args()

    search_documents(
        question=args.question,
        top_k=args.top_k,
    )