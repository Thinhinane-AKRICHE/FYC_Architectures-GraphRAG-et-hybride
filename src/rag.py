import argparse
import os

from openai import OpenAI

from search import DEFAULT_QUESTION, search_documents


MODEL_NAME = "gpt-5.6"


RAG_INSTRUCTIONS = """
Tu es un assistant spécialisé dans l'analyse documentaire.

Règles obligatoires :
1. Réponds uniquement à partir du contexte récupéré.
2. N'invente aucune information absente du contexte.
3. Si une partie de la réponse est introuvable, indique clairement :
   "Les documents récupérés ne permettent pas de préciser ce point."
4. Ne mélange jamais le projet Alpha et le projet Gamma.
5. Explique les relations causales dans leur ordre logique.
6. Cite les fichiers utilisés entre crochets.
7. Termine par la liste des sources utilisées.
"""


def build_context(results):
    context_parts = []

    for rank, (document, distance) in enumerate(results, start=1):
        file_name = document.metadata.get("file_name", "Inconnu")
        page = document.metadata.get("page", 0) + 1

        context_parts.append(
            f"""
--- SOURCE {rank} ---
Fichier : {file_name}
Page : {page}
Distance vectorielle : {distance:.4f}

{document.page_content}
""".strip()
        )

    return "\n\n".join(context_parts)


def generate_answer(question, results):
    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError(
            "La variable OPENAI_API_KEY n'est pas définie."
        )

    context = build_context(results)

    user_input = f"""
QUESTION :
{question}

CONTEXTE RÉCUPÉRÉ :
{context}

Réponds à la question en distinguant :
- la cause du retard ;
- les équipes directement touchées ;
- la nouvelle date de lancement ;
- les informations manquantes dans le contexte.
""".strip()

    client = OpenAI()

    response = client.responses.create(
        model=MODEL_NAME,
        instructions=RAG_INSTRUCTIONS,
        input=user_input,
        max_output_tokens=600,
        store=False,
    )

    return response.output_text


def run_rag(question, top_k):
    print(f"\nLancement du RAG avec top_k={top_k}")

    results = search_documents(
        question=question,
        top_k=top_k,
    )

    answer = generate_answer(
        question=question,
        results=results,
    )

    print("\n" + "=" * 70)
    print("RÉPONSE GÉNÉRÉE")
    print("=" * 70)
    print(answer)

    return answer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="RAG classique sur le corpus Projet Alpha"
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Nombre de chunks transmis au modèle",
    )

    parser.add_argument(
        "--question",
        type=str,
        default=DEFAULT_QUESTION,
        help="Question posée au RAG",
    )

    args = parser.parse_args()

    run_rag(
        question=args.question,
        top_k=args.top_k,
    )