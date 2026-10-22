import os

import streamlit as st

from rag import generate_answer
from search import DEFAULT_QUESTION, search_documents


EXPECTED_FACTS = {
    "Pièce critique X47": ("x47",),
    "Fournisseur Beta Mécanique": ("beta mécanique",),
    "Panne du four F-2": ("f-2",),
    "Industrialisation / cellule A3": ("industrialisation", "a3"),
    "Équipe Qualité Produit": ("qualité produit",),
    "Lancement le 6 octobre": ("6 octobre 2026",),
}


QUESTIONS = {
    "Question globale": DEFAULT_QUESTION,
    "Question locale": (
        "Quel incident s'est produit chez Beta Mécanique et quelles pièces "
        "sont concernées ?"
    ),
    "Question piège": (
        "Beta Mécanique est-elle responsable du retard du projet Gamma ?"
    ),
}


def prepare_sources(results):
    sources = []

    for rank, (document, distance) in enumerate(results, start=1):
        sources.append(
            {
                "rank": rank,
                "file_name": document.metadata.get("file_name", "Inconnu"),
                "page": document.metadata.get("page", 0) + 1,
                "distance": float(distance),
                "content": document.page_content,
            }
        )

    return sources


def evaluate_coverage(sources):
    retrieved_text = "\n".join(source["content"] for source in sources).lower()

    coverage = {}
    for label, keywords in EXPECTED_FACTS.items():
        coverage[label] = all(keyword in retrieved_text for keyword in keywords)

    return coverage


def display_retrieval_details(message):
    unique_files = {source["file_name"] for source in message["sources"]}
    coverage = message["coverage"]

    if coverage is not None:
        found_count = sum(coverage.values())
        total_count = len(coverage)
        percentage = round(100 * found_count / total_count)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("top_k utilisé", message["top_k"])
        col2.metric("Documents", len(unique_files))
        col3.metric("Faits retrouvés", f"{found_count}/{total_count}")
        col4.metric("Couverture", f"{percentage} %")

        with st.expander("Voir la couverture de la chaîne Alpha"):
            for label, found in coverage.items():
                icon = "✅" if found else "❌"
                st.write(f"{icon} {label}")
    else:
        col1, col2, col3 = st.columns(3)
        col1.metric("top_k utilisé", message["top_k"])
        col2.metric("Passages", len(message["sources"]))
        col3.metric("Documents", len(unique_files))

        st.caption(
            "La couverture automatique de la chaîne Alpha est calculée "
            "uniquement pour la question globale."
        )

    with st.expander("Voir les passages récupérés"):
        for source in message["sources"]:
            st.markdown(
                f"**Rang {source['rank']} — {source['file_name']} "
                f"(page {source['page']})**"
            )
            st.caption(f"Distance vectorielle : {source['distance']:.4f}")
            st.write(source["content"])
            st.divider()


def display_message(message):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant":
            display_retrieval_details(message)


st.set_page_config(
    page_title="Laboratoire RAG — Projet Alpha",
    page_icon="🔎",
    layout="wide",
)

st.title("🔎 Laboratoire RAG — Projet Alpha")
st.write(
    "Posez la même question avec plusieurs valeurs de **top_k** pour observer "
    "quels passages sont transmis au modèle et si toute la chaîne causale est "
    "retrouvée."
)

with st.sidebar:
    st.header("Paramètres de l'expérience")

    top_k = st.slider(
        "Nombre de passages récupérés (top_k)",
        min_value=1,
        max_value=14,
        value=3,
        step=1,
    )

    if os.getenv("OPENAI_API_KEY"):
        st.success("Clé OpenAI détectée")
    else:
        st.error("Variable OPENAI_API_KEY absente")

    st.divider()
    st.subheader("Questions de démonstration")

    selected_question = None
    for label, example_question in QUESTIONS.items():
        if st.button(label, use_container_width=True):
            selected_question = example_question

    st.divider()
    if st.button("Effacer la conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


if "messages" not in st.session_state:
    st.session_state.messages = []

for stored_message in st.session_state.messages:
    display_message(stored_message)

typed_question = st.chat_input("Posez une question sur les projets Alpha ou Gamma")
question = typed_question or selected_question

if question:
    user_message = {"role": "user", "content": question}
    st.session_state.messages.append(user_message)
    display_message(user_message)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Recherche des passages et génération de la réponse..."):
                results = search_documents(question=question, top_k=top_k)
                answer = generate_answer(question=question, results=results)
                sources = prepare_sources(results)
                coverage = (
                    evaluate_coverage(sources)
                    if question == DEFAULT_QUESTION
                    else None
                )

            assistant_message = {
                "role": "assistant",
                "content": answer,
                "top_k": top_k,
                "sources": sources,
                "coverage": coverage,
            }

            st.markdown(answer)
            display_retrieval_details(assistant_message)
            st.session_state.messages.append(assistant_message)

        except Exception as error:
            st.error(f"Erreur : {error}")
            st.info(
                "Vérifiez que l'index Chroma existe et que la variable "
                "OPENAI_API_KEY est définie."
            )
