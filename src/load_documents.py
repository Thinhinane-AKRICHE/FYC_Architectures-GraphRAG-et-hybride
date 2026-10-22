from pathlib import Path

from langchain_core.documents import Document
from pypdf import PdfReader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PDF_DIRECTORY = PROJECT_ROOT / "data" / "pdfs"


def load_pdfs():
    documents = []
    pdf_files = sorted(PDF_DIRECTORY.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(
            f"Aucun PDF trouvé dans le dossier : {PDF_DIRECTORY}"
        )

    for pdf_path in pdf_files:
        print(f"Chargement de : {pdf_path.name}")

        reader = PdfReader(str(pdf_path))

        for page_number, page in enumerate(reader.pages):
            text = page.extract_text() or ""

            if not text.strip():
                print(
                    f"Attention : aucun texte trouvé dans "
                    f"{pdf_path.name}, page {page_number + 1}"
                )
                continue

            document = Document(
                page_content=text,
                metadata={
                    "file_name": pdf_path.name,
                    "source": str(pdf_path),
                    "page": page_number,
                },
            )

            documents.append(document)

    return documents


if __name__ == "__main__":
    documents = load_pdfs()

    print("\nChargement terminé")
    print(f"Nombre total de pages : {len(documents)}")

    for index, document in enumerate(documents):
        print(f"\n--- Document {index + 1} ---")
        print("Fichier :", document.metadata["file_name"])
        print("Page :", document.metadata["page"] + 1)
        print(document.page_content[:200])