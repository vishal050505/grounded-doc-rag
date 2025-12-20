import os
from pypdf import PdfReader


def load_documents(path):
    documents = []

    #  CASE 1: Single PDF file 
    if os.path.isfile(path) and path.endswith(".pdf"):
        reader = PdfReader(path)
        text = ""

        for page in reader.pages:
            if page.extract_text():
                text += page.extract_text()

        documents.append({
            "source": os.path.basename(path),
            "text": text
        })

        return documents

    #  CASE 2: Folder with PDFs 
    if os.path.isdir(path):
        for file in os.listdir(path):
            if file.endswith(".pdf"):
                file_path = os.path.join(path, file)
                reader = PdfReader(file_path)
                text = ""

                for page in reader.pages:
                    if page.extract_text():
                        text += page.extract_text()

                documents.append({
                    "source": file,
                    "text": text
                })

        return documents

    #  INVALID PATH 
    raise ValueError("Provided path is neither a PDF file nor a directory")
