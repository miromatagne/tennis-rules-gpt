import os
from langchain.document_loaders import UnstructuredPDFLoader
from tqdm import tqdm
import json


def extract_files():
    pdf_contents = {}
    metadata = json.load(open("data/metadata.json"))
    for file in tqdm(os.listdir("data/PDF")):
        file_name = file.split(".pdf")[0]
        loader = UnstructuredPDFLoader(file)
        data = loader.load()
        text = data[0].page_content
        pdf_contents[file_name] = {"text": text, "doc_nb": metadata[file_name]["doc_nb"], "title": metadata[file_name]["title"], "url": metadata[file_name]["url"]}
    return pdf_contents
