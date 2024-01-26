import os
from tqdm import tqdm
import json
from pypdf import PdfReader


def extract_files(data_path="data/PDF", metadata_path="data/metadata.json"):
    pdf_contents = {}
    metadata = json.load(open(metadata_path))
    for file in tqdm(os.listdir(data_path)):
        file_name = file.split(".pdf")[0]
        reader = PdfReader(data_path + "/" + file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        pdf_contents[file_name] = {"text": text, "doc_nb": metadata["content"][file_name]["doc_nb"], "title": metadata["content"][file_name]["title"], "url": metadata["content"][file_name]["url"]}
    with open("outputs/pdf_contents.json", "w") as ofile:
        json.dump(pdf_contents, ofile)
    return pdf_contents


if __name__ == "__main__":
    pdf_content = extract_files()
