from sentence_transformers import SentenceTransformer
import json
import numpy as np
from tqdm import tqdm
from text_splitter import RecursiveCharacterTextSplitter


def generate_embeddings():
    pdf_contents = json.load(open("outputs/pdf_contents.json"))
    splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    model = SentenceTransformer('all-MiniLM-L6-v2')
    chunks = {}
    embeddings = []
    for pdf_id, pdf_data in tqdm(pdf_contents.items()):
        texts = splitter.split_text(pdf_data["text"])
        for i in range(len(texts)):
            temp_dict = {"pdf_id": str(pdf_id), "chunk_id": str(i)}
            for metadata_id, metadata in pdf_data.items():
                temp_dict[metadata_id] = str(metadata)
            temp_dict["text"] = texts[i]
            chunks[str(pdf_id) + "_" + str(i)] = json.dumps(temp_dict)
            embeddings.append(model.encode(texts[i]))
    embeddings = np.array(embeddings)
    np.save("outputs/embeddings.npy", embeddings)
    with open("outputs/chunks.json", "w") as out:
        json.dump(chunks, out, indent=2, ensure_ascii=False)
    return chunks, embeddings


if __name__ == "__main__":
    generate_embeddings()
