import text_splitter
from sentence_transformers import SentenceTransformer, util
import json
import numpy as np
from tqdm import tqdm


def generate_embeddings(pdf_contents):
    splitter = text_splitter.RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
    model = SentenceTransformer('all-MiniLM-L6-v2')
    chunks = {}
    embeddings = []
    print("Generating embeddings")
    for pdf_id, pdf_data in tqdm(pdf_contents.items()):
        texts = splitter.split_text(pdf_data["text"])
        print(f"Generated {len(texts)} splits for PDF {pdf_id}")
        if not texts:
            continue
        for i in range(len(texts)):
            temp_dict = {"pdf_id": str(pdf_id), "chunk_id": str(i)}
            for metadata_id, metadata in pdf_data.items():
                temp_dict[metadata_id] = str(metadata)
            temp_dict["text"] = texts[i]
            chunks[str(pdf_id) + "_" + str(i)] = json.dumps(temp_dict)
            embeddings.append(model.encode(texts[i]))
    embeddings = np.array(embeddings)
    np.save("embeddings.npy", embeddings)
    with open("chunks.json", "w") as out:
        json.dump(chunks, out, indent=2)
    return chunks, embeddings
