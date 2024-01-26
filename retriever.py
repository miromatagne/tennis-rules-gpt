from sentence_transformers import SentenceTransformer, util, CrossEncoder
import json


def get_relevant_chunks(chunks, embeddings, query, top_k, top_k_multiplier):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    query_embedding = model.encode(query, normalize_embeddings=True)
    similarities = util.dot_score(query_embedding, embeddings)
    scores = {}
    for chunk_id, score in zip(chunks, similarities[0]):
        chunk_json = json.loads(chunks[chunk_id])
        scores[chunk_id] = {"score": score}
        for metadata_id, metadata in chunk_json.items():
            scores[chunk_id][metadata_id] = str(metadata)
    scores = [(k, v) for k, v in sorted(scores.items(), key=lambda item: item[1]["score"], reverse=True)]
    scores = scores[:min(top_k*top_k_multiplier, len(scores))]
    scores = {k: v for k, v in scores}
    cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    cross_input = [[query, scores[i]["text"]] for i in scores.keys()]
    cross_scores = cross_encoder.predict(cross_input)
    final_scores = {}
    for i, key in enumerate(scores.keys()):
        final_scores[key] = scores[key]
        final_scores[key]["cross_score"] = cross_scores[i]

    final_scores = {k: v for k, v in sorted(final_scores.items(), key=lambda item: item[1]["cross_score"], reverse=True)}
    # final_scores = merge_neighbour_chunks(final_scores)
    final_scores = {k: v for k, v in list(final_scores.items())[:min(top_k, len(final_scores.keys()))]}
    # print(final_scores.keys())
    return final_scores

