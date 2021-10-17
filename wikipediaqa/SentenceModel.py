from sentence_transformers import SentenceTransformer
from scipy.spatial import distance
import numpy as np


class SentenceModel:
    def __init__(self, sentence_model=None, device="cpu"):
        if not sentence_model:
            sentence_model = "sentence-transformers/paraphrase-distilroberta-base-v2"

        self.model = SentenceTransformer(sentence_model, device=device)
        
    def compare(self, question, search):
        embeddings = self.model.encode([question] + search)

        distances = [distance.cosine(embeddings[0], embeddings[i])
                                    for i in range(1, len(embeddings))]

        best = np.argmin(distances)

        return best