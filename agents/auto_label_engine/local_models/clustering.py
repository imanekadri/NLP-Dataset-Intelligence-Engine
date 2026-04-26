from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans


class TextClustering:

    def __init__(self, n_clusters=5):
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        self.n_clusters = n_clusters

    def cluster(self, texts):
        embeddings = self.embedder.encode(texts)
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
        labels = kmeans.fit_predict(embeddings)

        return labels.tolist()