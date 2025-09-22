import chromadb
from sentence_transformers import SentenceTransformer

if __name__ == "__main__":
    DB = "chroma_db"
    COLLECTION = "knowledge-base"
    MODEL = "sentence-transformers/all-MiniLM-L6-v2"

    client = chromadb.PersistentClient(path=DB)
    collection = client.get_or_create_collection(name=COLLECTION)
    model = SentenceTransformer(MODEL)

    q = ""

    # where={"namespace": "products"}
    results = collection.query(query_embeddings=model.encode([q]).tolist(), n_results=5)
