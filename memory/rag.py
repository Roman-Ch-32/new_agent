# memory/rag.py
"""Qdrant Retriever — Поиск релевантных документов в проекте"""

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, Filter, FieldCondition, MatchValue
from sentence_transformers import SentenceTransformer


class QdrantRetriever:
    """Retriever для поиска в Qdrant (проект)"""

    def __init__(self, qdrant_url='http://localhost:6333'):
        self.client = QdrantClient(url=qdrant_url)
        self.model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        self.collection = 'ue_project'
        self.vector_size = self.model.get_sentence_embedding_dimension() or 384
        self._ensure_collection()

    def _ensure_collection(self):
        """Создаёт коллекцию если не существует"""
        try:
            exists = self.client.collection_exists(self.collection)
            if not exists:
                self.client.create_collection(
                    collection_name=self.collection,
                    vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE),
                )
                print(f"✅ Создана коллекция: {self.collection}")
        except Exception as e:
            print(f"⚠️ Warning creating collection: {e}")

    def search(self, query: str, limit: int = 10, project_filter: str = None):
        """Ищет релевантные документы через query_points"""
        embedding = self.model.encode(query, normalize_embeddings=True)
        vector = list(embedding)

        query_filter = None
        if project_filter:
            query_filter = Filter(
                must=[FieldCondition(key='project', match=MatchValue(value=project_filter))]
            )

        try:
            response = self.client.query_points(
                collection_name=self.collection,
                query=vector,
                query_filter=query_filter,
                limit=limit,
                with_payload=True,
                with_vectors=False,
            )

            return [
                {
                    'path': r.payload.get('path', '') if r.payload else '',
                    'content': r.payload.get('content', '') if r.payload else '',
                    'score': float(r.score) if r.score else 0.0,
                    'project': r.payload.get('project', '') if r.payload else '',
                }
                for r in response.points
            ]
        except Exception as e:
            # Коллекция не существует или другая ошибка — возвращаем пустой результат
            print(f"⚠️ RAG search error: {e}")
            return []