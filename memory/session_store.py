# memory/session_store.py
"""Session Store — хранение контекста сессий в Qdrant с семантическим поиском"""

from dataclasses import dataclass, field
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import hashlib
import time

from memory.embedding_model import get_embedding_model


@dataclass
class SessionContext:
    """Контекст сессии с накапливаемым контекстом"""
    session_id: str
    messages: list[dict] = field(default_factory=list)
    accumulated_context: list[dict] = field(default_factory=list)
    context_summary: str = ''
    projects: list[str] = field(default_factory=list)
    token_usage: dict = field(default_factory=dict)
    fix_iterations: int = 0
    updated_at: float = 0.0
    max_context_docs: int = 50


class SessionStore:
    """Хранение сессий в Qdrant с семантическим поиском"""

    def __init__(self, qdrant_url='http://localhost:6333'):
        self.client = QdrantClient(url=qdrant_url)
        self.collection = 'sessions'
        self.model = get_embedding_model()
        self.vector_size = self.model.get_sentence_embedding_dimension() or 384
        self._ensure_collection()

    def _ensure_collection(self):
        """Создаёт коллекцию если не существует (версия 1.17.x)"""
        exists = self.client.collection_exists(self.collection)

        if not exists:
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE),
            )

    def _session_vector(self, text: str):
        embedding = self.model.encode(text, normalize_embeddings=True)
        return list(embedding)

    def _session_key(self, session_id: str) -> int:
        return int(hashlib.md5(session_id.encode()).hexdigest(), 16) % (10 ** 9)

    def _get_session_text(self, ctx: SessionContext) -> str:
        texts = []
        for msg in ctx.messages[-10:]:
            content = msg.get('content', '')
            if content:
                texts.append(content)
        return ' '.join(texts)

    def get(self, session_id: str = 'default'):
        try:
            point = self.client.retrieve(
                collection_name=self.collection,
                ids=[self._session_key(session_id)],
            )
            if not point:
                return None

            payload = point[0].payload

            return SessionContext(
                session_id=session_id,
                messages=payload.get('messages', []) if payload else [],
                accumulated_context=payload.get('accumulated_context', []) if payload else [],
                context_summary=payload.get('context_summary', '') if payload else '',
                projects=payload.get('projects', []) if payload else [],
                token_usage=payload.get('token_usage', {}) if payload else {},
                fix_iterations=payload.get('fix_iterations', 0) if payload else 0,
                updated_at=payload.get('updated_at', 0.0) if payload else 0.0,
            )
        except Exception:
            return None

    def save(self, ctx: SessionContext):
        ctx.updated_at = time.time()
        session_text = self._get_session_text(ctx)
        vector = self._session_vector(session_text) if session_text else [0.0] * self.vector_size

        self.client.upsert(
            collection_name=self.collection,
            points=[PointStruct(
                id=self._session_key(ctx.session_id),
                vector=vector,
                payload={
                    'session_id': ctx.session_id,
                    'messages': ctx.messages,
                    'accumulated_context': ctx.accumulated_context,
                    'context_summary': ctx.context_summary,
                    'projects': ctx.projects,
                    'token_usage': ctx.token_usage,
                    'fix_iterations': ctx.fix_iterations,
                    'updated_at': ctx.updated_at,
                },
            )],
        )

    def search_similar_sessions(self, query: str, limit: int = 5):
        """Ищет похожие сессии через query_points (версия 1.17.x)"""
        vector = self._session_vector(query)

        response = self.client.query_points(
            collection_name=self.collection,
            query=vector,
            limit=limit,
            with_payload=True,
        )

        sessions = []
        for result in response.points:
            payload = result.payload if result.payload else {}
            sessions.append(SessionContext(
                session_id=payload.get('session_id', ''),
                messages=payload.get('messages', []),
                accumulated_context=payload.get('accumulated_context', []),
                context_summary=payload.get('context_summary', ''),
                projects=payload.get('projects', []),
                token_usage=payload.get('token_usage', {}),
                fix_iterations=payload.get('fix_iterations', 0),
                updated_at=payload.get('updated_at', 0.0),
            ))
        return sessions

    def delete(self, session_id: str):
        self.client.delete(
            collection_name=self.collection,
            points_selector=[self._session_key(session_id)],
        )

    def list_sessions(self):
        sessions = []
        offset = None
        while True:
            records, offset = self.client.scroll(
                collection_name=self.collection,
                limit=100,
                offset=offset,
            )
            for record in records:
                payload = record.payload if record.payload else {}
                sessions.append({
                    'session_id': payload.get('session_id', ''),
                    'updated_at': payload.get('updated_at', 0.0),
                    'messages_count': len(payload.get('messages', [])),
                    'context_docs_count': len(payload.get('accumulated_context', [])),
                    'projects': payload.get('projects', []),
                })
            if offset is None:
                break
        return sessions
