# agent/context_manager.py
"""Context Manager — управление накапливаемым контекстом"""

from typing import Any
from memory.session_store import SessionContext
from memory.rag import QdrantRetriever


class ContextManager:
    """Управляет контекстом сессии — накопление, компрессия, обогащение"""

    def __init__(self, session_ctx: SessionContext, qdrant_url: str) -> None:
        self.session = session_ctx
        self.retriever = QdrantRetriever(qdrant_url)
        self.max_docs = session_ctx.max_context_docs

    def enrich_with_rag(self, query: str, project: str | None = None, limit: int = 10) -> list[dict[str, Any]]:
        """Обогащает запрос RAG поиском"""
        new_docs = self.retriever.search(query, limit=limit, project_filter=project)
        return new_docs

    def build_current_context(self, new_docs: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], str]:
        """Строит текущий контекст для LLM"""
        self.session.accumulated_context.extend(new_docs)

        if len(self.session.accumulated_context) > self.max_docs:
            self.session.context_summary = self._compress_context()
            self.session.accumulated_context = []

        context_text = self._format_context(self.session.accumulated_context)
        if self.session.context_summary:
            context_text = f"📜 СУММАРИ ПРЕДЫДУЩЕГО КОНТЕКСТА:\n{self.session.context_summary}\n\n{context_text}"

        return self.session.accumulated_context, context_text

    def _compress_context(self) -> str:
        """Сжимает контекст через LLM (заглушка)"""
        return f"[Сжато {len(self.session.accumulated_context)} документов]"

    def _format_context(self, docs: list[dict[str, Any]]) -> str:
        """Форматирует документы для LLM"""
        if not docs:
            return ''

        text = '\n\n📚 РЕЛЕВАНТНЫЙ КОНТЕКСТ ИЗ ПРОЕКТА:\n'
        for i, doc in enumerate(docs[:20], 1):
            text += f"\n{i}. 📁 {doc.get('path', 'unknown')}\n```\n{doc.get('content', '')[:500]}\n```"
        return text

    def add_project(self, project: str) -> None:
        """Добавляет проект в сессию"""
        if project not in self.session.projects:
            self.session.projects.append(project)