# piplines/abc_pipline.py
"""Base Pipeline — Абстрактный интерфейс"""

from abc import ABC, abstractmethod
from typing import Any

from agent.state import AgentState


class BasePipeline(ABC):
    """Базовый класс пайплайна — каждый пайплайн это SubGraph"""

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        pass

    @property
    @abstractmethod
    def keywords(self) -> list[str]:
        pass

    @abstractmethod
    def build_graph(self) -> Any:
        """Строит и возвращает граф пайплайна"""
        pass

    @abstractmethod
    def execute(self, state: AgentState) -> AgentState:
        """Выполняет граф пайплайна"""
        pass

    def should_activate(self, query: str) -> bool:
        """Проверяет нужно ли активировать пайплайн"""
        query_lower = query.lower()
        return any(kw in query_lower for kw in self.keywords)