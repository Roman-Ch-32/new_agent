# piplines/registry.py
"""Pipeline Registry — управляет пайплайнами"""

from typing import Any
from langchain_openai import ChatOpenAI
from piplines.abc_pipline import BasePipeline


class PipelineRegistry:
    """Реестр пайплайнов"""

    def __init__(self, llm: ChatOpenAI) -> None:
        self.llm = llm
        self._pipelines: dict[str, BasePipeline] = {}
        self._default_name: str = 'chat'
        self._register_all()

    def _register_all(self) -> None:
        """Регистрирует все доступные пайплайны"""
        from piplines.chat import ChatPipeline
        # Пока только ChatPipeline — остальные добавишь позже
        self.register(ChatPipeline(self.llm), is_default=True)

    def register(self, pipeline: BasePipeline, is_default: bool = False) -> None:
        """Регистрирует пайплайн"""
        self._pipelines[pipeline.name] = pipeline
        if is_default:
            self._default_name = pipeline.name

    def get(self, name: str) -> BasePipeline | None:
        """Получает пайплайн по имени"""
        return self._pipelines.get(name)

    def select(self, query: str) -> BasePipeline:
        """Выбирает пайплайн по запросу"""
        for pipeline in self._pipelines.values():
            if pipeline.should_activate(query):
                return pipeline
        default = self._pipelines.get(self._default_name)
        if default is None:
            raise RuntimeError('No default pipeline registered')
        return default

    def list_all(self) -> list[dict[str, Any]]:
        """Список всех пайплайнов"""
        return [{'name': p.name, 'description': p.description, 'keywords': p.keywords} for p in self._pipelines.values()]