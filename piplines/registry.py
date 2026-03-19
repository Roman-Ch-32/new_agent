# piplines/registry.py
"""Pipeline Registry — управляет пайплайнами"""

from langchain_openai import ChatOpenAI
from piplines.abc_pipline import BasePipeline
from agent.config import config


class PipelineRegistry:
    """Реестр пайплайнов"""

    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
        self._pipelines = {}
        self._default_name = 'chat'
        self._register_all()

    def _register_all(self):
        """Регистрирует все доступные пайплайны"""
        from piplines.chat import ChatPipeline
        from piplines.tool_pipeline import ToolPipeline

        # Chat pipeline
        self.register(ChatPipeline(self.llm), is_default=True)

        # Tool pipeline с инструментами
        try:
            tool_pipeline = ToolPipeline(
                llm=self.llm,
                project_path=getattr(config.project, 'path', '/tmp'),
                qdrant_url=getattr(config.qdrant, 'url', 'http://localhost:6333')
            )
            self.register(tool_pipeline)
        except Exception as e:
            print(f"⚠️ Не удалось загрузить ToolPipeline: {e}")

    def register(self, pipeline: BasePipeline, is_default: bool = False):
        """Регистрирует пайплайн"""
        self._pipelines[pipeline.name] = pipeline
        if is_default:
            self._default_name = pipeline.name

    def get(self, name: str):
        """Получает пайплайн по имени"""
        return self._pipelines.get(name)

    def select(self, query: str) -> BasePipeline:
        """Выбирает пайплайн по запросу"""
        query_lower = query.lower()

        # Tool pipeline для поисковых запросов и команд
        tool_keywords = ['найди', 'покажи', 'прочитай', 'индексируй', 'поиск', 'класс', 'функция', 'файл', 'структур']
        if any(kw in query_lower for kw in tool_keywords):
            tool = self._pipelines.get('tool')
            if tool:
                return tool

        # Иначе ищем по ключевым словам
        for pipeline in self._pipelines.values():
            if pipeline.should_activate(query):
                return pipeline

        # Default
        default = self._pipelines.get(self._default_name)
        if default is None:
            raise RuntimeError('No default pipeline registered')
        return default

    def list_all(self):
        """Список всех пайплайнов"""
        return [
            {
                'name': p.name,
                'description': p.description,
                'keywords': p.keywords
            }
            for p in self._pipelines.values()
        ]