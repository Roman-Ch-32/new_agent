# agent/state.py
"""Agent State — Структура данных для LangGraph (Pydantic)"""

from enum import Enum
from typing import Annotated, Sequence, Any
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field


class AgentStatus(str, Enum):
    """Статусы агента"""
    IDLE = 'idle'
    PROCESSING = 'processing'
    WAITING = 'waiting'
    ERROR = 'error'
    SUCCESS = 'success'


class AgentState(BaseModel):
    """Состояние агента для LangGraph"""

    class Config:
        arbitrary_types_allowed = True

    messages: Annotated[Sequence[BaseMessage], add_messages] = Field([], description='История диалога с пользователем')
    result: str = Field('', description='Результат выполнения последней операции')
    status: AgentStatus = Field(AgentStatus.IDLE, description='Текущий статус агента')
    token_usage: dict[str, int] | None = Field(None, description='Статистика использования токенов LLM')
    asset_specs: list[dict[str, Any]] | None = Field(None, description='Специации сгенерированных ассетов')
    generated_assets: list[str] | None = Field(None, description='Пути к созданным файлам ассетов')
    ue_project_path: str | None = Field(None, description='Путь к корню проекта Unreal Engine')
    current_task: str | None = Field(None, description='Текущая задача в выполнении')
    detected_patterns: list[str] | None = Field(None, description='Обнаруженные паттерны в запросах пользователя')
    rag_context: list[dict[str, Any]] | None = Field(None, description='Контекст из векторной базы (Qdrant)')
    relevant_files: list[str] | None = Field(None, description='Найденные релевантные файлы проекта')
    git_branch: str | None = Field(None, description='Имя текущей Git-ветки')
    verification_result: dict[str, Any] | None = Field(None, description='Результат проверки выполненной операции')
    fix_iterations: int = Field(0, description='Количество попыток самоисправления')

    def add_message(self, message: BaseMessage) -> None:
        """Добавляет сообщение в историю"""
        if self.messages is None:
            self.messages = []
        self.messages = list(self.messages) + [message]

    def get_last_user_message(self) -> str | list[Any] | None:
        """Возвращает последнее сообщение пользователя"""
        if not self.messages:
            return None
        from langchain_core.messages import HumanMessage
        for msg in reversed(self.messages):
            if isinstance(msg, HumanMessage):
                return msg.content
        return None

    def get_token_count(self) -> int:
        """Подсчитывает общее количество токенов"""
        if not self.token_usage:
            return 0
        return self.token_usage.get('total_tokens', 0)