from enum import Enum
from typing import Annotated, Sequence, Any, Dict, List

from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field


class AgentStatus(str, Enum):
    IDLE = 'idle'
    PLANNING = 'planning'
    EXECUTING = 'executing'
    VERIFYING = 'verifying'
    RETRY = 'retry'
    SUCCESS = 'success'
    ERROR = 'error'


class AgentState(BaseModel):
    """Состояние агента для LangGraph."""

    model_config = {'arbitrary_types_allowed': True}

    messages: Annotated[Sequence[BaseMessage], add_messages] = Field(default_factory=list)
    status: AgentStatus = Field(default=AgentStatus.IDLE)

    plan: Dict[str, Any] = Field(default_factory=dict)
    tool_results: str = Field(default='')
    execution_result: Dict[str, Any] = Field(default_factory=dict)
    verification_result: Dict[str, Any] = Field(default_factory=dict)
    trace_events: List[Dict[str, Any]] = Field(default_factory=list)

    rag_context: List[Dict[str, Any]] = Field(default_factory=list)
    token_usage: Dict[str, Any] = Field(default_factory=dict)
    fix_iterations: int = Field(default=0)
    current_task: str = Field(default='')
    ue_project_path: str = Field(default='')

    result: str = Field(default='')

    def get_last_user_message(self) -> str | None:
        if not self.messages:
            return None
        for msg in reversed(self.messages):
            if isinstance(msg, HumanMessage):
                content = msg.content
                if isinstance(content, str):
                    return content
                return str(content)
        return None

    def is_max_iterations_reached(self, max_iterations: int = 5) -> bool:
        return self.fix_iterations >= max_iterations
