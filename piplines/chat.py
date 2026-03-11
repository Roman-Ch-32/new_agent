# piplines/chat.py
"""Chat Pipeline — SubGraph для обычного чата"""

from typing import Any
from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph

from agent.state import AgentStatus, AgentState
from piplines.abc_pipline import BasePipeline


class ChatPipeline(BasePipeline):
    """Пайплайн чата — полноценный граф"""

    def __init__(self, llm: ChatOpenAI) -> None:
        self.llm = llm
        self.graph: CompiledStateGraph = self.build_graph()

    @property
    def name(self) -> str:
        return 'chat'

    @property
    def description(self) -> str:
        return 'Обычный диалог без контекста'

    @property
    def keywords(self) -> list[str]:
        return ['привет', 'кто ты', 'помоги', 'вопрос']

    def _llm_call(self, state: AgentState) -> AgentState:
        """Узел — вызов LLM"""
        response: AIMessage = self.llm.invoke(state.messages)
        return AgentState(
            messages=[response],
            result=response.content,
            status=AgentStatus.SUCCESS,
        )

    def build_graph(self) -> Any:
        """Строит граф пайплайна"""
        workflow: StateGraph = StateGraph(AgentState)
        workflow.add_node('llm_call', self._llm_call)
        workflow.set_entry_point('llm_call')
        workflow.add_edge('llm_call', END)
        return workflow.compile()

    def execute(self, state: AgentState) -> AgentState:
        """Выполняет граф пайплайна"""
        return AgentState.model_validate(self.graph.invoke(state))