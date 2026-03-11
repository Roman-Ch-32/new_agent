# agent/agent.py
"""AI Agent — Главный граф LangGraph с накапливаемым контекстом"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from agent.state import AgentState, AgentStatus
from agent.config import config
from agent.system_prompt import get_system_prompt
from piplines.registry import PipelineRegistry
from piplines.abc_pipline import BasePipeline
from memory.session_store import SessionStore, SessionContext
from agent.context_manager import ContextManager


def _messages_to_dict(messages: list[BaseMessage]) -> list[dict]:
    """Конвертирует сообщения в dict для сериализации"""
    result = []
    for msg in messages:
        result.append({'type': type(msg).__name__, 'content': str(msg.content)})
    return result


def _dict_to_messages(data: list[dict]) -> list[BaseMessage]:
    """Восстанавливает сообщения из dict"""
    result = []

    for item in data:
        msg_type = item.get('type', 'HumanMessage')
        content = item.get('content', '')
        if msg_type == 'HumanMessage':
            result.append(HumanMessage(content=content))
        elif msg_type == 'AIMessage':
            result.append(AIMessage(content=content))
        elif msg_type == 'SystemMessage':
            result.append(SystemMessage(content=content))

    return result


@dataclass
class Agent:
    """AI Agent с накапливаемым контекстом"""

    qdrant_url: str = field(default_factory=lambda: getattr(config.qdrant, 'url', 'http://localhost:6333'))
    project_path: str = field(default_factory=lambda: getattr(config.project, 'path', '/tmp'))
    debug: bool = field(default_factory=lambda: getattr(config, 'debug', False))
    llm: ChatOpenAI = field(init=False)
    registry: PipelineRegistry = field(init=False)
    session_store: SessionStore = field(init=False)
    graph: Any = field(init=False)
    system_prompt: str = field(init=False)

    def __post_init__(self) -> None:
        self.system_prompt = get_system_prompt()
        self.llm = self._create_llm()
        self.session_store = SessionStore(self.qdrant_url)
        self.registry = PipelineRegistry(self.llm)
        self.graph = self._build_graph()

    def _create_llm(self) -> ChatOpenAI:
        return ChatOpenAI(
            model=getattr(config.llm, 'model', 'qwen-ue'),
            base_url=getattr(config.llm, 'base_url', 'http://localhost:8080/v1'),
            temperature=getattr(config.llm, 'temperature', 0.7),
            api_key='sk-no-key-required',
            top_p=getattr(config.llm, 'top_p', 0.9),
        )

    def _add_system_message(self, messages: list[BaseMessage]) -> list[BaseMessage]:
        """Добавляет системный промпт в начало сообщений"""
        system_msg = SystemMessage(content=self.system_prompt)
        return [system_msg] + list(messages)

    def _route_to_pipeline(self, state: AgentState) -> AgentState:
        """Узел роутинга — выбирает и выполняет пайплайн"""
        last_msg_result = state.get_last_user_message()
        last_msg = last_msg_result if isinstance(last_msg_result, str) else ''

        pipeline = self.registry.select(last_msg)

        if self.debug:
            print(f'\n[DEBUG] Pipeline: {pipeline.name}')

        # ✅ Добавляем системный промпт перед выполнением пайплайна
        messages_with_system = self._add_system_message(state.messages)
        state_with_system = AgentState(
            messages=messages_with_system,
            rag_context=state.rag_context,
            token_usage=state.token_usage,
            fix_iterations=state.fix_iterations,
        )

        result_state = pipeline.execute(state_with_system)

        return AgentState(
            messages=result_state.messages,
            result=result_state.result,
            status=result_state.status,
            rag_context=result_state.rag_context or state.rag_context,
            token_usage=result_state.token_usage or state.token_usage,
            fix_iterations=result_state.fix_iterations,
        )

    def _router(self, state: AgentState) -> str:
        if state.status == AgentStatus.ERROR:
            return 'debug'
        return 'end'

    def _debug_node(self, state: AgentState) -> AgentState:
        debug_info = {
            'status': state.status.value,
            'messages_count': len(state.messages) if state.messages else 0,
            'token_usage': state.token_usage,
            'fix_iterations': state.fix_iterations,
        }
        msg = AIMessage(content=f'🔍 Debug:\n```json\n{debug_info}\n```')
        return AgentState(messages=[msg], result=msg.content, status=AgentStatus.SUCCESS)

    def _build_graph(self):
        workflow = StateGraph(AgentState)
        workflow.add_node('route', self._route_to_pipeline)
        workflow.add_node('debug', self._debug_node)
        workflow.set_entry_point('route')
        workflow.add_conditional_edges('route', self._router, {'end': END, 'debug': 'debug'})
        workflow.add_edge('debug', END)
        return workflow.compile()

    def invoke(self, messages=None, session_id='default', project=None, **kwargs):
        """Вызов агента с RAG обогащением и накапливаемым контекстом"""
        session_ctx = self.session_store.get(session_id)
        if not session_ctx:
            session_ctx = SessionContext(session_id=session_id)

        last_msg_result = None
        if messages:
            last_msg_result = messages[-1].content
        last_msg = last_msg_result if isinstance(last_msg_result, str) else ''

        context_mgr = ContextManager(session_ctx, self.qdrant_url)

        if project:
            context_mgr.add_project(project)

        new_docs = context_mgr.enrich_with_rag(last_msg, project=project, limit=10)
        accumulated_context, context_text = context_mgr.build_current_context(new_docs)

        stored_messages = []
        if session_ctx.messages:
            stored_messages = _dict_to_messages(session_ctx.messages)

        all_messages = stored_messages + (messages or [])

        initial_state = AgentState(
            messages=all_messages,
            rag_context=accumulated_context,
            token_usage=session_ctx.token_usage,
            fix_iterations=session_ctx.fix_iterations,
            **kwargs
        )

        result = AgentState.model_validate(self.graph.invoke(initial_state))

        messages_list = list(result.messages) if result.messages else []
        new_ctx = SessionContext(
            session_id=session_id,
            messages=_messages_to_dict(messages_list),
            accumulated_context=context_mgr.session.accumulated_context,
            context_summary=context_mgr.session.context_summary,
            projects=context_mgr.session.projects,
            token_usage=result.token_usage or {},
            fix_iterations=result.fix_iterations,
        )
        self.session_store.save(new_ctx)

        return result

    async def ainvoke(self, messages=None, session_id='default', project=None, **kwargs):
        return self.invoke(messages, session_id, project, **kwargs)

    def list_pipelines(self):
        return self.registry.list_all()

    def clear_session(self, session_id='default'):
        self.session_store.delete(session_id)

    def find_similar_sessions(self, query, limit=5):
        return self.session_store.search_similar_sessions(query, limit)

    def list_sessions(self):
        return self.session_store.list_sessions()