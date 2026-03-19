# agent/agent.py
"""AI Agent — JSON парсинг для llama.cpp (рабочая версия)"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, List
import re
import json
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, SystemMessage, ToolMessage
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool, BaseTool

from agent.config import config
from agent.state import AgentState, AgentStatus
from agent.system_prompt import get_system_prompt
from memory.session_store import SessionStore, SessionContext


# ============================================================================
# ИНСТРУМЕНТЫ (LangChain @tool)
# ============================================================================

@tool
def index_directory(directory: str, project_name: str = "ue_project", recursive: bool = True,
                    limit: int = None) -> dict:
    """Индексирует директорию проекта в Qdrant"""
    from mcp.indexer import FileIndexer
    indexer = FileIndexer()
    return indexer.index_directory(directory, project_name, recursive, limit)


@tool
def index_file(file_path: str, project_name: str = "ue_project") -> dict:
    """Индексирует файл в Qdrant"""
    from mcp.indexer import FileIndexer
    indexer = FileIndexer()
    return indexer.index_file(file_path, project_name)


@tool
def search_indexed(query: str, limit: int = 10) -> list:
    """Поиск по проиндексированным файлам в Qdrant"""
    from mcp.indexer import FileIndexer
    indexer = FileIndexer()
    return indexer.search_indexed(query, limit)


@tool
def get_project_structure(max_depth: int = 3) -> dict:
    """Получить структуру проекта"""
    from mcp.file_system import FileSystemTools
    import os
    tools = FileSystemTools(os.getcwd())
    return tools.get_project_structure(max_depth)


@tool
def find_class(class_name: str) -> list:
    """Найти класс в проекте"""
    from mcp.code_analyzer import CodeAnalyzer
    import os
    analyzer = CodeAnalyzer(os.getcwd())
    return analyzer.find_class(class_name)


@tool
def find_function(function_name: str) -> list:
    """Найти функцию в проекте"""
    from mcp.code_analyzer import CodeAnalyzer
    import os
    analyzer = CodeAnalyzer(os.getcwd())
    return analyzer.find_function(function_name)


@tool
def read_file(file_path: str) -> str:
    """Читать файл из проекта"""
    from mcp.file_system import FileSystemTools
    import os
    tools = FileSystemTools(os.getcwd())
    return tools.read_file(file_path)


@tool
def get_indexed_files(project_name: str = "ue_project") -> list:
    """Список проиндексированных файлов"""
    from mcp.indexer import FileIndexer
    indexer = FileIndexer()
    return indexer.get_indexed_files(project_name)


@tool
def search_duckduckgo(query: str, num_results: int = 5) -> list:
    """Поиск в интернете"""
    from mcp.internet_search import InternetSearch
    search = InternetSearch()
    return search.search_duckduckgo(query, num_results)


ALL_TOOLS: list[BaseTool] = [
    index_directory,
    index_file,
    search_indexed,
    get_project_structure,
    find_class,
    find_function,
    read_file,
    get_indexed_files,
    search_duckduckgo,
]

TOOLS_BY_NAME: dict[str, BaseTool] = {t.name: t for t in ALL_TOOLS}


# ============================================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ============================================================================

def _get_message_content(message: BaseMessage) -> str:
    if not hasattr(message, 'content'):
        return ''
    content = message.content
    if content is None:
        return ''
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts = []
        for item in content:
            if isinstance(item, dict):
                texts.append(str(item.get('text', item.get('content', ''))))
            else:
                texts.append(str(item))
        return '\n'.join(texts)
    return str(content)


def _messages_to_dict(messages: list[BaseMessage]) -> list[dict]:
    result = []
    for msg in messages:
        result.append({'type': type(msg).__name__, 'content': str(msg.content)})
    return result


def _dict_to_messages(data: list[dict]) -> list[BaseMessage]:
    result = []
    for item in data:
        msg_type = item.get('type', 'HumanMessage')
        content = item.get('content', '')
        if msg_type == 'HumanMessage':
            result.append(HumanMessage(content=content))
        elif msg_type == 'AIMessage':
            result.append(AIMessage(content=content))
        elif msg_type == 'ToolMessage':
            result.append(ToolMessage(content=content, tool_call_id=''))
        elif msg_type == 'SystemMessage':
            result.append(SystemMessage(content=content))
    return result


# ✅ ДОБАВЛЕН ПАРСЕР
def _parse_tool_calls(text: str) -> list[dict]:
    """
    Парсит tool calls из текста LLM (для llama.cpp без нативных tool_calls)

    Поддерживаемые форматы:
    1. ```json {"tool": "...", "parameters": {...}} ```
    2. {"tool": "...", "parameters": {...}}
    """
    if not text or not isinstance(text, str):
        return []

    tool_calls = []

    # Формат 1: ```json {...}```
    pattern1 = r'```json\s*(\{.*?"tool".*?\})\s*```'
    matches1 = re.findall(pattern1, text, re.DOTALL)

    for match_str in matches1:
        try:
            tool_call = json.loads(match_str)
            tool_name = tool_call.get('tool', '')
            if tool_name:
                tool_calls.append({
                    'name': tool_name,
                    'args': tool_call.get('parameters', {})
                })
        except Exception as e:
            pass

    # Формат 2: Просто {...} с "tool" (без markdown)
    if not tool_calls:
        for match in re.finditer(r'\{[^{}]*"tool"[^{}]*\}', text, re.DOTALL):
            try:
                tool_call = json.loads(match.group())
                tool_name = tool_call.get('tool', '')
                if tool_name:
                    tool_calls.append({
                        'name': tool_name,
                        'args': tool_call.get('parameters', {})
                    })
            except:
                pass

    return tool_calls


# ============================================================================
# АГЕНТ
# ============================================================================

@dataclass
class Agent:
    """AI Agent — JSON парсинг для llama.cpp"""

    qdrant_url: str = field(default_factory=lambda: getattr(config.qdrant, 'url', 'http://localhost:6333'))
    project_path: str = field(default_factory=lambda: getattr(config.project, 'path', '/tmp'))
    debug: bool = field(default_factory=lambda: getattr(config, 'debug', True))
    llm: ChatOpenAI = field(init=False)
    session_store: SessionStore = field(init=False)
    graph: Any = field(init=False)
    system_prompt: str = field(init=False)

    def __post_init__(self) -> None:
        if self.debug:
            print(f"\n{'=' * 60}")
            print(f"[AGENT] Инициализация...")
            print(f"[AGENT] Project path: {self.project_path}")
            print(f"[AGENT] Qdrant URL: {self.qdrant_url}")

        self.system_prompt = get_system_prompt(self.project_path)
        self.llm = self._create_llm()

        # ❌ bind_tools() НЕ РАБОТАЕТ с llama.cpp — не используем
        # self.llm_with_tools = self.llm.bind_tools(ALL_TOOLS)

        self.session_store = SessionStore(self.qdrant_url)
        self.graph = self._build_graph()

        if self.debug:
            print(f"[AGENT] Готов! Инструментов: {len(ALL_TOOLS)}")
            print(f"{'=' * 60}\n")

    def _create_llm(self) -> ChatOpenAI:
        return ChatOpenAI(
            model=getattr(config.llm, 'model', 'qwen-ue'),
            base_url=getattr(config.llm, 'base_url', 'http://localhost:8080/v1'),
            temperature=getattr(config.llm, 'temperature', 0.1),
            api_key='sk-no-key-required',
            top_p=getattr(config.llm, 'top_p', 0.9),
            max_tokens=getattr(config.llm, 'num_predict', 2048),
        )

    def _llm_node(self, state: AgentState) -> dict:
        """LLM генерирует ответ"""
        query = _get_message_content(state.messages[-1]) if state.messages else ''

        if not query:
            return {'messages': state.messages, 'status': AgentStatus.SUCCESS, 'result': ''}

        if self.debug:
            print(f"\n[LLM] Запрос: {query[:100]}...")

        system_msg = SystemMessage(content=self.system_prompt)
        messages = [system_msg] + list(state.messages)

        # ✅ Обычный invoke (не llm_with_tools)
        response = self.llm.invoke(messages)

        # ✅ Парсинг JSON из текста
        tool_calls = _parse_tool_calls(response.content)

        if self.debug:
            print(f"[LLM] Найдено tool calls: {len(tool_calls)}")
            for tc in tool_calls:
                print(f"  - {tc['name']}: {tc['args']}")
            print(f"[LLM] Content: {response.content[:300] if response.content else 'None'}...")

        return {
            'messages': [response],
            'status': AgentStatus.SUCCESS,
            'result': response.content if response.content else '',
        }

    def _tool_executor_node(self, state: AgentState) -> dict:
        """Выполняет tool calls"""
        last_message = state.messages[-1] if state.messages else None

        if not last_message or not last_message.content:
            return {'messages': state.messages, 'status': AgentStatus.SUCCESS}

        # ✅ Парсинг JSON из текста
        tool_calls = _parse_tool_calls(last_message.content)

        if self.debug:
            print(f"\n[TOOL] Выполняю {len(tool_calls)} инструментов")

        if not tool_calls:
            return {'messages': state.messages, 'status': AgentStatus.SUCCESS}

        tool_results = []
        for tool_call in tool_calls:
            tool_name = tool_call.get('name')
            tool_args = tool_call.get('args', {})

            if self.debug:
                print(f"[TOOL] Вызываю: {tool_name}({tool_args})")

            try:
                tool_func = TOOLS_BY_NAME.get(tool_name)
                if tool_func:
                    result = tool_func.invoke(tool_args)
                    tool_results.append({
                        'tool': tool_name,
                        'result': result
                    })
                    if self.debug:
                        print(f"[TOOL] Результат: {str(result)[:300]}")
            except Exception as e:
                if self.debug:
                    print(f"[TOOL] Ошибка: {e}")
                tool_results.append({
                    'tool': tool_name,
                    'result': {'error': str(e)}
                })

        results_text = "\n\n📎 РЕЗУЛЬТАТЫ ИНСТРУМЕНТОВ:\n"
        for tr in tool_results:
            results_text += f"\n### {tr['tool']}:\n{str(tr['result'])[:2000]}\n"

        tool_message = ToolMessage(content=results_text, tool_call_id='')
        new_messages = list(state.messages) + [tool_message]

        return {
            'messages': new_messages,
            'status': AgentStatus.SUCCESS,
        }

    def _final_answer_node(self, state: AgentState) -> dict:
        """Финальный ответ — форматирует результаты для пользователя"""
        if self.debug:
            print("\n[LLM] Финальный ответ...")

        last_message = state.messages[-1] if state.messages else None

        # ✅ Если были результаты инструментов — форматируем их красиво
        if state.rag_context and any(c.get('type') == 'tool_results' for c in state.rag_context):
            if self.debug:
                print(f"[LLM] Форматируем результаты инструментов...")

            system_msg = SystemMessage(
                content="""Ты — AI-ассистент. Отформатируй результаты инструментов в понятный ответ на русском языке.

    ПРАВИЛА:
    1. НЕ показывай сырые JSON/словари
    2. Используй человеческий язык
    3. Выделяй важное жирным
    4. Будь кратким но информативным

    ПРИМЕР:
    Вместо: [{'path': '/file.cpp', 'chunks': 5}]
    Пиши: ✅ Найдено 1 файл: file.cpp (5 чанков)
    """
            )
            messages = [system_msg] + list(state.messages)

            response = self.llm.invoke(messages)

            return {
                'messages': [response],
                'result': response.content,
                'status': AgentStatus.SUCCESS,
            }

        # ✅ Если нет результатов инструментов — используем контент LLM
        if last_message and hasattr(last_message, 'content') and last_message.content:
            if self.debug:
                print(f"[LLM] Используем существующий контент")

            return {
                'messages': state.messages,
                'result': last_message.content,
                'status': AgentStatus.SUCCESS,
            }

        return {
            'messages': state.messages,
            'result': state.result or 'Готово',
            'status': AgentStatus.SUCCESS,
        }

    def _router(self, state: AgentState) -> str:
        """Роутинг"""
        last_message = state.messages[-1] if state.messages else None

        if not last_message or not last_message.content:
            return 'final_answer'

        # ✅ Парсинг для роутинга
        tool_calls = _parse_tool_calls(last_message.content)

        if self.debug:
            print(f"\n[ROUTER] tool_calls: {len(tool_calls)}")

        if tool_calls:
            return 'execute_tools'

        return 'final_answer'

    def _build_graph(self):
        workflow = StateGraph(AgentState)

        workflow.add_node('llm', self._llm_node)
        workflow.add_node('execute_tools', self._tool_executor_node)
        workflow.add_node('final_answer', self._final_answer_node)

        workflow.set_entry_point('llm')
        workflow.add_conditional_edges('llm', self._router, {
            'execute_tools': 'execute_tools',
            'final_answer': 'final_answer',
        })
        workflow.add_edge('execute_tools', 'final_answer')
        workflow.add_edge('final_answer', END)

        return workflow.compile()

    def invoke(self, messages=None, session_id='default', **kwargs):
        """Вызов агента"""
        if self.debug:
            print(f"\n{'=' * 60}")
            print(f"[INVOKE] Session: {session_id}")
            print(f"{'=' * 60}")

        session_ctx = self.session_store.get(session_id)
        if not session_ctx:
            session_ctx = SessionContext(session_id=session_id)

        stored_messages = []
        if session_ctx.messages:
            stored_messages = _dict_to_messages(session_ctx.messages)

        all_messages = stored_messages + (messages or [])

        initial_state = AgentState(
            messages=all_messages,
            rag_context=session_ctx.accumulated_context or [],
            token_usage=session_ctx.token_usage or {},
            fix_iterations=session_ctx.fix_iterations,
            **kwargs
        )

        result_dict = self.graph.invoke(initial_state)

        if isinstance(result_dict, dict):
            result = AgentState.model_validate(result_dict)
        else:
            result = result_dict

        if self.debug:
            print(f"\n[INVOKE] Результат тип: {type(result)}")
            print(f"[INVOKE] Messages: {len(result.messages) if result.messages else 0}")
            print(f"[INVOKE] Result: {result.result[:100] if result.result else 'None'}...")
            print(f"{'=' * 60}\n")

        messages_list = list(result.messages) if result.messages else []
        new_ctx = SessionContext(
            session_id=session_id,
            messages=_messages_to_dict(messages_list),
            accumulated_context=result.rag_context or [],
            token_usage=result.token_usage or {},
            fix_iterations=result.fix_iterations,
        )
        self.session_store.save(new_ctx)

        return result

    def clear_session(self, session_id='default'):
        self.session_store.delete(session_id)