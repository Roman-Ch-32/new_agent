# piplines/tool_pipeline.py
"""Tool Pipeline — Выполнение инструментов по решению LLM"""

import re
import json
from typing import Any
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from agent.state import AgentState, AgentStatus
from piplines.abc_pipline import BasePipeline
from mcp.tools import MCPTools


class ToolPipeline(BasePipeline):
    """Пайплайн с выполнением инструментов по решению LLM"""

    def __init__(
            self,
            llm: ChatOpenAI,
            project_path: str,
            qdrant_url: str = 'http://localhost:6333'
    ):
        self.llm = llm
        self.mcp = MCPTools(project_path, qdrant_url)
        self.graph = self.build_graph()

    @property
    def name(self) -> str:
        return 'tool'

    @property
    def description(self) -> str:
        return 'Выполнение инструментов по решению LLM'

    @property
    def keywords(self) -> list[str]:
        return ['найди', 'покажи', 'прочитай', 'индексируй', 'поиск', 'класс', 'функция', 'файл', 'структур']

    def _parse_tool_call(self, text: str) -> list[dict]:
        """Извлекает tool_call из ответа LLM"""
        tool_calls = []

        # Ищем все <tool_call>...</tool_call>
        pattern = r'<tool_call>\s*(\{.*?\})\s*</tool_call>'
        matches = re.findall(pattern, text, re.DOTALL)

        for match in matches:
            try:
                tool_call = json.loads(match)
                tool_calls.append(tool_call)
            except json.JSONDecodeError:
                continue

        return tool_calls

    def _execute_tool_call(self, tool_call: dict) -> Any:
        """Выполняет один tool_call"""
        tool_name = tool_call.get('tool')
        parameters = tool_call.get('parameters', {})

        if not tool_name:
            return {'error': 'No tool name specified'}

        return self.mcp.execute_tool(tool_name, **parameters)

    def _llm_decide(self, state: AgentState) -> AgentState:
        """LLM решает какие инструменты вызвать"""
        query = state.get_last_user_message()
        if not query:
            return state

        # Добавляем описание инструментов в системное сообщение
        tool_descriptions = self.mcp.get_tool_descriptions()

        system_msg = SystemMessage(
            content=f"""{self.mcp.get_system_prompt()}

## ДОСТУПНЫЕ ИНСТРУМЕНТЫ:
{tool_descriptions}

Текущий запрос пользователя: {query}

Реши какие инструменты нужно вызвать. Верни <tool_call> с JSON или сразу <answer> если инструменты не нужны."""
        )

        messages_with_system = [system_msg] + list(state.messages)

        response = self.llm.invoke(messages_with_system)

        return AgentState(
            messages=[response],
            result=response.content,
            status=AgentStatus.SUCCESS,
            rag_context=state.rag_context,
            token_usage=state.token_usage,
            fix_iterations=state.fix_iterations,
        )

    def _execute_tools(self, state: AgentState) -> AgentState:
        """Выполняет инструменты которые решил LLM"""
        last_message = state.messages[-1].content if state.messages else ''

        tool_calls = self._parse_tool_call(last_message)

        if not tool_calls:
            # Инструменты не нужны — возвращаем как есть
            return state

        # Выполняем все tool_call
        tool_results = []
        for tool_call in tool_calls:
            result = self._execute_tool_call(tool_call)
            tool_results.append({
                'tool': tool_call.get('tool'),
                'result': result
            })

        # Форматируем результаты
        results_text = "\n\n## РЕЗУЛЬТАТЫ ИНСТРУМЕНТОВ:\n"
        for tr in tool_results:
            results_text += f"\n### {tr['tool']}:\n{str(tr['result'])[:1000]}\n"

        # Добавляем результаты к сообщениям для финального ответа LLM
        results_msg = HumanMessage(content=results_text)

        return AgentState(
            messages=list(state.messages) + [results_msg],
            result=results_text,
            status=AgentStatus.SUCCESS,
            rag_context=state.rag_context,
            token_usage=state.token_usage,
            fix_iterations=state.fix_iterations,
        )

    def _final_answer(self, state: AgentState) -> AgentState:
        """LLM генерирует финальный ответ на основе результатов инструментов"""
        system_msg = SystemMessage(
            content="Используй результаты инструментов для ответа пользователю. Верни ответ в формате <answer>...</answer>"
        )

        messages_with_system = [system_msg] + list(state.messages)

        response = self.llm.invoke(messages_with_system)

        return AgentState(
            messages=[response],
            result=response.content,
            status=AgentStatus.SUCCESS,
            rag_context=state.rag_context,
            token_usage=state.token_usage,
            fix_iterations=state.fix_iterations,
        )

    def _router(self, state: AgentState) -> str:
        """Роутинг между узлами"""
        last_message = state.messages[-1].content if state.messages else ''

        # Если уже есть результаты инструментов — идём на финальный ответ
        if 'РЕЗУЛЬТАТЫ ИНСТРУМЕНТОВ' in last_message:
            return 'final_answer'

        # Если есть tool_call — идём на выполнение
        if '<tool_call>' in last_message:
            return 'execute_tools'

        # Иначе сразу конец
        return 'end'

    def build_graph(self):
        """Строит граф пайплайна"""
        workflow = StateGraph(AgentState)

        workflow.add_node('llm_decide', self._llm_decide)
        workflow.add_node('execute_tools', self._execute_tools)
        workflow.add_node('final_answer', self._final_answer)

        workflow.set_entry_point('llm_decide')
        workflow.add_conditional_edges('llm_decide', self._router, {
            'execute_tools': 'execute_tools',
            'final_answer': 'final_answer',
            'end': END
        })
        workflow.add_edge('execute_tools', 'final_answer')
        workflow.add_edge('final_answer', END)

        return workflow.compile()

    def execute(self, state: AgentState) -> AgentState:
        """Выполняет граф пайплайна"""
        result = self.graph.invoke(state)
        return AgentState.model_validate(result)