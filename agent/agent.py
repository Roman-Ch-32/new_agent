"""Production-oriented planner/executor/verification agent."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List
import json

from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, SystemMessage, ToolMessage
from langgraph.graph import END, StateGraph
from langchain_openai import ChatOpenAI
from langchain_core.tools import BaseTool

from agent.config import config
from agent.state import AgentState, AgentStatus
from agent.system_prompt import get_planner_prompt, get_responder_prompt
from agent.tool_executor import ExecutionPlan, ToolExecutor
from agent.tools import AgentTools
from memory.session_store import SessionContext, SessionStore


@dataclass
class Agent:
    qdrant_url: str = field(default_factory=lambda: getattr(config.qdrant, 'url', 'http://localhost:6333'))
    project_path: str = field(default_factory=lambda: getattr(config.project, 'path', '/tmp'))
    debug: bool = field(default_factory=lambda: getattr(config, 'debug', True))
    max_iterations: int = 6
    max_actions_per_plan: int = 3
    llm: ChatOpenAI = field(init=False)
    session_store: SessionStore = field(init=False)
    graph: Any = field(init=False)
    tools: AgentTools = field(init=False)
    tools_dict: Dict[str, BaseTool] = field(init=False)
    tool_executor: ToolExecutor = field(init=False)

    def __post_init__(self) -> None:
        self.llm = self._create_llm()
        self.tools = AgentTools()
        self.tools_dict = self.tools.get_tools_by_name()
        self.tool_executor = ToolExecutor(self.tools_dict, debug=self.debug)
        self.session_store = SessionStore(self.qdrant_url)
        self.graph = self._build_graph()

    def _create_llm(self) -> ChatOpenAI:
        return ChatOpenAI(
            model=getattr(config.llm, 'model', 'qwen-ue'),
            base_url=getattr(config.llm, 'base_url', 'http://localhost:8080/v1'),
            temperature=getattr(config.llm, 'temperature', 0.03),
            api_key='sk-no-key-required',
            top_p=getattr(config.llm, 'top_p', 0.9),
            max_tokens=getattr(config.llm, 'num_predict', 2048),
        )

    def _append_trace(self, state: AgentState, *events: Dict[str, Any]) -> List[Dict[str, Any]]:
        trace = list(state.trace_events or [])
        trace.extend(events)
        return trace

    def _get_tools_list_for_prompt(self) -> str:
        items = []
        for name, tool in self.tools_dict.items():
            desc = (tool.description or 'Нет описания').splitlines()[0]
            items.append(f'- {name}: {desc}')
        return '\n'.join(items)

    def _candidate_json_strings(self, text: str) -> List[str]:
        candidates: List[str] = []
        stripped = text.strip()
        if stripped:
            candidates.append(stripped)

        if '```' in text:
            for block in text.split('```'):
                block = block.strip()
                if not block:
                    continue
                if block.lower().startswith('json'):
                    block = block[4:].strip()
                candidates.append(block)

        start = text.find('{')
        while start != -1:
            depth = 0
            for idx in range(start, len(text)):
                ch = text[idx]
                if ch == '{':
                    depth += 1
                elif ch == '}':
                    depth -= 1
                    if depth == 0:
                        candidates.append(text[start:idx + 1])
                        break
            start = text.find('{', start + 1)

        uniq: List[str] = []
        seen = set()
        for c in candidates:
            if c not in seen:
                uniq.append(c)
                seen.add(c)
        return uniq

    def _normalize_plan(self, parsed: Dict[str, Any]) -> Dict[str, Any]:
        actions = parsed.get('actions', [])
        normalized_actions = []
        if isinstance(actions, list):
            for action in actions[: self.max_actions_per_plan]:
                if not isinstance(action, dict):
                    continue
                tool = action.get('tool')
                params = action.get('parameters', {})
                if isinstance(tool, str) and tool in self.tools_dict and isinstance(params, dict):
                    normalized_actions.append({'tool': tool, 'parameters': params})

        done = bool(parsed.get('done', False))
        final_message = str(parsed.get('final_message', '') or '')
        if normalized_actions:
            done = False
        return {'actions': normalized_actions, 'done': done, 'final_message': final_message}

    def _repair_planner_json(self, raw_text: str) -> Dict[str, Any]:
        repair_prompt = (
            'Исправь ответ planner и верни только валидный JSON без пояснений. '
            'Схема: {"actions": [{"tool": "name", "parameters": {}}], "done": false, "final_message": ""}. '
            f'Допустимые инструменты: {", ".join(self.tools.get_tool_names())}.\n\n'
            f'Сломанный ответ:\n{raw_text}'
        )
        repaired = self.llm.invoke([HumanMessage(content=repair_prompt)])
        for candidate in self._candidate_json_strings(str(repaired.content)):
            try:
                return self._normalize_plan(json.loads(candidate))
            except Exception:
                continue
        return {'actions': [], 'done': True, 'final_message': 'Planner не вернул корректный JSON-план.'}

    def _parse_planner_json(self, text: str) -> Dict[str, Any]:
        if not text or not isinstance(text, str):
            return {'actions': [], 'done': True, 'final_message': 'Пустой ответ planner.'}

        for candidate in self._candidate_json_strings(text):
            try:
                return self._normalize_plan(json.loads(candidate))
            except Exception:
                continue
        return self._repair_planner_json(text)

    def _planner_node(self, state: AgentState) -> Dict[str, Any]:
        query = state.get_last_user_message() or state.current_task
        if not query:
            return {'status': AgentStatus.SUCCESS, 'result': '', 'trace_events': state.trace_events}

        system_prompt = get_planner_prompt(
            project_path=self.project_path,
            tools_list=self._get_tools_list_for_prompt(),
            max_actions=self.max_actions_per_plan,
        )

        response = self.llm.invoke([SystemMessage(content=system_prompt)] + list(state.messages))
        plan = self._parse_planner_json(str(response.content))
        plan_json = json.dumps(plan, ensure_ascii=False)

        trace_events = self._append_trace(
            state,
            {'type': 'plan', 'iteration': state.fix_iterations + 1, 'plan': plan},
        )

        return {
            'messages': list(state.messages) + [AIMessage(content=plan_json)],
            'plan': plan,
            'status': AgentStatus.PLANNING,
            'fix_iterations': state.fix_iterations + 1,
            'current_task': query,
            'trace_events': trace_events,
        }

    def _planner_router(self, state: AgentState) -> str:
        if state.is_max_iterations_reached(self.max_iterations):
            return 'responder'
        if state.plan.get('actions'):
            return 'executor'
        return 'responder'

    def _executor_node(self, state: AgentState) -> Dict[str, Any]:
        actions = state.plan.get('actions', []) if state.plan else []
        trace = list(state.trace_events or [])
        for action in actions:
            trace.append({'type': 'tool_call', 'tool': action.get('tool'), 'parameters': action.get('parameters', {})})

        execution = self.tool_executor.execute_plan(ExecutionPlan(actions=actions, max_retries=1))
        results_text = self.tool_executor.get_results_for_responder()

        for item in execution.get('execution_history', []):
            trace.append({
                'type': 'tool_result',
                'tool': item.tool_name,
                'success': item.success,
                'result': str(item.result)[:1500] if item.result is not None else '',
                'error': item.error,
            })

        return {
            'messages': list(state.messages) + [ToolMessage(content=results_text, tool_call_id='execution_plan')],
            'tool_results': results_text,
            'execution_result': execution,
            'status': AgentStatus.EXECUTING if execution.get('success') else AgentStatus.RETRY,
            'trace_events': trace,
        }

    def _verification_node(self, state: AgentState) -> Dict[str, Any]:
        execution = state.execution_result or {}
        success = bool(execution.get('success'))
        summary_parts = []
        diff_text = ''
        diff_stat = ''
        git_info: Dict[str, Any] = {}
        trace = list(state.trace_events or [])

        if success:
            summary_parts.append('Исполнение плана завершилось без ошибок инструмента.')
        else:
            summary_parts.append(f'Исполнение завершилось ошибкой: {execution.get("error", "неизвестно")}.')

        if self.tools.git:
            try:
                status = self.tools.git.get_status()
                git_info['status'] = status
                summary_parts.append(f"Текущая ветка: {status.get('branch', 'unknown')}. Изменённых файлов: {len(status.get('changed_files', [])) + len(status.get('untracked_files', []))}.")

                branches = set(self.tools.git.get_branches(False))
                current_branch = status.get('branch')
                base_branch = 'main' if 'main' in branches else 'master' if 'master' in branches else None
                if current_branch and base_branch and current_branch != base_branch:
                    diff_payload = self.tools.git.diff(current_branch, base_branch)
                    if isinstance(diff_payload, dict) and diff_payload.get('success'):
                        diff_text = diff_payload.get('diff', '')
                        diff_stat = diff_payload.get('stat', '')
                        if diff_text or diff_stat:
                            summary_parts.append(f'Есть diff относительно {base_branch}.')
                            trace.append({
                                'type': 'diff',
                                'branch': current_branch,
                                'base_branch': base_branch,
                                'stat': diff_stat[:2000],
                                'diff': diff_text[:4000],
                            })
            except Exception as e:
                summary_parts.append(f'Git verification не удалась: {e}')

        verification_result = {
            'success': success,
            'need_replan': not success and not state.is_max_iterations_reached(self.max_iterations),
            'continue_planning': success and bool(state.plan.get('actions')) and not state.is_max_iterations_reached(self.max_iterations),
            'summary': ' '.join(summary_parts),
            'git': git_info,
            'diff': diff_text[:4000],
            'diff_stat': diff_stat[:2000],
        }

        trace.append({'type': 'verification', 'verification': verification_result})

        return {
            'verification_result': verification_result,
            'status': AgentStatus.VERIFYING,
            'trace_events': trace,
            'plan': {},
        }

    def _verification_router(self, state: AgentState) -> str:
        verification = state.verification_result or {}
        if state.is_max_iterations_reached(self.max_iterations):
            return 'responder'
        if verification.get('need_replan'):
            return 'planner'
        if verification.get('continue_planning'):
            return 'planner'
        return 'responder'

    def _responder_node(self, state: AgentState) -> Dict[str, Any]:
        rag_context = '\n'.join(str(c) for c in (state.rag_context or [])[:3])
        verification_summary = json.dumps(state.verification_result or {}, ensure_ascii=False, indent=2)
        plan_json = json.dumps(state.plan or {}, ensure_ascii=False, indent=2)

        final_message = (state.plan or {}).get('final_message', '')
        if final_message and not state.tool_results:
            result_text = final_message
        else:
            prompt = get_responder_prompt(
                plan_json=plan_json,
                tool_results=state.tool_results or 'Нет результатов инструментов.',
                verification_summary=verification_summary,
                rag_context=rag_context,
            )
            response = self.llm.invoke([SystemMessage(content=prompt)] + list(state.messages))
            result_text = str(response.content)

        return {
            'messages': list(state.messages),
            'result': result_text,
            'status': AgentStatus.SUCCESS,
        }

    def _build_graph(self):
        workflow = StateGraph(AgentState)
        workflow.add_node('planner', self._planner_node)
        workflow.add_node('executor', self._executor_node)
        workflow.add_node('verification', self._verification_node)
        workflow.add_node('responder', self._responder_node)

        workflow.set_entry_point('planner')
        workflow.add_conditional_edges('planner', self._planner_router, {'executor': 'executor', 'responder': 'responder'})
        workflow.add_edge('executor', 'verification')
        workflow.add_conditional_edges('verification', self._verification_router, {'planner': 'planner', 'responder': 'responder'})
        workflow.add_edge('responder', END)
        return workflow.compile()

    def _messages_to_dict(self, messages: List[BaseMessage]) -> List[dict]:
        result = []
        for msg in messages:
            result.append({'type': type(msg).__name__, 'content': str(msg.content)})
        return result

    def _dict_to_messages(self, data: List[dict]) -> List[BaseMessage]:
        result: List[BaseMessage] = []
        for item in data:
            msg_type = item.get('type', 'HumanMessage')
            content = item.get('content', '')
            if msg_type == 'HumanMessage':
                result.append(HumanMessage(content=content))
            elif msg_type == 'AIMessage':
                result.append(AIMessage(content=content))
            elif msg_type == 'ToolMessage':
                result.append(ToolMessage(content=content, tool_call_id='restored'))
            elif msg_type == 'SystemMessage':
                result.append(SystemMessage(content=content))
        return result

    def invoke(self, messages=None, session_id: str = 'default', **kwargs):
        session_ctx = self.session_store.get(session_id) or SessionContext(session_id=session_id)
        stored_messages = self._dict_to_messages(session_ctx.messages) if session_ctx.messages else []
        all_messages = stored_messages + (messages or [])

        initial_state = AgentState(
            messages=all_messages,
            rag_context=session_ctx.accumulated_context or [],
            token_usage=session_ctx.token_usage or {},
            fix_iterations=0,
            plan={},
            tool_results='',
            execution_result={},
            verification_result={},
            trace_events=[],
            status=AgentStatus.IDLE,
            result='',
            ue_project_path=self.project_path,
        )

        result_dict = self.graph.invoke(initial_state)
        result = AgentState.model_validate(result_dict) if isinstance(result_dict, dict) else result_dict

        new_ctx = SessionContext(
            session_id=session_id,
            messages=self._messages_to_dict(list(result.messages) if result.messages else []),
            accumulated_context=result.rag_context or [],
            token_usage=result.token_usage or {},
            fix_iterations=result.fix_iterations,
        )
        self.session_store.save(new_ctx)
        return result

    def clear_session(self, session_id: str = 'default'):
        self.session_store.delete(session_id)
