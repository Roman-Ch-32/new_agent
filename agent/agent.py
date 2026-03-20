# agent/agent.py
"""AI Agent — Planner/Executor архитектура."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import json
import re

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, StateGraph
from langchain_openai import ChatOpenAI

from agent.config import config
from agent.state import AgentState, AgentStatus
from agent.system_prompt import get_planner_prompt, get_responder_prompt
from agent.tool_executor import ExecutionPlan, ToolExecutor
from agent.tools import AgentTools
from memory.session_store import SessionContext, SessionStore


@dataclass
class Agent:
    """AI Agent с разделением Planner/Executor."""
    qdrant_url: str = field(default_factory=lambda: getattr(config.qdrant, "url", "http://localhost:6333"))
    project_path: str = field(default_factory=lambda: getattr(config.project, "path", "/tmp"))
    debug: bool = field(default_factory=lambda: getattr(config, "debug", True))
    max_iterations: int = 5

    llm: ChatOpenAI = field(init=False)
    session_store: SessionStore = field(init=False)
    graph: Any = field(init=False)
    tools: AgentTools = field(init=False)
    executor: ToolExecutor = field(init=False)

    def __post_init__(self) -> None:
        if self.debug:
            print(f"\n{'=' * 60}")
            print("[АГЕНТ] Инициализация...")
            print(f"[АГЕНТ] Путь к проекту: {self.project_path}")
            print(f"[АГЕНТ] Qdrant URL: {self.qdrant_url}")

        self.llm = self._create_llm()
        self.tools = AgentTools(project_path=self.project_path, qdrant_url=self.qdrant_url)
        self.executor = ToolExecutor(tools=self.tools, debug=self.debug)
        self.session_store = SessionStore(self.qdrant_url)
        self.graph = self._build_graph()

        if self.debug:
            print(f"[АГЕНТ] Готов! Инструментов: {len(self.tools.get_tool_names())}")
            print(f"[АГЕНТ] Инструменты: {self.tools.get_tool_names()}")
            print(f"{'=' * 60}\n")

    def _create_llm(self) -> ChatOpenAI:
        return ChatOpenAI(model=getattr(config.llm, "model", "qwen-ue"),
                          base_url=getattr(config.llm, "base_url", "http://localhost:8080/v1"),
                          temperature=getattr(config.llm, "temperature", 0.1),
                          api_key="sk-no-key-required",
                          top_p=getattr(config.llm, "top_p", 0.9),
                          max_tokens=getattr(config.llm, "num_predict", 2048),
                          )

    def _get_tools_list_for_prompt(self) -> str:
        return self.tools.render_tools_for_prompt()

    def _extract_json_object(self, text: str) -> str | None:
        if not text or not isinstance(text, str):
            return None

        start = text.find("{")
        if start == -1:
            return None

        depth = 0
        in_string = False
        escape = False

        for i, ch in enumerate(text[start:], start=start):
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                continue

            if ch == '"':
                in_string = True
                continue

            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start:i + 1]

        return None

    def _default_plan(self) -> dict[str, Any]:
        return {"actions": [],
                "done": True,
                "final_message": "",
                }

    def _repair_planner_json(self, raw_text: str) -> dict[str, Any]:
        repair_prompt = (
            "Исправь ответ planner и верни только валидный JSON.\n"
            "Нужна схема:\n"
            '{"actions":[{"tool":"tool_name","parameters":{}}],"done":false,"final_message":""}\n'
            "Никакого markdown и пояснений.\n\n"
            f"Сломанный ответ:\n{raw_text}"
        )

        response = self.llm.invoke([HumanMessage(content=repair_prompt)])
        return self._parse_planner_json(response.content, allow_repair=False)

    def _parse_planner_json(self, text: str, allow_repair: bool = True) -> dict[str, Any]:
        if not text or not isinstance(text, str):
            return self._default_plan()

        candidates: list[str] = []

        stripped = text.strip()
        if stripped.startswith("{") and stripped.endswith("}"):
            candidates.append(stripped)

        fenced = re.findall(r"```(?:json)?\s*(\{.*?})\s*```", text, re.DOTALL)
        candidates.extend(fenced)

        extracted = self._extract_json_object(text)
        if extracted:
            candidates.append(extracted)

        for candidate in candidates:
            try:
                parsed = json.loads(candidate)
            except Exception:
                continue

            if not isinstance(parsed, dict):
                continue

            actions = parsed.get("actions", [])
            if not isinstance(actions, list):
                continue

            normalized: list[dict[str, Any]] = []
            for action in actions:
                if not isinstance(action, dict):
                    continue

                tool_name = action.get("tool")
                parameters = action.get("parameters", {})

                if not isinstance(tool_name, str) or not tool_name.strip():
                    continue
                if not isinstance(parameters, dict):
                    parameters = {}

                normalized.append({"tool": tool_name,
                                   "parameters": parameters,
                                   })

            return {"actions": normalized,
                    "done": bool(parsed.get("done", len(normalized) == 0)),
                    "final_message": str(parsed.get("final_message", "")),
                    }

        if allow_repair:
            try:
                return self._repair_planner_json(text)
            except Exception:
                pass

        return self._default_plan()

    def _planner_node(self, state: AgentState) -> dict[str, Any]:
        query = state.get_last_user_message()

        if not query:
            return {"messages": state.messages,
                    "status": AgentStatus.SUCCESS,
                    "result": "",
                    }

        if self.debug:
            print(f"\n[PLANNER] Запрос: {str(query)[:100]}...")

        system_prompt = get_planner_prompt(self.project_path, self._get_tools_list_for_prompt())
        response = self.llm.invoke([SystemMessage(content=system_prompt), *list(state.messages)])
        plan = self._parse_planner_json(response.content)

        if self.debug:
            print(f"[PLANNER] План: {plan}")

        planner_message = AIMessage(content=json.dumps(plan, ensure_ascii=False))
        new_messages = list(state.messages) + [planner_message]

        return {"messages": new_messages,
                "plan": plan,
                "status": AgentStatus.PLANNING,
                "fix_iterations": state.fix_iterations + 1,
                }

    def _executor_node(self, state: AgentState) -> dict[str, Any]:
        plan_dict = state.plan or {}
        actions = plan_dict.get("actions", [])

        if not isinstance(actions, list) or not actions:
            if self.debug:
                print("[EXECUTOR] Нет действий для выполнения")

            return {"messages": state.messages,
                    "execution_result": {"success": True, "need_replan": False},
                    "tool_results": "",
                    "status": AgentStatus.EXECUTING,
                    }

        execution_plan = ExecutionPlan(actions=actions,
                                       max_retries=1,
                                       retry_count=0,
                                       )

        if self.debug:
            print(f"\n[EXECUTOR] Выполняю {len(actions)} шаг(ов)")

        execution_result = self.executor.execute_plan(execution_plan)
        results_text = self.executor.get_results_for_responder()
        tool_message = ToolMessage(content=results_text, tool_call_id="executor")
        new_messages = list(state.messages) + [tool_message]

        return {"messages": new_messages,
                "tool_results": results_text,
                "execution_result": execution_result,
                "status": AgentStatus.RETRY if execution_result.get("need_replan") else AgentStatus.EXECUTING,
                }

    def _responder_node(self, state: AgentState) -> dict[str, Any]:
        if self.debug:
            print("\n[RESPONDER] Генерирую ответ...")

        plan_dict = state.plan or {}
        tool_results = state.tool_results or "Нет результатов выполнения."
        rag_context = ""

        if state.rag_context:
            rag_context = "\n".join(str(item) for item in state.rag_context[:3])

        final_message = ""
        if isinstance(plan_dict, dict):
            final_message = str(plan_dict.get("final_message", ""))

        system_prompt = get_responder_prompt(tool_results, rag_context)
        extra_context = ""

        if final_message:
            extra_context += f"\nПодсказка planner:\n{final_message}\n"

        if state.execution_result:
            extra_context += f"\nДетали выполнения:\n{state.execution_result}\n"

        response = self.llm.invoke([SystemMessage(content=system_prompt),
                                    *list(state.messages),
                                    HumanMessage(content=extra_context or "Сформируй финальный ответ пользователю на русском языке."),
                                    ])

        if self.debug:
            print(f"[RESPONDER] Ответ: {response.content[:200]}...")

        return {"messages": list(state.messages) + [AIMessage(content=response.content)],
                "result": response.content,
                "status": AgentStatus.SUCCESS,
                }

    def _router_after_planner(self, state: AgentState) -> str:
        if self.debug:
            print(f"\n[ROUTER:PLANNER] Iterations: {state.fix_iterations}")

        if state.is_max_iterations_reached(self.max_iterations):
            if self.debug:
                print("[ROUTER:PLANNER] ❌ Превышено количество итераций")
            return "responder"

        plan_dict = state.plan or {}
        actions = plan_dict.get("actions", [])

        if isinstance(actions, list) and actions:
            if self.debug:
                print("[ROUTER:PLANNER] ⚙️ Переход к Executor")
            return "executor"

        if self.debug:
            print("[ROUTER:PLANNER] ✅ Переход к Responder")
        return "responder"

    def _router_after_executor(self, state: AgentState) -> str:
        execution_result = state.execution_result or {}

        if self.debug:
            print(f"\n[ROUTER:EXECUTOR] Результат выполнения: {execution_result}")

        if execution_result.get("need_replan"):
            if state.is_max_iterations_reached(self.max_iterations):
                if self.debug:
                    print("[ROUTER:EXECUTOR] ❌ Превышено количество итераций")
                return "responder"

            if self.debug:
                print("[ROUTER:EXECUTOR] 🔄 Возврат к Planner")
            return "planner"

        if self.debug:
            print("[ROUTER:EXECUTOR] ✅ Переход к Responder")
        return "responder"

    def _build_graph(self):
        workflow = StateGraph(AgentState)

        workflow.add_node("planner", self._planner_node)
        workflow.add_node("executor", self._executor_node)
        workflow.add_node("responder", self._responder_node)

        workflow.set_entry_point("planner")
        workflow.add_conditional_edges("planner",
                                       self._router_after_planner,
                                       {"executor": "executor",
                                        "responder": "responder",
                                        },
                                       )
        workflow.add_conditional_edges("executor",
                                       self._router_after_executor,
                                       {"planner": "planner",
                                        "responder": "responder",
                                        },
                                       )
        workflow.add_edge("responder", END)

        return workflow.compile()

    def _get_message_content(self, message: BaseMessage) -> str:
        if not hasattr(message, "content"):
            return ""

        content = message.content
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, dict):
                    parts.append(str(item.get("text", item.get("content", ""))))
                else:
                    parts.append(str(item))
            return "\n".join(parts)

        return str(content)

    def _messages_to_dict(self, messages: list[BaseMessage]) -> list[dict[str, str]]:
        result: list[dict[str, str]] = []

        for msg in messages:
            result.append({"type": type(msg).__name__,
                           "content": self._get_message_content(msg),
                           })

        return result

    def _dict_to_messages(self, data: list[dict[str, str]]) -> list[BaseMessage]:
        result: list[BaseMessage] = []

        for item in data:
            msg_type = item.get("type", "HumanMessage")
            content = item.get("content", "")

            if msg_type == "HumanMessage":
                result.append(HumanMessage(content=content))
            elif msg_type == "AIMessage":
                result.append(AIMessage(content=content))
            elif msg_type == "ToolMessage":
                result.append(ToolMessage(content=content, tool_call_id="restored"))
            elif msg_type == "SystemMessage":
                result.append(SystemMessage(content=content))

        return result

    def invoke(self, messages: list[BaseMessage] | None = None, session_id: str = "default", **kwargs) -> AgentState:
        if self.debug:
            print(f"\n{'=' * 60}")
            print(f"[ВЫЗОВ] Сессия: {session_id}")
            print(f"{'=' * 60}")

        session_ctx = self.session_store.get(session_id)
        if not session_ctx:
            session_ctx = SessionContext(session_id=session_id)

        stored_messages = self._dict_to_messages(session_ctx.messages) if session_ctx.messages else []
        all_messages = stored_messages + (messages or [])

        initial_state = AgentState(messages=all_messages,
                                   rag_context=session_ctx.accumulated_context or [],
                                   fix_iterations=session_ctx.fix_iterations,
                                   **kwargs,
                                   )

        result_dict = self.graph.invoke(initial_state)
        result = AgentState.model_validate(result_dict) if isinstance(result_dict, dict) else result_dict

        if self.debug:
            print(f"\n[ВЫЗОВ] Результат тип: {type(result)}")
            print(f"[ВЫЗОВ] Сообщений: {len(result.messages) if result.messages else 0}")
            print(f"[ВЫЗОВ] Результат: {result.result[:100] if result.result else 'None'}...")
            print(f"{'=' * 60}\n")

        messages_list = list(result.messages) if result.messages else []
        new_ctx = SessionContext(session_id=session_id,
                                 messages=self._messages_to_dict(messages_list),
                                 accumulated_context=result.rag_context or [],
                                 token_usage={},
                                 fix_iterations=result.fix_iterations,
                                 )
        self.session_store.save(new_ctx)

        return result

    def clear_session(self, session_id: str = "default") -> None:
        self.session_store.delete(session_id)