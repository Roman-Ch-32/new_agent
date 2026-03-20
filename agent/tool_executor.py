"""Tool Executor — выполнение цепочки инструментов с retry."""

from __future__ import annotations

from dataclasses import dataclass, field
import inspect

from langchain_core.tools import BaseTool, StructuredTool

from agent.tools import AgentTools


@dataclass
class ToolExecutionResult:
    """Результат выполнения инструмента."""
    tool_name: str
    success: bool
    result: object
    error: str | None = None


@dataclass
class ExecutionPlan:
    """План выполнения от Planner."""
    actions: list[dict[str, object]] = field(default_factory=list)
    current_step: int = 0
    max_retries: int = 3
    retry_count: int = 0


@dataclass
class ToolExecutor:
    """Выполняет цепочку инструментов с обработкой ошибок."""
    tools: AgentTools
    debug: bool = True
    execution_history: list[ToolExecutionResult] = field(default_factory=list)

    def _get_raw_tool(self, tool_name: str) -> BaseTool | None:
        return type(self.tools).declared_tools().get(tool_name)

    def _bind_tool(self, tool_name: str) -> BaseTool | None:
        raw_tool = self._get_raw_tool(tool_name)
        if raw_tool is None:
            return None

        func = getattr(raw_tool, "func", None)
        if func is None:
            return None

        bound_method = func.__get__(self.tools, type(self.tools))
        description = raw_tool.description or inspect.getdoc(func) or tool_name

        return StructuredTool.from_function(func=bound_method, name=raw_tool.name, description=description)

    def _invoke_bound_tool(self, tool_name: str, parameters: dict[str, object]) -> object:
        tool_obj = self._bind_tool(tool_name)
        if tool_obj is None:
            raise ValueError(f"Инструмент не найден: {tool_name}")
        return tool_obj.invoke(parameters)

    def _precheck_action(self, action: dict[str, object]) -> str | None:
        tool_name = str(action.get("tool", ""))

        if tool_name != "write_file":
            return None

        try:
            branch_info = self._invoke_bound_tool("git_get_current_branch", {})
        except Exception as e:
            return str(e)

        if not isinstance(branch_info, dict):
            return "git_get_current_branch вернул неожиданный формат"

        if not branch_info.get("success", False):
            return str(branch_info.get("error", "Git инструменты недоступны; запись в production-режиме запрещена."))

        branch = branch_info.get("branch")
        protected = {"main", "master", "develop", "dev"}
        if isinstance(branch, str) and branch in protected:
            return f"Запись в protected branch запрещена: {branch}. Сначала создай feature/fix/docs ветку."

        return None

    def execute_action(self, action: dict[str, object]) -> ToolExecutionResult:
        """Выполняет одно действие."""
        tool_name = str(action.get("tool", ""))
        parameters_raw = action.get("parameters", {})
        parameters = parameters_raw if isinstance(parameters_raw, dict) else {}

        if self.debug:
            print(f"[EXECUTOR] Вызываю: {tool_name}({parameters})")

        precheck_error = self._precheck_action(action)
        if precheck_error:
            if self.debug:
                print(f"[EXECUTOR] Guard error: {precheck_error}")

            return ToolExecutionResult(tool_name=tool_name,
                                       success=False,
                                       result=None,
                                       error=precheck_error)

        try:
            result = self._invoke_bound_tool(tool_name, parameters)

            if self.debug:
                print(f"[EXECUTOR] Результат: {str(result)[:300]}")

            if isinstance(result, dict) and ("ошибка" in result or "error" in result):
                return ToolExecutionResult(tool_name=tool_name,
                                           success=False,
                                           result=result,
                                           error=str(result.get("ошибка", result.get("error", "Неизвестная ошибка"))))

            return ToolExecutionResult(tool_name=tool_name,
                                       success=True,
                                       result=result)

        except Exception as e:
            if self.debug:
                print(f"[EXECUTOR] Ошибка: {e}")

            return ToolExecutionResult(tool_name=tool_name,
                                       success=False,
                                       result=None,
                                       error=str(e))

    def execute_plan(self, plan: ExecutionPlan) -> dict[str, object]:
        """Выполняет весь план с retry логикой."""
        self.execution_history = []

        for i, action in enumerate(plan.actions):
            result = self.execute_action(action)
            self.execution_history.append(result)

            if not result.success:
                if self.debug:
                    print(f"[EXECUTOR] ❌ Ошибка на шаге {i + 1}: {result.error}")

                if plan.retry_count < plan.max_retries:
                    plan.retry_count += 1
                    if self.debug:
                        print(f"[EXECUTOR] 🔄 Попытка {plan.retry_count}/{plan.max_retries}")

                    return {"success": False,
                            "completed_steps": i,
                            "total_steps": len(plan.actions),
                            "error": result.error,
                            "failed_tool": result.tool_name,
                            "execution_history": self.execution_history,
                            "need_replan": True,
                            }

                if self.debug:
                    print("[EXECUTOR] ❌ Превышено количество попыток")

                return {"success": False,
                        "completed_steps": i,
                        "total_steps": len(plan.actions),
                        "error": result.error,
                        "failed_tool": result.tool_name,
                        "execution_history": self.execution_history,
                        "need_replan": False,
                        }

        if self.debug:
            print("[EXECUTOR] ✅ Все шаги выполнены успешно")

        return {"success": True,
                "completed_steps": len(plan.actions),
                "total_steps": len(plan.actions),
                "execution_history": self.execution_history,
                "need_replan": False,
                }

    def get_results_summary(self) -> str:
        """Возвращает краткую сводку результатов."""
        summary: list[str] = []

        for result in self.execution_history:
            if result.success:
                summary.append(f"✅ {result.tool_name}: успешно")
            else:
                summary.append(f"❌ {result.tool_name}: {result.error}")

        return "\n".join(summary)

    def get_results_for_responder(self) -> str:
        """Возвращает результаты в формате для Responder."""
        results_text = "РЕЗУЛЬТАТЫ ВЫПОЛНЕНИЯ:\n\n"

        for result in self.execution_history:
            results_text += f"### {result.tool_name}:\n"
            if result.success:
                results_text += "Статус: ✅ Успешно\n"
                results_text += f"Результат: {str(result.result)[:500]}\n"
            else:
                results_text += "Статус: ❌ Ошибка\n"
                results_text += f"Ошибка: {result.error}\n"
            results_text += "\n"

        return results_text