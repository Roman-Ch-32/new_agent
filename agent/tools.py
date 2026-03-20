"""Agent tools registry and implementations."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import inspect

from langchain_core.tools import BaseTool, tool

from agent.config import config
from mcp.code_analyzer import CodeAnalyzer
from mcp.file_system import FileSystemTools
from mcp.indexer import FileIndexer
from mcp.internet_search import InternetSearch
from mcp.git_tools import GitTools


@dataclass
class AgentTools:
    project_path: str = field(default_factory=lambda: config.project.path)
    qdrant_url: str = field(default_factory=lambda: config.qdrant.url)
    project_root: Path = field(init=False)
    _fs: FileSystemTools = field(init=False, repr=False)
    _analyzer: CodeAnalyzer = field(init=False, repr=False)
    _indexer: FileIndexer = field(init=False, repr=False)
    _git_backend: GitTools | None = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        self.project_root = Path(self.project_path).expanduser().resolve()
        self.project_path = str(self.project_root)
        self._fs = FileSystemTools(self.project_path)
        self._analyzer = CodeAnalyzer(self.project_path)
        self._indexer = FileIndexer(qdrant_url=self.qdrant_url)
        self._git_backend = GitTools(self.project_path)

    @classmethod
    def declared_tools(cls) -> dict[str, BaseTool]:
        tools: dict[str, BaseTool] = {}

        for name, value in cls.__dict__.items():
            if isinstance(value, BaseTool):
                tools[name] = value

        return tools

    @classmethod
    def get_tool_names(cls) -> list[str]:
        return list(cls.declared_tools().keys())

    @classmethod
    def get_tool_specs(cls) -> list[dict[str, str]]:
        specs: list[dict[str, str]] = []

        for name, raw_tool in cls.declared_tools().items():
            func = getattr(raw_tool, "func", None)
            if func is None:
                continue

            specs.append({"name": name,
                          "description": raw_tool.description or inspect.getdoc(func) or "",
                          })

        return specs

    @classmethod
    def render_tools_for_prompt(cls) -> str:
        lines: list[str] = []

        for spec in cls.get_tool_specs():
            lines.append(f'- {spec["name"]} — {spec["description"]}')

        return "\n".join(lines)

    @tool
    def index_directory(self, directory: str | None = None, project_name: str = "ue_project", recursive: bool = True,
                        limit: int | None = None) -> dict:
        """Индексирует директорию проекта в Qdrant."""
        target = directory or self.project_path
        return self._indexer.index_directory(target, project_name, recursive, limit)

    @tool
    def index_file(self, file_path: str, project_name: str = "ue_project") -> dict:
        """Индексирует файл в Qdrant."""
        return self._indexer.index_file(file_path, project_name)

    @tool
    def search_indexed(self, query: str, limit: int = 10) -> list:
        """Поиск по проиндексированным файлам в Qdrant."""
        return self._indexer.search_indexed(query, limit)

    @tool
    def get_indexed_files(self, project_name: str = "ue_project") -> list:
        """Список проиндексированных файлов."""
        return self._indexer.get_indexed_files(project_name)

    @tool
    def get_project_structure(self, max_depth: int = 3) -> dict:
        """Получить структуру проекта."""
        return self._fs.get_project_structure(max_depth)

    @tool
    def find_class(self, class_name: str) -> list:
        """Найти класс в проекте."""
        return self._analyzer.find_class(class_name)

    @tool
    def find_function(self, function_name: str) -> list:
        """Найти функцию в проекте."""
        return self._analyzer.find_function(function_name)

    @tool
    def read_file(self, file_path: str) -> str:
        """Читать файл из проекта."""
        return self._fs.read_file(file_path)

    @tool
    def write_file(self, file_path: str, content: str, overwrite: bool = False) -> dict:
        """Записать содержимое в файл."""
        return self._fs.write_file(file_path=file_path, content=content, overwrite=overwrite)

    @tool
    def git_status(self) -> dict:
        """Получить статус Git репозитория."""
        if not self._git_backend:
            return {"success": False, "error": "Git инструменты недоступны"}
        return self._git_backend.get_status()

    @tool
    def git_get_current_branch(self) -> dict:
        """Получить текущую Git ветку."""
        if not self._git_backend:
            return {"success": False, "error": "Git инструменты недоступны"}

        try:
            return {"success": True, "branch": self._git_backend.get_current_branch()}
        except Exception as e:
            return {"success": False, "error": f"Не удалось определить ветку: {e}"}

    @tool
    def git_create_branch(self, branch_name: str, from_branch: str | None = None) -> dict:
        """Создать новую Git ветку."""
        if not self._git_backend:
            return {"success": False, "error": "Git инструменты недоступны"}
        return self._git_backend.create_branch(branch_name, from_branch)

    @tool
    def git_commit(self, message: str, files: list[str] | None = None) -> dict:
        """Сделать Git коммит."""
        if not self._git_backend:
            return {"success": False, "error": "Git инструменты недоступны"}
        return self._git_backend.commit(message, files)

    @tool
    def git_push(self, branch: str | None = None, force: bool = False) -> dict:
        """Пуш в удалённый Git репозиторий."""
        if not self._git_backend:
            return {"success": False, "error": "Git инструменты недоступны"}
        return self._git_backend.push(branch, force)

    @tool
    def git_pull(self, branch: str | None = None) -> dict:
        """Пулл из удалённого Git репозитория."""
        if not self._git_backend:
            return {"success": False, "error": "Git инструменты недоступны"}
        return self._git_backend.pull(branch)

    @tool
    def git_log(self, limit: int = 10) -> list:
        """История Git коммитов."""
        if not self._git_backend:
            return []
        return self._git_backend.get_log(limit)

    @tool
    def git_diff(self, branch1: str | None = None, branch2: str | None = None) -> str:
        """Разница между Git ветками."""
        if not self._git_backend:
            return ""
        result = self._git_backend.diff(branch1, branch2)
        return result.get("diff", "") if isinstance(result, dict) else ""

    @tool
    def git_checkout(self, branch: str) -> dict:
        """Переключиться на Git ветку."""
        if not self._git_backend:
            return {"success": False, "error": "Git инструменты недоступны"}
        return self._git_backend.checkout(branch)

    @tool
    def git_merge(self, branch: str) -> dict:
        """Слияние Git веток."""
        if not self._git_backend:
            return {"success": False, "error": "Git инструменты недоступны"}
        return self._git_backend.merge(branch)

    @tool
    def git_stash(self, message: str = "WIP") -> dict:
        """Сохранить изменения в Git stash."""
        if not self._git_backend:
            return {"success": False, "error": "Git инструменты недоступны"}
        return self._git_backend.stash(message)

    @tool
    def git_stash_pop(self) -> dict:
        """Восстановить из Git stash."""
        if not self._git_backend:
            return {"success": False, "error": "Git инструменты недоступны"}
        return self._git_backend.stash_pop()

    @tool
    def git_get_branches(self, remote: bool = False) -> list:
        """Список Git веток."""
        if not self._git_backend:
            return []
        return self._git_backend.get_branches(remote)

    @tool
    def search_duckduckgo(self, query: str, num_results: int = 5) -> list:
        """Поиск в интернете."""
        return InternetSearch().search_duckduckgo(query, num_results)
