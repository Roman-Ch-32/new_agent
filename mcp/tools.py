# mcp/tools.py
"""MCP Tools — Реестр инструментов для LLM"""

from typing import Dict, Any, List, Callable
from mcp.file_system import FileSystemTools
from mcp.code_analyzer import CodeAnalyzer
from mcp.internet_search import InternetSearch
from mcp.indexer import FileIndexer


class MCPTools:
    """Реестр всех доступных инструментов для LLM"""

    def __init__(
            self,
            project_path: str,
            qdrant_url: str = 'http://localhost:6333'
    ):
        self.project_path = project_path
        self.qdrant_url = qdrant_url

        # Инициализация инструментов
        self.file_system = FileSystemTools(project_path)
        self.code_analyzer = CodeAnalyzer(project_path)
        self.internet_search = InternetSearch()
        self.indexer = FileIndexer(qdrant_url)

        # Реестр инструментов с описанием для LLM
        self.tools: Dict[str, Dict[str, Any]] = {
            # File System
            'read_file': {
                'description': 'Читает файл из проекта. Используй для анализа конкретного файла.',
                'function': self.file_system.read_file,
                'parameters': {
                    'file_path': 'str (required) — Путь к файлу относительно проекта',
                    'lines': 'tuple (optional) — (start, end) для чтения диапазона строк'
                },
                'examples': [
                    'read_file(file_path="Source/MyProject/MyCharacter.h")',
                    'read_file(file_path="Config/DefaultEngine.ini", lines=(1, 50))'
                ]
            },

            'list_files': {
                'description': 'Список файлов в директории. Используй для исследования структуры.',
                'function': self.file_system.list_files,
                'parameters': {
                    'directory': 'str (optional) — Путь к директории',
                    'recursive': 'bool (optional) — Рекурсивный поиск',
                    'extensions': 'list (optional) — Фильтр по расширениям [.cpp, .h]'
                },
                'examples': [
                    'list_files(directory="Source/MyProject", recursive=True, extensions=[".cpp", ".h"])'
                ]
            },

            'search_files': {
                'description': 'Поиск файлов по паттерну в имени.',
                'function': self.file_system.search_files,
                'parameters': {
                    'pattern': 'str (required) — Паттерн для поиска'
                },
                'examples': [
                    'search_files(pattern="Character")',
                    'search_files(pattern=".uasset")'
                ]
            },

            'get_project_structure': {
                'description': 'Получить структуру проекта. Используй в начале для понимания архитектуры.',
                'function': self.file_system.get_project_structure,
                'parameters': {
                    'max_depth': 'int (optional) — Максимальная глубина (default: 3)'
                },
                'examples': [
                    'get_project_structure(max_depth=2)'
                ]
            },

            # Code Analyzer
            'find_class': {
                'description': 'Найти определение класса в проекте.',
                'function': self.code_analyzer.find_class,
                'parameters': {
                    'class_name': 'str (required) — Имя класса для поиска'
                },
                'examples': [
                    'find_class(class_name="AMyCharacter")',
                    'find_class(class_name="UGameInstance")'
                ]
            },

            'find_function': {
                'description': 'Найти функцию в проекте.',
                'function': self.code_analyzer.find_function,
                'parameters': {
                    'function_name': 'str (required) — Имя функции'
                },
                'examples': [
                    'find_function(function_name="BeginPlay")',
                    'find_function(function_name="Tick")'
                ]
            },

            # Indexer
            'index_file': {
                'description': 'Индексировать файл в Qdrant для RAG поиска. Используй когда нужно запомнить файл для будущего поиска.',
                'function': self.indexer.index_file,
                'parameters': {
                    'file_path': 'str (required) — Путь к файлу',
                    'project_name': 'str (optional) — Имя проекта (default: "default")'
                },
                'examples': [
                    'index_file(file_path="Source/MyProject/MyCharacter.h", project_name="MyProject")'
                ]
            },

            'index_directory': {
                'description': 'Индексировать директорию в Qdrant. Используй для массовой индексации модуля или всего проекта.',
                'function': self.indexer.index_directory,
                'parameters': {
                    'directory': 'str (required) — Путь к директории',
                    'project_name': 'str (optional) — Имя проекта',
                    'recursive': 'bool (optional) — Рекурсивно (default: True)',
                    'limit': 'int (optional) — Лимит файлов'
                },
                'examples': [
                    'index_directory(directory="Source/MyProject", project_name="MyProject")',
                    'index_directory(directory="Content", recursive=True, limit=50)'
                ]
            },

            'search_indexed': {
                'description': 'Поиск по проиндексированным файлам в Qdrant. Основной RAG поиск.',
                'function': self.indexer.search_indexed,
                'parameters': {
                    'query': 'str (required) — Поисковый запрос',
                    'limit': 'int (optional) — Количество результатов (default: 10)'
                },
                'examples': [
                    'search_indexed(query="character movement", limit=5)'
                ]
            },

            'get_indexed_files': {
                'description': 'Получить список всех проиндексированных файлов.',
                'function': self.indexer.get_indexed_files,
                'parameters': {
                    'project_name': 'str (optional) — Фильтр по проекту'
                },
                'examples': [
                    'get_indexed_files(project_name="MyProject")'
                ]
            },

            # Internet Search
            'search_duckduckgo': {
                'description': 'Поиск в интернете через DuckDuckGo. Используй для документации UE, ответов на вопросы.',
                'function': self.internet_search.search_duckduckgo,
                'parameters': {
                    'query': 'str (required) — Поисковый запрос',
                    'num_results': 'int (optional) — Количество результатов (default: 5)'
                },
                'examples': [
                    'search_duckduckgo(query="Unreal Engine 5 character movement component")'
                ]
            },

            'search_github': {
                'description': 'Поиск кода на GitHub.',
                'function': self.internet_search.search_github,
                'parameters': {
                    'query': 'str (required) — Поисковый запрос',
                    'language': 'str (optional) — Язык (default: "C++")'
                },
                'examples': [
                    'search_github(query="Unreal Engine character", language="C++")'
                ]
            },
        }

    def execute_tool(self, tool_name: str, **kwargs) -> Any:
        """Выполняет инструмент по имени"""
        if tool_name not in self.tools:
            return {'error': f'Unknown tool: {tool_name}'}

        tool = self.tools[tool_name]
        try:
            result = tool['function'](**kwargs)
            return result
        except Exception as e:
            return {'error': str(e)}

    def get_tool_descriptions(self) -> str:
        """Возвращает описание всех инструментов для LLM system prompt"""
        descriptions = []

        for name, info in self.tools.items():
            desc = f"**{name}**:\n"
            desc += f"  Description: {info['description']}\n"
            desc += f"  Parameters: {info['parameters']}\n"
            desc += f"  Examples: {info['examples']}\n"
            descriptions.append(desc)

        return "\n".join(descriptions)

    def list_tools(self) -> List[str]:
        """Список всех доступных инструментов"""
        return list(self.tools.keys())