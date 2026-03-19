# mcp/code_analyzer.py
"""Code Analyzer — Анализ кода UE5 проекта"""

import re
from pathlib import Path
from typing import List, Dict, Any


class CodeAnalyzer:
    """Инструменты для анализа кода"""

    def __init__(self, project_path: str):
        self.project_path = Path(project_path)

    def find_class(self, class_name: str) -> List[Dict[str, Any]]:
        """Ищет определение класса в проекте"""
        results = []

        for file_path in self.project_path.rglob('*.h'):
            if self._should_ignore(file_path):
                continue
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if f'class {class_name}' in content or f'struct {class_name}' in content:
                        results.append({
                            'file': str(file_path.relative_to(self.project_path)),
                            'type': 'header',
                            'matches': content.count(f'{class_name}')
                        })
            except:
                continue

        for file_path in self.project_path.rglob('*.cpp'):
            if self._should_ignore(file_path):
                continue
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if f'{class_name}::' in content:
                        results.append({
                            'file': str(file_path.relative_to(self.project_path)),
                            'type': 'implementation',
                            'matches': content.count(f'{class_name}::')
                        })
            except:
                continue

        return results[:20]

    def find_function(self, function_name: str) -> List[Dict[str, Any]]:
        """Ищет определение функции"""
        results = []

        for ext in ['*.cpp', '*.h', '*.cs', '*.py']:
            for file_path in self.project_path.rglob(ext):
                if self._should_ignore(file_path):
                    continue
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        if function_name in content:
                            results.append({
                                'file': str(file_path.relative_to(self.project_path)),
                                'matches': content.count(function_name)
                            })
                except:
                    continue

        return results[:50]

    def _should_ignore(self, path: Path) -> bool:
        """Проверяет нужно ли игнорировать файл"""
        ignore_dirs = {'Binaries', 'Intermediate', 'Saved', '.git', '__pycache__'}
        return any(ignore in path.parts for ignore in ignore_dirs)