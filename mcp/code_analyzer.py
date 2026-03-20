"""Code analyzer constrained to project root."""

from pathlib import Path
from typing import List, Dict, Any


class CodeAnalyzer:
    def __init__(self, project_path: str):
        self.project_path = Path(project_path).expanduser().resolve()

    def find_class(self, class_name: str) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        for pattern, check in (('*.h', lambda c: f'class {class_name}' in c or f'struct {class_name}' in c), ('*.cpp', lambda c: f'{class_name}::' in c)):
            for file_path in self.project_path.rglob(pattern):
                if self._should_ignore(file_path):
                    continue
                try:
                    content = file_path.read_text(encoding='utf-8')
                except Exception:
                    continue
                if check(content):
                    results.append({
                        'file': str(file_path.relative_to(self.project_path)),
                        'matches': content.count(class_name),
                    })
        return results[:20]

    def find_function(self, function_name: str) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        for ext in ('*.cpp', '*.h', '*.cs', '*.py'):
            for file_path in self.project_path.rglob(ext):
                if self._should_ignore(file_path):
                    continue
                try:
                    content = file_path.read_text(encoding='utf-8')
                except Exception:
                    continue
                if function_name in content:
                    results.append({
                        'file': str(file_path.relative_to(self.project_path)),
                        'matches': content.count(function_name),
                    })
        return results[:50]

    def _should_ignore(self, path: Path) -> bool:
        ignore_dirs = {'Binaries', 'Intermediate', 'Saved', '.git', '__pycache__', 'DerivedDataCache'}
        rel_parts = path.relative_to(self.project_path).parts
        return any(part in ignore_dirs for part in rel_parts)
