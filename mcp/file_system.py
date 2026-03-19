# mcp/file_system.py
"""File System Tools — Доступ к файловой системе UE5 проекта"""

import os
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional


class FileSystemTools:
    """Инструменты для работы с файловой системой"""

    IGNORE_DIRS = {
        'Binaries', 'Intermediate', 'Saved', 'DerivedDataCache',
        '.git', '.svn', '__pycache__', 'node_modules', '.venv',
        'Content/Developers', 'Content/Engine', 'Plugins/Engine'
    }

    def __init__(self, project_path: str):
        self.project_path = Path(project_path)
        if not self.project_path.exists():
            raise ValueError(f"Project path does not exist: {project_path}")

    def read_file(self, file_path: str, lines: Optional[tuple] = None) -> str:
        """Читает файл целиком или по строкам"""
        path = self.project_path / file_path

        if not path.exists():
            return f"❌ Файл не найден: {file_path}"

        if not path.is_file():
            return f"❌ Это не файл: {file_path}"

        try:
            with open(path, 'r', encoding='utf-8') as f:
                content = f.readlines()

            if lines:
                start, end = lines
                content = content[start - 1:end]

            return ''.join(content)
        except UnicodeDecodeError:
            for encoding in ['cp1251', 'latin-1']:
                try:
                    with open(path, 'r', encoding=encoding) as f:
                        return f.read()
                except:
                    continue
            return f"❌ Не удалось прочитать файл (кодировка)"
        except Exception as e:
            return f"❌ Ошибка чтения: {e}"

    def write_file(self, file_path: str, content: str, create_dirs: bool = True) -> str:
        """Записывает файл"""
        path = self.project_path / file_path

        for part in path.parts:
            if part in self.IGNORE_DIRS:
                return f"❌ Запись запрещена в: {part}"

        try:
            if create_dirs:
                path.parent.mkdir(parents=True, exist_ok=True)

            with open(path, 'w', encoding='utf-8') as f:
                f.write(content)

            return f"✅ Файл записан: {file_path}"
        except Exception as e:
            return f"❌ Ошибка записи: {e}"

    def list_files(self, directory: str = '', recursive: bool = False,
                   extensions: Optional[List[str]] = None) -> List[str]:
        """Список файлов в директории"""
        dir_path = self.project_path / directory if directory else self.project_path

        if not dir_path.exists():
            return []

        files = []
        pattern = '**/*' if recursive else '*'

        for f in dir_path.glob(pattern):
            if f.is_file():
                if any(ignore in f.parts for ignore in self.IGNORE_DIRS):
                    continue

                if extensions and f.suffix not in extensions:
                    continue

                files.append(str(f.relative_to(self.project_path)))

        return files[:500]

    def search_files(self, pattern: str) -> List[str]:
        """Поиск файлов по паттерну"""
        matches = []

        for f in self.project_path.rglob(f'*{pattern}*'):
            if f.is_file():
                if any(ignore in f.parts for ignore in self.IGNORE_DIRS):
                    continue
                matches.append(str(f.relative_to(self.project_path)))

        return matches[:100]

    def get_project_structure(self, max_depth: int = 3) -> Dict[str, Any]:
        """Возвращает структуру проекта"""

        def build_tree(path: Path, depth: int) -> Dict[str, Any]:
            if depth > max_depth:
                return {'type': 'directory', 'name': path.name, 'truncated': True}

            result = {
                'type': 'directory',
                'name': path.name if path != self.project_path else 'project_root',
                'children': []
            }

            try:
                for item in sorted(path.iterdir()):
                    if item.name in self.IGNORE_DIRS:
                        continue

                    if item.is_file():
                        result['children'].append({
                            'type': 'file',
                            'name': item.name,
                            'size': item.stat().st_size,
                            'suffix': item.suffix
                        })
                    elif item.is_dir():
                        result['children'].append(build_tree(item, depth + 1))
            except PermissionError:
                pass

            return result

        return build_tree(self.project_path, 0)