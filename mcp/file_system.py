"""File system tools with workspace-root protection."""

from pathlib import Path
from typing import List, Dict, Any, Optional


class FileSystemTools:
    IGNORE_DIRS = {
        'Binaries', 'Intermediate', 'Saved', 'DerivedDataCache',
        '.git', '.svn', '__pycache__', 'node_modules', '.venv',
        'Content/Developers', 'Content/Engine', 'Plugins/Engine',
    }

    def __init__(self, project_path: str):
        self.project_path = Path(project_path).expanduser().resolve()
        if not self.project_path.exists():
            raise ValueError(f'Project path does not exist: {project_path}')

    def _resolve_path(self, file_path: str) -> Path:
        candidate = (self.project_path / file_path).resolve()
        try:
            candidate.relative_to(self.project_path)
        except ValueError as exc:
            raise ValueError(f'Путь вне workspace root запрещён: {file_path}') from exc

        for part in candidate.parts:
            if part in self.IGNORE_DIRS:
                raise ValueError(f'Операция запрещена в каталоге: {part}')

        return candidate

    def read_file(self, file_path: str, lines: Optional[tuple[int, int]] = None) -> str:
        try:
            path = self._resolve_path(file_path)
        except ValueError as e:
            return f'❌ {e}'

        if not path.exists():
            return f'❌ Файл не найден: {file_path}'
        if not path.is_file():
            return f'❌ Это не файл: {file_path}'

        try:
            content = path.read_text(encoding='utf-8').splitlines(keepends=True)
        except UnicodeDecodeError:
            for encoding in ('cp1251', 'latin-1'):
                try:
                    return path.read_text(encoding=encoding)
                except Exception:
                    pass
            return '❌ Не удалось прочитать файл (кодировка)'
        except Exception as e:
            return f'❌ Ошибка чтения: {e}'

        if lines:
            start, end = lines
            content = content[max(start - 1, 0):end]
        return ''.join(content)

    def write_file(self, file_path: str, content: str, overwrite: bool = False, create_dirs: bool = True) -> Dict[str, Any]:
        try:
            path = self._resolve_path(file_path)
        except ValueError as e:
            return {'success': False, 'error': str(e)}

        if path.exists() and not overwrite:
            return {'success': False, 'error': f'Файл уже существует: {file_path}. Используйте overwrite=true.'}

        try:
            if create_dirs:
                path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding='utf-8')
            return {
                'success': True,
                'path': str(path),
                'relative_path': str(path.relative_to(self.project_path)),
                'size': len(content),
                'message': f'Файл записан: {file_path}',
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def list_files(self, directory: str = '', recursive: bool = False, extensions: Optional[List[str]] = None) -> List[str]:
        try:
            dir_path = self._resolve_path(directory) if directory else self.project_path
        except ValueError:
            return []

        if not dir_path.exists():
            return []

        pattern = '**/*' if recursive else '*'
        files: List[str] = []
        for path in dir_path.glob(pattern):
            if not path.is_file():
                continue
            try:
                relative = path.relative_to(self.project_path)
            except ValueError:
                continue
            if any(part in self.IGNORE_DIRS for part in relative.parts):
                continue
            if extensions and path.suffix not in extensions:
                continue
            files.append(str(relative))
        return files[:500]

    def search_files(self, pattern: str) -> List[str]:
        matches: List[str] = []
        for path in self.project_path.rglob(f'*{pattern}*'):
            if not path.is_file():
                continue
            rel = path.relative_to(self.project_path)
            if any(part in self.IGNORE_DIRS for part in rel.parts):
                continue
            matches.append(str(rel))
        return matches[:100]

    def get_project_structure(self, max_depth: int = 3) -> Dict[str, Any]:
        def build_tree(path: Path, depth: int) -> Dict[str, Any]:
            if depth > max_depth:
                return {'type': 'directory', 'name': path.name, 'truncated': True}

            result: Dict[str, Any] = {
                'type': 'directory',
                'name': path.name if path != self.project_path else 'project_root',
                'children': [],
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
                            'suffix': item.suffix,
                        })
                    elif item.is_dir():
                        result['children'].append(build_tree(item, depth + 1))
            except PermissionError:
                result['permission_denied'] = True
            return result

        return build_tree(self.project_path, 0)
