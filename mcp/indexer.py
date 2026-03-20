# mcp/indexer.py
"""File Indexer — Индексация файлов и папок проекта в Qdrant"""

import hashlib
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue
)
from sentence_transformers import SentenceTransformer


class FileIndexer:
    """Индексатор файлов для RAG поиска"""

    # ✅ Игнорируемые расширения (бинарники, кэш, логи)
    IGNORE_EXTENSIONS = {
        '.dll', '.so', '.exe', '.bin', '.dat', '.dylib',
        '.pdb', '.pch', '.obj', '.lib', '.a', '.o',
        '.cache', '.tmp', '.swp', '.swo',
        '.png', '.jpg', '.jpeg', '.tga', '.dds', '.exr',
        '.log',
    }

    # ✅ UE бинарные ассеты (обрабатываем отдельно)
    BINARY_UE_FILES = {'.uasset', '.umap'}

    # ✅ Текстовые файлы которые ВСЕГДА индексируем
    TEXT_EXTENSIONS = {
        '.uproject', '.uplugin',
        '.ini',
        '.cpp', '.h', '.hpp', '.cc', '.cxx',
        '.cs',
        '.py',
        '.txt', '.md', '.json', '.xml', '.yaml', '.yml',
        '.shader', '.hlsl', '.glsl',
        '.anim', '.state', '.graph',
    }

    # ✅ Игнорируемые папки
    IGNORE_DIRS = {
        'Binaries',
        'Intermediate',
        'DerivedDataCache',
        'Saved',
        '.git',
        '__pycache__',
        'node_modules',
        '.venv', 'venv',
    }

    def __init__(
            self,
            qdrant_url: str = 'http://localhost:6333',
            collection_name: str = 'ue_project',
            model_name: str = 'paraphrase-multilingual-MiniLM-L12-v2'
    ):
        self.client = QdrantClient(url=qdrant_url)
        self.collection_name = collection_name
        self.model = SentenceTransformer(model_name)
        self.vector_size = self.model.get_sentence_embedding_dimension() or 384
        self._ensure_collection()

    def _ensure_collection(self):
        """Создаёт коллекцию если не существует"""
        try:
            exists = self.client.collection_exists(self.collection_name)
            if not exists:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE),
                )
                print(f"✅ Создана коллекция: {self.collection_name}")
        except Exception as e:
            print(f"⚠️ Warning: {e}")

    def _should_ignore(self, path: Path) -> bool:
        """Проверяет нужно ли игнорировать файл или папку"""

        # ✅ Проверяем папки в пути
        for part in path.parts:
            if part in self.IGNORE_DIRS:
                return True

        # ✅ Если это папка — не игнорируем (будем индексировать структуру)
        if path.is_dir():
            return False

        suffix = path.suffix.lower()

        # ✅ Текстовые файлы — всегда индексируем
        if suffix in self.TEXT_EXTENSIONS:
            return False

        # ✅ UE бинарные ассеты — пробуем прочитать заголовок
        if suffix in self.BINARY_UE_FILES:
            return False

        # ✅ Игнорируем по расширению
        if suffix in self.IGNORE_EXTENSIONS:
            return True

        # ✅ Пропускаем очень большие файлы (>5MB)
        try:
            if path.stat().st_size > 5 * 1024 * 1024:
                return True
        except:
            pass

        # ✅ Неизвестные расширения — игнорируем
        return True

    def _read_file(self, path: Path) -> Optional[str]:
        """Читает содержимое файла"""
        suffix = path.suffix.lower()

        # ✅ .ini — конфиги
        if suffix == '.ini':
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    return f.read()
            except:
                return None

        # ✅ .uproject / .uplugin — JSON
        if suffix in ('.uproject', '.uplugin'):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                return json.dumps(data, indent=2, ensure_ascii=False)
            except:
                return None

        # ✅ UE бинарные ассеты — только заголовок
        if suffix in self.BINARY_UE_FILES:
            try:
                with open(path, 'rb') as f:
                    header = f.read(1024)
                    text = header.decode('utf-8', errors='ignore')
                    import re
                    refs = re.findall(r'/[A-Za-z0-9_/]+', text)
                    if refs:
                        return f"Asset references: {' '.join(set(refs)[:20])}"
            except:
                pass
            return None

        # ✅ Текстовые файлы — читаем с разными кодировками
        try:
            for encoding in ['utf-8', 'cp1251', 'latin-1']:
                try:
                    with open(path, 'r', encoding=encoding) as f:
                        return f.read()
                except UnicodeDecodeError:
                    continue
            return None
        except:
            return None

    def _chunk_content(self, content: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """Разбивает контент на чанки"""
        if not content or len(content) < chunk_size:
            return [content] if content else []

        chunks = []
        start = 0
        content_length = len(content)

        while start < content_length:
            end = start + chunk_size
            chunk = content[start:end]

            if end < content_length:
                last_newline = chunk.rfind('\n')
                if last_newline > chunk_size // 2:
                    end = start + last_newline + 1
                    chunk = content[start:end]

            chunks.append(chunk.strip())
            start = end - overlap

        return chunks

    def _index_directory_structure(self, dir_path: Path, project_name: str = 'default') -> Dict[str, Any]:
        """Индексирует структуру папки как отдельную запись"""
        if not dir_path.exists() or not dir_path.is_dir():
            return {'status': 'error', 'message': f'Not a directory: {dir_path}'}

        # ✅ Собираем информацию о папке
        try:
            subdirs = [d.name for d in dir_path.iterdir() if d.is_dir() and d.name not in self.IGNORE_DIRS]
            files = [f.name for f in dir_path.iterdir() if f.is_file() and not self._should_ignore(f)]
        except PermissionError:
            return {'status': 'error', 'message': f'Permission denied: {dir_path}'}

        # ✅ Создаём текстовое описание папки
        description = f"Directory: {dir_path.name}\n"
        description += f"Path: {str(dir_path.absolute())}\n"

        if subdirs:
            description += f"Subdirectories: {', '.join(sorted(subdirs))}\n"
        else:
            description += "Subdirectories: none\n"

        if files:
            file_exts = {}
            for f in files:
                ext = Path(f).suffix.lower() or 'no_extension'
                file_exts[ext] = file_exts.get(ext, 0) + 1
            description += f"Files: {len(files)} total ("
            description += ', '.join(f"{count} {ext}" for ext, count in sorted(file_exts.items()))
            description += ")\n"
            description += f"File names: {', '.join(sorted(files)[:20])}"
        else:
            description += "Files: none\n"

        # ✅ Создаём эмбеддинг и сохраняем в Qdrant
        embedding = self.model.encode(description, normalize_embeddings=True)
        vector = list(embedding)

        payload = {
            'path': str(dir_path.absolute()),
            'file_name': dir_path.name,
            'file_ext': '.directory',
            'project': project_name,
            'chunk_index': 0,
            'total_chunks': 1,
            'content': description,
            'type': 'directory',
            'subdirectories': subdirs,
            'files_count': len(files),
            'indexed_at': time.time()
        }

        point_id = int(hashlib.md5(f"{dir_path.absolute()}:dir".encode()).hexdigest(), 16) % (10 ** 9)

        self.client.upsert(
            collection_name=self.collection_name,
            points=[PointStruct(id=point_id, vector=vector, payload=payload)]
        )

        return {
            'status': 'indexed',
            'path': str(dir_path),
            'subdirs': len(subdirs),
            'files': len(files)
        }

    def index_file(self, file_path: str, project_name: str = 'default') -> Dict[str, Any]:
        """Индексирует один файл"""
        path = Path(file_path)

        if not path.exists():
            return {'status': 'error', 'message': f'File not found: {file_path}'}

        if self._should_ignore(path):
            return {'status': 'skipped', 'message': f'File ignored: {file_path}'}

        content = self._read_file(path)
        if not content:
            return {'status': 'error', 'message': f'Could not read file: {file_path}'}

        chunks = self._chunk_content(content)
        if not chunks:
            return {'status': 'skipped', 'message': f'No content to index: {file_path}'}

        points = []
        for i, chunk in enumerate(chunks):
            if not chunk.strip():
                continue

            embedding = self.model.encode(chunk, normalize_embeddings=True)
            vector = list(embedding)

            payload = {
                'path': str(path.absolute()),
                'file_name': path.name,
                'file_ext': path.suffix,
                'project': project_name,
                'chunk_index': i,
                'total_chunks': len(chunks),
                'content': chunk,
                'file_hash': hashlib.md5(f"{path.absolute()}:{i}".encode()).hexdigest(),
                'indexed_at': time.time(),
                'type': 'file'
            }

            point_id = int(hashlib.md5(f"{path.absolute()}:{i}".encode()).hexdigest(), 16) % (10 ** 9)

            points.append(PointStruct(id=point_id, vector=vector, payload=payload))

        if points:
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )

        return {
            'status': 'indexed',
            'file': str(path),
            'chunks': len(points),
            'message': f'Indexed {len(points)} chunks from {path.name}'
        }

    def index_directory(
            self,
            directory: str,
            project_name: str = 'default',
            recursive: bool = True,
            limit: Optional[int] = None
    ) -> Dict[str, Any]:
        """Индексирует всю директорию (файлы + папки)"""
        dir_path = Path(directory)

        if not dir_path.exists():
            return {'status': 'error', 'message': f'Directory not found: {directory}'}

        stats = {
            'indexed': 0,
            'skipped': 0,
            'errors': 0,
            'files': [],
            'directories': [],
            'skip_reasons': {}
        }

        all_items = list(dir_path.rglob('*')) if recursive else list(dir_path.glob('*'))
        total_files = len([f for f in all_items if f.is_file()])
        total_dirs = len([d for d in all_items if d.is_dir()])

        print(f"📂 Найдено: {total_files} файлов, {total_dirs} папок")

        processed = 0

        # ✅ 1. Сначала индексируем папки (структуру)
        dirs_to_index = [d for d in all_items if d.is_dir() and not self._should_ignore(d)]

        if recursive:
            dirs_to_index.insert(0, dir_path)

        for dir_path_item in dirs_to_index:
            if limit and processed >= limit:
                break

            result = self._index_directory_structure(dir_path_item, project_name)

            if result.get('status') == 'indexed':
                stats['directories'].append({
                    'path': str(dir_path_item),
                    'subdirs': result.get('subdirs', 0),
                    'files': result.get('files', 0)
                })
                stats['indexed'] += 1

            processed += 1

        # ✅ 2. Затем индексируем файлы
        for file_path in all_items:
            if limit and processed >= limit:
                break

            if file_path.is_file():
                if self._should_ignore(file_path):
                    stats['skipped'] += 1

                    suffix = file_path.suffix.lower()
                    if suffix in self.IGNORE_EXTENSIONS:
                        reason = f'ext:{suffix}'
                    elif any(part in self.IGNORE_DIRS for part in file_path.parts):
                        reason = 'dir:ignored'
                    else:
                        reason = 'unknown'

                    stats['skip_reasons'][reason] = stats['skip_reasons'].get(reason, 0) + 1
                    processed += 1
                    continue

                result = self.index_file(str(file_path), project_name)

                if result.get('status') == 'indexed':
                    stats['indexed'] += result.get('chunks', 0)
                    stats['files'].append({
                        'path': str(file_path),
                        'chunks': result.get('chunks', 0)
                    })
                    if len(stats['files']) % 100 == 0:
                        print(f"✅ [{len(stats['files'])}/{total_files}] файлов проиндексировано")
                elif result.get('status') == 'error':
                    stats['errors'] += 1

                processed += 1

        # ✅ Печатаем статистику
        print(f"\n📊 ИТОГИ:")
        print(f"   Папок: {len(stats['directories'])}")
        print(f"   Файлов: {len(stats['files'])}")
        print(f"   Чанков: {stats['indexed']}")
        print(f"   Пропущено: {stats['skipped']}")

        if stats['skip_reasons']:
            print(f"\n⚠️ Причины пропуска (топ-5):")
            for reason, count in sorted(stats['skip_reasons'].items(), key=lambda x: -x[1])[:5]:
                print(f"   {reason}: {count}")

        stats[
            'message'] = f"Indexed {len(stats['directories'])} directories + {len(stats['files'])} files ({stats['indexed']} chunks)"
        return stats

    # ========================================================================
    # ПОИСК
    # ========================================================================

    def search_indexed(
            self,
            query: str,
            limit: int = 10,
            project_filter: Optional[str] = None,
            include_directories: bool = True
    ) -> List[Dict[str, Any]]:
        """Поиск по проиндексированным файлам и папкам"""
        embedding = self.model.encode(query, normalize_embeddings=True)
        vector = list(embedding)

        must_conditions = []
        if project_filter:
            must_conditions.append(
                FieldCondition(key='project', match=MatchValue(value=project_filter))
            )

        if not include_directories:
            must_conditions.append(
                FieldCondition(key='type', match=MatchValue(value='file'))
            )

        query_filter = Filter(must=must_conditions) if must_conditions else None

        response = self.client.query_points(
            collection_name=self.collection_name,
            query=vector,
            query_filter=query_filter,
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )

        results = []
        for r in response.points:
            if r.payload:
                result = {
                    'path': r.payload.get('path', ''),
                    'file_name': r.payload.get('file_name', ''),
                    'content': r.payload.get('content', ''),
                    'score': float(r.score) if r.score else 0.0,
                    'project': r.payload.get('project', ''),
                    'type': r.payload.get('type', 'file'),
                }

                if result['type'] == 'directory':
                    result['subdirectories'] = r.payload.get('subdirectories', [])
                    result['files_count'] = r.payload.get('files_count', 0)

                results.append(result)

        return results

    # ========================================================================
    # УПРАВЛЕНИЕ ИНДЕКСОМ
    # ========================================================================

    def get_indexed_files(
            self,
            project_name: Optional[str] = None,
            include_directories: bool = False
    ) -> List[Dict[str, Any]]:
        """Возвращает список всех проиндексированных файлов (и опционально папок)"""
        items = {}
        offset = None

        while True:
            records, offset = self.client.scroll(
                collection_name=self.collection_name,
                limit=100,
                offset=offset,
                with_payload=True,
                with_vectors=False
            )

            for record in records:
                if record.payload:
                    path = record.payload.get('path', '')
                    item_type = record.payload.get('type', 'file')

                    if not include_directories and item_type == 'directory':
                        continue

                    if path and path not in items:
                        item = {
                            'path': path,
                            'file_name': record.payload.get('file_name', ''),
                            'file_ext': record.payload.get('file_ext', ''),
                            'project': record.payload.get('project', ''),
                            'chunks': record.payload.get('total_chunks', 1),
                            'indexed_at': record.payload.get('indexed_at', 0),
                            'type': item_type,
                        }

                        if item_type == 'directory':
                            item['subdirectories'] = record.payload.get('subdirectories', [])
                            item['files_count'] = record.payload.get('files_count', 0)

                        items[path] = item

            if offset is None:
                break

        if project_name:
            items = {k: v for k, v in items.items() if v.get('project') == project_name}

        return list(items.values())

    def get_project_structure(self, max_depth: int = 3) -> Dict[str, Any]:
        """Возвращает структуру проекта как дерево"""
        all_items = self.get_indexed_files(include_directories=True)

        directories = [i for i in all_items if i.get('type') == 'directory']
        files = [i for i in all_items if i.get('type') != 'directory']

        def build_tree(path: str, depth: int = 0) -> Dict[str, Any]:
            if depth > max_depth:
                return {}

            node = {
                'name': Path(path).name or path,
                'path': path,
                'type': 'directory',
                'children': {},
                'files': []
            }

            for d in directories:
                d_path = d['path']
                if d_path.startswith(path + '/') and d_path.count('/') == path.count('/') + 1:
                    node['children'][d['file_name']] = build_tree(d_path, depth + 1)

            for f in files:
                f_path = f['path']
                if f_path.startswith(path + '/') and f_path.count('/') == path.count('/') + 1:
                    node['files'].append({
                        'name': f['file_name'],
                        'path': f['path'],
                        'ext': f['file_ext']
                    })

            return node

        root_paths = set()
        for d in directories:
            parts = d['path'].split('/')
            if len(parts) <= max_depth + 2:
                root_paths.add('/'.join(parts[:max_depth + 2]))

        root_path = min(root_paths) if root_paths else '/'

        return build_tree(root_path)

    def delete_file_index(self, file_path: str) -> Dict[str, Any]:
        """Удаляет индексацию конкретного файла"""
        path = str(Path(file_path).absolute())

        try:
            file_filter = Filter(
                must=[
                    FieldCondition(
                        key='path',
                        match=MatchValue(value=path)
                    )
                ]
            )

            count_result = self.client.count(
                collection_name=self.collection_name,
                count_filter=file_filter
            )

            self.client.delete(
                collection_name=self.collection_name,
                points_selector=file_filter
            )

            return {
                'status': 'ok',
                'message': f'Deleted {count_result.count} points for {file_path}',
                'deleted_count': count_result.count
            }
        except Exception as e:
            return {
                'status': 'error',
                'message': str(e)
            }

    def update_file_index(self, file_path: str, project_name: str = 'default') -> Dict[str, Any]:
        """Обновляет индексацию файла"""
        delete_result = self.delete_file_index(file_path)
        index_result = self.index_file(file_path, project_name)

        return {
            'status': 'updated',
            'deleted': delete_result,
            'indexed': index_result
        }

    def clear_collection(self) -> Dict[str, Any]:
        """Очищает всю коллекцию"""
        try:
            count_result = self.client.count(
                collection_name=self.collection_name,
                count_filter=None
            )

            self.client.delete_collection(self.collection_name)
            self._ensure_collection()

            return {
                'status': 'ok',
                'message': f'Collection cleared. Deleted {count_result.count} points.',
                'deleted_count': count_result.count
            }
        except Exception as e:
            return {
                'status': 'error',
                'message': str(e)
            }

    def count_indexed(self, project_name: Optional[str] = None) -> Dict[str, Any]:
        """Считает количество проиндексированных точек"""
        filter_condition = None
        if project_name:
            filter_condition = Filter(
                must=[
                    FieldCondition(
                        key='project',
                        match=MatchValue(value=project_name)
                    )
                ]
            )

        count_result = self.client.count(
            collection_name=self.collection_name,
            count_filter=filter_condition,
            exact=True
        )

        files = self.get_indexed_files(project_name)

        return {
            'status': 'ok',
            'total_points': count_result.count,
            'total_files': len(files),
            'project': project_name or 'all'
        }

    def get_collection_info(self) -> Dict[str, Any]:
        """Получает информацию о коллекции"""
        try:
            info = self.client.get_collection(self.collection_name)

            return {
                'status': 'ok',
                'collection_name': self.collection_name,
                'vectors_count': info.points_count,
                'vector_size': info.config.params.vectors.size,
                'distance': str(info.config.params.vectors.distance),
                'indexed_vectors_count': info.indexed_vectors_count or 0
            }
        except Exception as e:
            return {
                'status': 'error',
                'message': str(e)
            }

    def is_file_indexed(self, file_path: str) -> Dict[str, Any]:
        """Проверяет проиндексирован ли файл"""
        path = str(Path(file_path).absolute())

        file_filter = Filter(
            must=[
                FieldCondition(
                    key='path',
                    match=MatchValue(value=path)
                )
            ]
        )

        count_result = self.client.count(
            collection_name=self.collection_name,
            count_filter=file_filter
        )

        return {
            'status': 'ok',
            'is_indexed': count_result.count > 0,
            'chunks_count': count_result.count,
            'file_path': file_path
        }