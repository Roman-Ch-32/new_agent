# mcp/indexer.py
"""File Indexer — Индексация файлов проекта в Qdrant (полная версия)"""

import hashlib
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
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

    IGNORE_EXTENSIONS = {
        '.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tga', '.exr',
        '.ini', '.dll', '.so', '.exe', '.bin', '.dat',
        '.cache', '.log', '.pdb', '.pch', '.obj', '.lib',
        '.dds', '.texp', '.shader', '.usmap', '.uproject'
    }

    BINARY_UE_FILES = {'.uasset', '.umap', }

    IGNORE_DIRS = {
        'Binaries', 'Intermediate', 'Saved', 'DerivedDataCache',
        '.git', '.svn', '__pycache__', 'node_modules', '.venv',
        'Content/Developers', 'Content/Engine', 'Plugins/Engine'
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

    # ========================================================================
    # ИНДЕКСАЦИЯ
    # ========================================================================

    def _should_ignore(self, path: Path) -> bool:
        """Проверяет нужно ли игнорировать файл"""
        for part in path.parts:
            if part in self.IGNORE_DIRS:
                return True

        suffix = path.suffix.lower()
        if suffix in self.BINARY_UE_FILES:
            return False

        if suffix in self.IGNORE_EXTENSIONS:
            return True

        try:
            if path.stat().st_size > 5 * 1024 * 1024:
                return True
        except:
            pass

        return False

    def _read_file(self, path: Path) -> Optional[str]:
        """Читает содержимое файла"""
        suffix = path.suffix.lower()
        """Читает содержимое файла"""
        suffix = path.suffix.lower()

        # ✅ Читаем .ini файлы
        if suffix == '.ini':
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    return f.read()
            except:
                return None

        if suffix == '.uproject':
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                return json.dumps(data, indent=2, ensure_ascii=False)
            except:
                return None

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
                'indexed_at': time.time()
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
        """Индексирует всю директорию"""
        dir_path = Path(directory)

        if not dir_path.exists():
            return {'status': 'error', 'message': f'Directory not found: {directory}'}

        stats = {'indexed': 0, 'skipped': 0, 'errors': 0, 'files': []}

        files = list(dir_path.rglob('*')) if recursive else list(dir_path.glob('*'))
        total = len([f for f in files if f.is_file()])

        processed = 0
        for file_path in files:
            if limit and processed >= limit:
                break

            if file_path.is_file():
                if self._should_ignore(file_path):
                    stats['skipped'] += 1
                    continue

                result = self.index_file(str(file_path), project_name)

                if result.get('status') == 'indexed':
                    stats['indexed'] += result.get('chunks', 0)
                    stats['files'].append({
                        'path': str(file_path),
                        'chunks': result.get('chunks', 0)
                    })
                elif result.get('status') == 'error':
                    stats['errors'] += 1

                processed += 1

        stats['message'] = f"Indexed {stats['indexed']} chunks from {len(stats['files'])} files"
        return stats

    # ========================================================================
    # ПОИСК
    # ========================================================================

    def search_indexed(self, query: str, limit: int = 10, project_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Поиск по проиндексированным файлам

        Args:
            query: Поисковый запрос
            limit: Количество результатов
            project_filter: Фильтр по проекту

        Returns:
            Список найденных документов
        """
        embedding = self.model.encode(query, normalize_embeddings=True)
        vector = list(embedding)

        # Создаём фильтр по проекту если указан
        query_filter = None
        if project_filter:
            query_filter = Filter(
                must=[
                    FieldCondition(
                        key='project',
                        match=MatchValue(value=project_filter)
                    )
                ]
            )

        response = self.client.query_points(
            collection_name=self.collection_name,
            query=vector,
            query_filter=query_filter,
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )

        return [
            {
                'path': r.payload.get('path', '') if r.payload else '',
                'file_name': r.payload.get('file_name', '') if r.payload else '',
                'content': r.payload.get('content', '') if r.payload else '',
                'score': float(r.score) if r.score else 0.0,
                'project': r.payload.get('project', '') if r.payload else '',
            }
            for r in response.points
        ]

    # ========================================================================
    # УПРАВЛЕНИЕ ИНДЕКСОМ
    # ========================================================================

    def get_indexed_files(self, project_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Возвращает список всех проиндексированных файлов

        Args:
            project_name: Фильтр по проекту

        Returns:
            Список файлов с метаданными
        """
        files = {}
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
                    if path and path not in files:
                        files[path] = {
                            'path': path,
                            'file_name': record.payload.get('file_name', ''),
                            'file_ext': record.payload.get('file_ext', ''),
                            'project': record.payload.get('project', ''),
                            'chunks': record.payload.get('total_chunks', 1),
                            'indexed_at': record.payload.get('indexed_at', 0)
                        }

            if offset is None:
                break

        if project_name:
            files = {k: v for k, v in files.items() if v.get('project') == project_name}

        return list(files.values())

    def delete_file_index(self, file_path: str) -> Dict[str, Any]:
        """
        Удаляет индексацию конкретного файла

        Args:
            file_path: Путь к файлу

        Returns:
            Статус операции
        """
        path = str(Path(file_path).absolute())

        try:
            # Находим все точки для этого файла
            file_filter = Filter(
                must=[
                    FieldCondition(
                        key='path',
                        match=MatchValue(value=path)
                    )
                ]
            )

            # Считаем сколько точек будет удалено
            count_result = self.client.count(
                collection_name=self.collection_name,
                count_filter=file_filter
            )

            # Удаляем точки
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
        """
        Обновляет индексацию файла (удаляет старую + индексирует заново)

        Args:
            file_path: Путь к файлу
            project_name: Имя проекта

        Returns:
            Статус операции
        """
        # Сначала удаляем старую индексацию
        delete_result = self.delete_file_index(file_path)

        # Затем индексируем заново
        index_result = self.index_file(file_path, project_name)

        return {
            'status': 'updated',
            'deleted': delete_result,
            'indexed': index_result
        }

    def clear_collection(self) -> Dict[str, Any]:
        """
        Очищает всю коллекцию

        Returns:
            Статус операции
        """
        try:
            # Считаем количество точек перед очисткой
            count_result = self.client.count(
                collection_name=self.collection_name,
                count_filter=None
            )

            # Пересоздаём коллекцию
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
        """
        Считает количество проиндексированных точек

        Args:
            project_name: Фильтр по проекту

        Returns:
            Количество точек и файлов
        """
        # Общий подсчёт
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

        # Получаем уникальные файлы
        files = self.get_indexed_files(project_name)

        return {
            'status': 'ok',
            'total_points': count_result.count,
            'total_files': len(files),
            'project': project_name or 'all'
        }

    def get_collection_info(self) -> Dict[str, Any]:
        """
        Получает информацию о коллекции

        Returns:
            Информация о коллекции
        """
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
        """
        Проверяет проиндексирован ли файл

        Args:
            file_path: Путь к файлу

        Returns:
            Статус индексации
        """
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