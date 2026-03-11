# memory/embedding_model.py
"""Единая модель для всех компонентов"""

from sentence_transformers import SentenceTransformer

_model_instance = None


def get_embedding_model() -> SentenceTransformer:
    """Ленивая загрузка модели (singleton)"""
    global _model_instance
    if _model_instance is None:
        print("📥 Загрузка модели эмбеддингов...")
        _model_instance = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        print("✅ Модель загружена")
    return _model_instance