# memory/embedding_model.py
"""Единая локальная embedding-модель для всех компонентов."""

from pathlib import Path

from sentence_transformers import SentenceTransformer

DEFAULT_MODEL_REPO = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
HF_HUB_CACHE = Path.home() / ".cache" / "huggingface" / "hub"

_model_instance: SentenceTransformer | None = None
_loaded_model_repo: str | None = None


def _normalize_model_repo(model_name: str | None) -> str:
    if not model_name:
        return DEFAULT_MODEL_REPO
    if "/" in model_name:
        return model_name
    return f"sentence-transformers/{model_name}"


def _resolve_local_snapshot(model_repo: str) -> Path | None:
    repo_dir = HF_HUB_CACHE / f"models--{model_repo.replace('/', '--')}"
    if not repo_dir.exists():
        return None

    ref_main = repo_dir / "refs" / "main"
    if ref_main.exists():
        snapshot_name = ref_main.read_text(encoding="utf-8").strip()
        snapshot_dir = repo_dir / "snapshots" / snapshot_name
        if snapshot_dir.exists():
            return snapshot_dir

    snapshots_dir = repo_dir / "snapshots"
    if not snapshots_dir.exists():
        return None

    candidates = sorted(path for path in snapshots_dir.iterdir() if path.is_dir())
    return candidates[-1] if candidates else None


def get_embedding_model(model_name: str | None = None) -> SentenceTransformer:
    """Ленивая загрузка модели только из локального cache."""
    global _model_instance, _loaded_model_repo

    model_repo = _normalize_model_repo(model_name)
    if _model_instance is not None and _loaded_model_repo == model_repo:
        return _model_instance

    snapshot_dir = _resolve_local_snapshot(model_repo)
    load_target = str(snapshot_dir) if snapshot_dir else model_repo

    print("📥 Загрузка модели эмбеддингов...")
    try:
        _model_instance = SentenceTransformer(load_target, local_files_only=True)
    except Exception as exc:
        hint = str(snapshot_dir) if snapshot_dir else model_repo
        raise RuntimeError(
            f"Не удалось загрузить локальную embedding-модель: {hint}. "
            f"Проверь локальный HuggingFace cache."
        ) from exc

    _loaded_model_repo = model_repo
    print("✅ Модель загружена")
    return _model_instance
