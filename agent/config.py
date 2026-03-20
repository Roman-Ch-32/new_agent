# agent/config.py
"""Конфигурация агента"""

from dataclasses import dataclass, field


@dataclass
class LLMConfig:
    model: str = 'qwen-ue'
    base_url: str = 'http://localhost:8080/v1'
    temperature: float = 0.03
    top_p: float = 0.9
    num_predict: int = 2048
    repeat_penalty: float = 1.1


@dataclass
class QdrantConfig:
    url: str = 'http://localhost:6333'
    collection: str = 'ue_project'


@dataclass
class ProjectConfig:
    path: str = '/home/roman/Документы/Unreal Projects/мульт'


@dataclass
class Config:
    llm: LLMConfig = field(default_factory=LLMConfig)
    qdrant: QdrantConfig = field(default_factory=QdrantConfig)
    project: ProjectConfig = field(default_factory=ProjectConfig)
    debug: bool = True


config = Config()
