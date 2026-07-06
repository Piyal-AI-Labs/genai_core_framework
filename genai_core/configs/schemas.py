from typing import Optional
from pydantic import BaseModel, Field


class LLMConfig(BaseModel):
    provider: str
    model: str
    temperature: float = Field(default=0.7)
    max_tokens: int = Field(default=1000)


class EmbeddingConfig(BaseModel):
    provider: str
    model: str


class ChunkingConfig(BaseModel):
    strategy: str = "fixed"
    chunk_size: int = 512
    chunk_overlap: int = 50


class RetrieverConfig(BaseModel):
    top_k: int = 5


class ExperimentConfig(BaseModel):
    name: str
    output_dir: str = "outputs"
    seed: int = 42


class PathsConfig(BaseModel):
    prompt_templates: str = "templates"
    datasets: str = "datasets"
    outputs: str = "outputs"
    logs: str = "logs"
    reports: str = "reports"


class EvaluationConfig(BaseModel):
    bleu: bool = True
    rouge: bool = True
    bertscore: bool = True
    ragas: bool = False


class AppConfig(BaseModel):
    llm: LLMConfig
    embeddings: EmbeddingConfig
    chunking: Optional[ChunkingConfig] = ChunkingConfig()
    retriever: Optional[RetrieverConfig] = RetrieverConfig()
    experiment: ExperimentConfig
    evaluation: Optional[EvaluationConfig] = EvaluationConfig()
    paths: Optional[PathsConfig] = PathsConfig()