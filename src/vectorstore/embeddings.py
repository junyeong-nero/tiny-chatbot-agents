import logging
from pathlib import Path
from typing import Any

import yaml
from chromadb import Documents, EmbeddingFunction, Embeddings
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).parent.parent.parent / "configs" / "embedding_config.yaml"


def load_embedding_config() -> dict[str, Any]:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_model_config(model_key: str | None = None) -> dict[str, Any]:
    config = load_embedding_config()

    if model_key is None:
        model_key = config["default_model"]

    if model_key not in config["models"]:
        raise ValueError(f"Unknown model: {model_key}. Available: {list(config['models'].keys())}")

    model_config = config["models"][model_key]
    model_config["device"] = config.get("device")
    return model_config


class LocalEmbeddingFunction(EmbeddingFunction[Documents]):
    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
        normalize_embeddings: bool | None = None,
        prefix: str = "",
    ) -> None:
        model_config = get_model_config(model_name)

        self.model_name = model_config["name"]
        self.model_type = model_config.get("type", "standard")
        self.normalize_embeddings = (
            normalize_embeddings if normalize_embeddings is not None else model_config.get("normalize", True)
        )
        self.prefix = prefix

        resolved_device = device or model_config.get("device")

        logger.info(f"Loading embedding model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name, device=resolved_device)
        logger.info(f"Model loaded. Dimension: {self.model.get_sentence_embedding_dimension()}")

    def __call__(self, input: Documents) -> Embeddings:
        if self.prefix:
            input = [f"{self.prefix}{text}" for text in input]

        embeddings = self.model.encode(
            input,
            normalize_embeddings=self.normalize_embeddings,
            show_progress_bar=len(input) > 100,
        )
        return embeddings.tolist()

    @property
    def dimension(self) -> int:
        return self.model.get_sentence_embedding_dimension()


class E5EmbeddingFunction(LocalEmbeddingFunction):
    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
        normalize_embeddings: bool | None = None,
    ) -> None:
        if model_name is None:
            model_name = "multilingual-e5-large"

        super().__init__(
            model_name=model_name,
            device=device,
            normalize_embeddings=normalize_embeddings,
            prefix="",
        )

        if self.model_type != "e5":
            logger.warning(f"Model {model_name} is not E5 type. Query/passage prefixes may not be optimal.")

    def __call__(self, input: Documents) -> Embeddings:
        return self.embed_documents(input)

    def embed_documents(self, documents: Documents) -> Embeddings:
        prefixed = [f"passage: {doc}" for doc in documents]
        embeddings = self.model.encode(
            prefixed,
            normalize_embeddings=self.normalize_embeddings,
            show_progress_bar=len(documents) > 100,
        )
        return embeddings.tolist()

    def embed_query(self, query: str) -> list[float]:
        prefixed = f"query: {query}"
        embedding = self.model.encode(
            prefixed,
            normalize_embeddings=self.normalize_embeddings,
        )
        return embedding.tolist()

    def embed_queries(self, queries: list[str]) -> Embeddings:
        prefixed = [f"query: {q}" for q in queries]
        embeddings = self.model.encode(
            prefixed,
            normalize_embeddings=self.normalize_embeddings,
            show_progress_bar=len(queries) > 100,
        )
        return embeddings.tolist()
