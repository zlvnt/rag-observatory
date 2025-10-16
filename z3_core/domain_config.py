# Domain configuration management for RAG Observatory.

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import yaml


@dataclass
class DomainConfig:
    """Configuration for a specific domain.

    Attributes:
        domain_name: Name of the domain (e.g., "tokopedia", "coffeeshop")
        knowledge_base_dir: Directory containing domain documents
        vector_store_dir: Directory where FAISS vector store is/will be saved
        personality_config_path: Path to personality JSON config (optional)
        supervisor_prompt_path: Path to supervisor routing prompt (optional)
        embedding_model: HuggingFace model name for embeddings
        llm_model: Gemini model name for text generation
        llm_temperature: Temperature for LLM generation
        relevance_threshold: Minimum relevance score for RAG filtering
        chunk_size: Size of document chunks for splitting
        chunk_overlap: Overlap between document chunks
        retrieval_k: Number of documents to retrieve
    """

    domain_name: str
    knowledge_base_dir: Path
    vector_store_dir: Path
    golden_dataset: Optional[Path] = None
    personality_config_path: Optional[Path] = None
    supervisor_prompt_path: Optional[Path] = None
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    llm_model: str = "gemini-pro"
    llm_temperature: float = 0.7
    relevance_threshold: float = 0.8
    chunk_size: int = 700
    chunk_overlap: int = 100
    retrieval_k: int = 4

    @classmethod
    def from_yaml(cls, config_path: Path) -> "DomainConfig":
        # Load domain configuration from YAML file.

        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if not data:
            raise ValueError(f"Empty configuration file: {config_path}")

        # Validate required fields
        required_fields = ["domain_name", "knowledge_base_dir", "vector_store_dir"]
        missing = [f for f in required_fields if f not in data]
        if missing:
            raise ValueError(f"Missing required fields in {config_path}: {missing}")

        # Convert string paths to Path objects
        data["knowledge_base_dir"] = Path(data["knowledge_base_dir"])
        data["vector_store_dir"] = Path(data["vector_store_dir"])

        # Convert optional path fields
        if "golden_dataset" in data and data["golden_dataset"]:
            data["golden_dataset"] = Path(data["golden_dataset"])

        if "personality_config_path" in data and data["personality_config_path"]:
            data["personality_config_path"] = Path(data["personality_config_path"])

        if "supervisor_prompt_path" in data and data["supervisor_prompt_path"]:
            data["supervisor_prompt_path"] = Path(data["supervisor_prompt_path"])

        return cls(**data)

    def validate_paths(self) -> list[str]:
        # Validate that required paths exist.

        errors = []

        if not self.knowledge_base_dir.exists():
            errors.append(f"Knowledge base directory not found: {self.knowledge_base_dir}")

        if self.personality_config_path and not self.personality_config_path.exists():
            errors.append(f"Personality config not found: {self.personality_config_path}")

        if self.supervisor_prompt_path and not self.supervisor_prompt_path.exists():
            errors.append(f"Supervisor prompt not found: {self.supervisor_prompt_path}")

        return errors

    def to_dict(self) -> dict:
        # Convert config to dictionary with string paths.

        return {
            "domain_name": self.domain_name,
            "knowledge_base_dir": str(self.knowledge_base_dir),
            "vector_store_dir": str(self.vector_store_dir),
            "personality_config_path": str(self.personality_config_path) if self.personality_config_path else None,
            "supervisor_prompt_path": str(self.supervisor_prompt_path) if self.supervisor_prompt_path else None,
            "embedding_model": self.embedding_model,
            "llm_model": self.llm_model,
            "llm_temperature": self.llm_temperature,
            "relevance_threshold": self.relevance_threshold,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "retrieval_k": self.retrieval_k,
        }


def load_domain_config(domain_name: str, config_dir: Path = Path("configs")) -> DomainConfig:
    """Load domain configuration by name.
    Looks for a YAML file named {domain_name}_config.yaml in the config directory.
    """
    config_path = config_dir / f"{domain_name}_config.yaml"
    if not config_path.exists():
        config_path = config_dir / f"{domain_name}.yaml"
        if not config_path.exists():
            raise FileNotFoundError(
                f"Configuration file not found: {config_path}\n"
                f"Please create a config file for domain '{domain_name}' at {config_path}"
            )

    return DomainConfig.from_yaml(config_path)


def list_available_domains(config_dir: Path = Path("configs")) -> list[str]:
   # List all available domain configurations.

    if not config_dir.exists():
        return []

    domain_names = []
    for config_file in config_dir.glob("*_config.yaml"):
        # Extract domain name from filename (remove _config.yaml)
        domain_name = config_file.stem.replace("_config", "")
        domain_names.append(domain_name)

    return sorted(domain_names)
