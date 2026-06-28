"""
config.py — Configuración centralizada del proyecto (Patrón Singleton)

Carga todas las variables de entorno desde .env y expone una única
instancia de configuración reutilizable en todos los módulos.
"""

from __future__ import annotations

import os
from pathlib import Path
from dotenv import load_dotenv


class Config:
    """
    Singleton de configuración del proyecto.

    Carga variables de entorno desde .env y provee acceso tipado
    a todos los parámetros del pipeline.

    Uso:
        cfg = Config.get_instance()
        print(cfg.gemini_model)
    """

    _instance: Config | None = None

    def __new__(cls) -> "Config":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return
        self._initialized = True
        self._load()

    def _load(self) -> None:
        """Carga y valida todas las variables de entorno."""
        # Buscar .env desde el directorio raíz del proyecto
        root = Path(__file__).parent.parent
        env_path = root / ".env"
        load_dotenv(dotenv_path=env_path, override=True)

        # --- API Keys ---
        self.gemini_api_key: str = self._require("GEMINI_API_KEY")
        self.kaggle_username: str = os.getenv("KAGGLE_USERNAME", "")
        self.kaggle_key: str = os.getenv("KAGGLE_KEY", "")

        # --- Rutas ---
        self.root_path: Path = root
        self.dataset_path: Path = root / os.getenv("DATASET_PATH", "data/bbc-news-summary")
        self.chroma_db_path: Path = root / os.getenv("CHROMA_DB_PATH", "chroma_db")
        self.outputs_path: Path = root / os.getenv("OUTPUTS_PATH", "outputs")
        self.figures_path: Path = self.outputs_path / "figures"

        # Crear directorios si no existen
        for path in [self.dataset_path, self.chroma_db_path, self.outputs_path, self.figures_path]:
            path.mkdir(parents=True, exist_ok=True)

        # --- Modelo Gemini ---
        self.gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
        self.gemini_max_tokens: int = int(os.getenv("GEMINI_MAX_TOKENS", "200"))
        self.gemini_temperature: float = float(os.getenv("GEMINI_TEMPERATURE", "0.7"))

        # --- Embeddings ---
        self.embedding_model: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        self.embedding_device: str = os.getenv("EMBEDDING_DEVICE", "cuda")

        # --- Procesamiento ---
        self.summary_sentences: int = int(os.getenv("SUMMARY_SENTENCES", "3"))
        self.batch_size: int = int(os.getenv("BATCH_SIZE", "32"))

        # --- ChromaDB ---
        self.chroma_collection_name: str = "bbc_news_articles"

    @classmethod
    def get_instance(cls) -> "Config":
        """Retorna la única instancia de configuración (Singleton)."""
        return cls()

    def _require(self, key: str) -> str:
        """Obtiene una variable de entorno requerida o lanza error."""
        value = os.getenv(key)
        if not value:
            raise EnvironmentError(
                f"Variable de entorno requerida no encontrada: '{key}'\n"
                f"Verificá que el archivo .env existe y contiene {key}=<valor>"
            )
        return value

    def __repr__(self) -> str:
        return (
            f"Config(\n"
            f"  gemini_model={self.gemini_model!r},\n"
            f"  embedding_model={self.embedding_model!r},\n"
            f"  embedding_device={self.embedding_device!r},\n"
            f"  dataset_path={self.dataset_path},\n"
            f"  summary_sentences={self.summary_sentences},\n"
            f"  batch_size={self.batch_size}\n"
            f")"
        )


if __name__ == "__main__":
    cfg = Config.get_instance()
    print(cfg)
    # Verificar Singleton
    cfg2 = Config.get_instance()
    assert cfg is cfg2, "Singleton violado!"
    print("✓ Singleton verificado: misma instancia")
