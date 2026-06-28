"""
embedder.py — Generación de embeddings semánticos con sentence-transformers y GPU

Responsabilidades:
  - Cargar el modelo de embeddings (all-MiniLM-L6-v2) en GPU/CPU
  - Generar embeddings en batch para eficiencia
  - Proveer utilidades de similitud coseno
"""

from __future__ import annotations

import numpy as np
import torch
from rich.console import Console
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from src.config import Config

console = Console()


class Embedder:
    """
    Generador de embeddings semánticos usando sentence-transformers.

    Usa GPU automáticamente si está disponible (CUDA), con fallback a CPU.
    Implementa batch encoding para eficiencia en datasets grandes.

    Atributos:
        model_name: Nombre del modelo de sentence-transformers.
        device: Dispositivo usado ('cuda' o 'cpu').

    Ejemplo:
        embedder = Embedder()
        embeddings = embedder.encode(["texto 1", "texto 2"])
        # embeddings.shape == (2, 384)  # para MiniLM
    """

    def __init__(self) -> None:
        self.cfg = Config.get_instance()
        self._model: SentenceTransformer | None = None
        self._determine_device()

    def _determine_device(self) -> None:
        """Determina si usar GPU o CPU."""
        requested = self.cfg.embedding_device.lower()

        if requested == "cuda" and torch.cuda.is_available():
            self.device = "cuda"
            gpu_name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / 1e9
            console.print(
                f"[green]✓ GPU detectada: {gpu_name} ({vram:.1f} GB VRAM)[/green]"
            )
        elif requested == "cuda":
            console.print("[yellow]⚠ CUDA solicitado pero no disponible. Usando CPU.[/yellow]")
            self.device = "cpu"
        else:
            self.device = "cpu"
            console.print("[cyan]Usando CPU para embeddings.[/cyan]")

    @property
    def model(self) -> SentenceTransformer:
        """Lazy loading del modelo (carga solo cuando se necesita)."""
        if self._model is None:
            console.print(
                f"[cyan]Cargando modelo de embeddings: "
                f"[bold]{self.cfg.embedding_model}[/bold] en {self.device}...[/cyan]"
            )
            self._model = SentenceTransformer(
                self.cfg.embedding_model,
                device=self.device,
            )
            dim = self._model.get_sentence_embedding_dimension()
            console.print(
                f"[green]✓ Modelo cargado. Dimensión de embeddings: {dim}[/green]"
            )
        return self._model

    def encode(
        self,
        texts: list[str],
        batch_size: int | None = None,
        show_progress: bool = True,
        normalize: bool = True,
    ) -> np.ndarray:
        """
        Genera embeddings para una lista de textos.

        Args:
            texts: Lista de strings a vectorizar.
            batch_size: Tamaño del batch (default: cfg.batch_size).
            show_progress: Mostrar barra de progreso.
            normalize: Normalizar vectores a norma unitaria (recomendado para cosine similarity).

        Returns:
            Array numpy de shape (n_textos, embedding_dim).
        """
        if not texts:
            return np.array([])

        bs = batch_size or self.cfg.batch_size

        embeddings = self.model.encode(
            texts,
            batch_size=bs,
            show_progress_bar=show_progress,
            normalize_embeddings=normalize,
            convert_to_numpy=True,
        )
        return embeddings

    def encode_single(self, text: str) -> np.ndarray:
        """
        Genera el embedding de un solo texto.

        Args:
            text: Texto a vectorizar.

        Returns:
            Array numpy de shape (embedding_dim,).
        """
        return self.encode([text], show_progress=False)[0]

    def cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Calcula la similitud coseno entre dos vectores.

        Args:
            vec1: Primer vector.
            vec2: Segundo vector.

        Returns:
            Similitud coseno en [-1, 1]. Con normalización, rango es [0, 1].
        """
        if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
            return 0.0
        return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))

    def batch_cosine_similarity(
        self, query_vec: np.ndarray, corpus_vecs: np.ndarray
    ) -> np.ndarray:
        """
        Calcula la similitud coseno entre un vector query y un corpus.

        Args:
            query_vec: Vector de consulta, shape (dim,).
            corpus_vecs: Matriz del corpus, shape (n, dim).

        Returns:
            Array de similitudes, shape (n,).
        """
        # Si los vectores están normalizados, dot product == cosine similarity
        return corpus_vecs @ query_vec

    @property
    def embedding_dim(self) -> int:
        """Dimensión del espacio de embeddings."""
        return self.model.get_sentence_embedding_dimension()


if __name__ == "__main__":
    embedder = Embedder()
    print(f"Device: {embedder.device}")
    print(f"Embedding dim: {embedder.embedding_dim}")

    texts = [
        "Scientists discover new planet in distant galaxy.",
        "New planet found by astronomers using Hubble telescope.",
        "Stock market hits record high amid economic uncertainty.",
    ]
    embs = embedder.encode(texts, show_progress=True)
    print(f"\nShape embeddings: {embs.shape}")

    # Similitud: los primeros dos deberían ser más similares entre sí
    sim_12 = embedder.cosine_similarity(embs[0], embs[1])
    sim_13 = embedder.cosine_similarity(embs[0], embs[2])
    print(f"\nSimilitud 'planeta' vs 'planeta': {sim_12:.4f}")
    print(f"Similitud 'planeta' vs 'mercado': {sim_13:.4f}")
    assert sim_12 > sim_13, "Los textos similares deberían tener mayor cosine similarity"
    print("✓ Similitud coseno funciona correctamente")
