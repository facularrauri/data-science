"""
vector_store.py — Base vectorial con ChromaDB (Patrón Repository)

Responsabilidades:
  - Indexar artículos + embeddings en ChromaDB
  - Búsqueda semántica por similitud
  - Filtrado por metadatos (categoría, filename) con API tipo MongoDB
  - Persistencia en disco (sin servidor, sin Docker)

ChromaDB API:
    # Filtro simple (tipo MongoDB / SQL)
    results = store.query_similar("text", where={"category": "sport"})

    # Múltiples filtros
    results = store.query_similar("text", where={"$and": [
        {"category": "tech"},
        {"article_len": {"$gt": 500}}
    ]})
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import chromadb
import numpy as np
import pandas as pd
from chromadb.config import Settings
from rich.console import Console
from tqdm import tqdm

from src.config import Config
from src.embedder import Embedder

console = Console()


class VectorStore:
    """
    Repositorio vectorial sobre ChromaDB embedded (sin servidor, sin Docker).

    Persiste en disco en la ruta cfg.chroma_db_path. Usa el patrón Repository
    para abstraer las operaciones CRUD sobre la base vectorial.

    Métodos principales:
        - index_dataframe(): Indexa todos los artículos del DataFrame
        - query_similar(): Busca artículos similares a una query
        - filter_by_category(): Filtra artículos por categoría
        - get_by_filename(): Obtiene un artículo por nombre de archivo

    Ejemplo:
        store = VectorStore()
        store.index_dataframe(df)
        results = store.query_similar("economy financial crisis", n_results=5)
        sport_results = store.filter_by_category("sport", n_results=10)
    """

    def __init__(self, embedder: Embedder | None = None) -> None:
        self.cfg = Config.get_instance()
        self.embedder = embedder or Embedder()

        # Cliente ChromaDB en modo persistente (embedded)
        self._client = chromadb.PersistentClient(
            path=str(self.cfg.chroma_db_path),
        )
        self._collection = self._get_or_create_collection()

    def _get_or_create_collection(self) -> chromadb.Collection:
        """Obtiene o crea la colección principal."""
        collection = self._client.get_or_create_collection(
            name=self.cfg.chroma_collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        count = collection.count()
        if count > 0:
            console.print(
                f"[cyan]ChromaDB: Colección '{self.cfg.chroma_collection_name}' "
                f"cargada con {count:,} documentos[/cyan]"
            )
        else:
            console.print(
                f"[cyan]ChromaDB: Colección '{self.cfg.chroma_collection_name}' "
                f"creada (vacía)[/cyan]"
            )
        return collection

    # ------------------------------------------------------------------
    # Indexado
    # ------------------------------------------------------------------

    def index_dataframe(
        self,
        df: pd.DataFrame,
        text_column: str = "input_summary",
        batch_size: int = 100,
        force_reindex: bool = False,
    ) -> None:
        """
        Indexa todos los artículos del DataFrame en ChromaDB.

        Args:
            df: DataFrame con artículos preprocesados.
            text_column: Columna de texto a vectorizar (default: 'input_summary').
            batch_size: Tamaño del batch para inserción.
            force_reindex: Si True, borra y re-indexa todo.
        """
        if force_reindex:
            console.print("[yellow]Re-indexando: borrando colección existente...[/yellow]")
            self._client.delete_collection(self.cfg.chroma_collection_name)
            self._collection = self._get_or_create_collection()

        existing_count = self._collection.count()
        if existing_count >= len(df) and not force_reindex:
            console.print(
                f"[green]✓ ChromaDB ya tiene {existing_count:,} documentos indexados. "
                f"Usá force_reindex=True para re-indexar.[/green]"
            )
            return

        console.print(f"[cyan]Generando embeddings para {len(df):,} artículos...[/cyan]")
        texts = df[text_column].fillna("").tolist()
        embeddings = self.embedder.encode(texts, show_progress=True)

        console.print("[cyan]Indexando en ChromaDB...[/cyan]")
        total_batches = (len(df) + batch_size - 1) // batch_size

        for i in tqdm(range(0, len(df), batch_size), total=total_batches, desc="Indexando batches"):
            batch_df = df.iloc[i : i + batch_size]
            batch_embs = embeddings[i : i + batch_size].tolist()

            ids = [f"{row['category']}_{row['filename']}" for _, row in batch_df.iterrows()]
            documents = batch_df[text_column].fillna("").tolist()
            metadatas = [
                {
                    "category": row["category"],
                    "filename": row["filename"],
                    "original_title": row.get("original_title", ""),
                    "article_len": int(row.get("article_len", 0)),
                    "summary_len": int(row.get("summary_len", 0)),
                }
                for _, row in batch_df.iterrows()
            ]

            self._collection.upsert(
                ids=ids,
                embeddings=batch_embs,
                documents=documents,
                metadatas=metadatas,
            )

        console.print(f"[green]✓ {len(df):,} artículos indexados en ChromaDB[/green]")

    # ------------------------------------------------------------------
    # Consultas
    # ------------------------------------------------------------------

    def query_similar(
        self,
        query_text: str,
        n_results: int = 5,
        where: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        """
        Busca los n_results artículos más similares semánticamente a query_text.

        Soporta filtros tipo MongoDB en el parámetro where:
            where={"category": "sport"}
            where={"$and": [{"category": "tech"}, {"article_len": {"$gt": 500}}]}

        Args:
            query_text: Texto de consulta.
            n_results: Número de resultados a retornar.
            where: Filtro de metadatos (opcional).

        Returns:
            DataFrame con columnas: id, document, distance, category, filename, original_title.
        """
        query_emb = self.embedder.encode_single(query_text).tolist()

        kwargs: dict[str, Any] = {
            "query_embeddings": [query_emb],
            "n_results": min(n_results, self._collection.count()),
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where

        results = self._collection.query(**kwargs)

        rows = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            rows.append({
                "document": doc,
                "distance": dist,
                "similarity": 1 - dist,  # ChromaDB cosine distance = 1 - similarity
                **meta,
            })

        return pd.DataFrame(rows)

    def filter_by_category(
        self,
        category: str,
        n_results: int = 10,
    ) -> pd.DataFrame:
        """
        Retorna artículos de una categoría específica.

        API similar a MongoDB: filtra por metadato 'category'.

        Args:
            category: Categoría (business, entertainment, politics, sport, tech).
            n_results: Máximo de resultados.

        Returns:
            DataFrame con los artículos de la categoría.
        """
        results = self._collection.get(
            where={"category": category},
            limit=n_results,
            include=["documents", "metadatas"],
        )

        rows = []
        for doc, meta in zip(results["documents"], results["metadatas"]):
            rows.append({"document": doc, **meta})

        return pd.DataFrame(rows)

    def get_by_filename(self, category: str, filename: str) -> dict[str, Any] | None:
        """
        Obtiene un artículo por su ID (categoría + filename).

        Args:
            category: Categoría del artículo.
            filename: Nombre del archivo (sin extensión).

        Returns:
            Dict con el artículo, o None si no existe.
        """
        doc_id = f"{category}_{filename}"
        result = self._collection.get(
            ids=[doc_id],
            include=["documents", "metadatas"],
        )
        if not result["ids"]:
            return None
        return {
            "id": doc_id,
            "document": result["documents"][0],
            **result["metadatas"][0],
        }

    @property
    def count(self) -> int:
        """Número de documentos indexados."""
        return self._collection.count()

    def reset(self) -> None:
        """Borra todos los documentos de la colección."""
        self._client.delete_collection(self.cfg.chroma_collection_name)
        self._collection = self._get_or_create_collection()
        console.print("[yellow]⚠ Colección ChromaDB reseteada[/yellow]")


if __name__ == "__main__":
    # Test básico (requiere que el dataset esté cargado)
    store = VectorStore()
    print(f"Documentos indexados: {store.count}")

    if store.count > 0:
        results = store.query_similar("technology innovation startup", n_results=3)
        print("\nResultados de búsqueda semántica:")
        print(results[["category", "original_title", "similarity"]].to_string())

        sport = store.filter_by_category("sport", n_results=3)
        print("\nArtículos de Sport:")
        print(sport[["category", "original_title"]].to_string())
