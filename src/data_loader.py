"""
data_loader.py — Carga y exploración del dataset BBC News Summary

Responsabilidades:
  - Leer los artículos y resúmenes desde la estructura de carpetas de Kaggle
  - Construir un DataFrame consolidado con columnas estandarizadas
  - Proveer estadísticas básicas para la exploración inicial
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
from rich.console import Console
from rich.table import Table

from src.config import Config

console = Console()


class DataLoader:
    """
    Carga el dataset BBC News Summary desde el sistema de archivos.

    El dataset de Kaggle tiene esta estructura:
        BBC News Summary/
        ├── News Articles/
        │   ├── business/
        │   ├── entertainment/
        │   ├── politics/
        │   ├── sport/
        │   └── tech/
        └── Summaries/
            ├── business/
            ├── entertainment/
            ├── politics/
            ├── sport/
            └── tech/

    El CSV consolidado (si existe) tiene columnas:
        category, filename, article, summary, original_title,
        article_len, summary_len, title_len
    """

    CATEGORIES = ["business", "entertainment", "politics", "sport", "tech"]

    def __init__(self) -> None:
        self.cfg = Config.get_instance()
        self._df: pd.DataFrame | None = None

    # ------------------------------------------------------------------
    # Carga principal
    # ------------------------------------------------------------------

    def load(self, force_reload: bool = False) -> pd.DataFrame:
        """
        Carga el dataset. Si ya fue cargado, retorna el cache (salvo force_reload).

        Args:
            force_reload: Si True, recarga desde disco ignorando el cache.

        Returns:
            DataFrame consolidado con todos los artículos.
        """
        if self._df is not None and not force_reload:
            return self._df

        parquet_cache = self.cfg.outputs_path / "dataset_cache.parquet"
        csv_cache = self.cfg.outputs_path / "dataset_cache.csv"

        # 1. Intentar desde cache Parquet (rápido) o CSV (fallback)
        if not force_reload:
            if parquet_cache.exists():
                try:
                    console.print("[cyan]Cargando dataset desde cache Parquet...[/cyan]")
                    self._df = pd.read_parquet(parquet_cache)
                    console.print(f"[green]✓ Dataset cargado: {len(self._df):,} artículos[/green]")
                    return self._df
                except Exception:
                    console.print("[yellow]Cache Parquet no disponible, intentando CSV...[/yellow]")
            if csv_cache.exists():
                console.print("[cyan]Cargando dataset desde cache CSV...[/cyan]")
                self._df = pd.read_csv(csv_cache)
                # Deserializar columnas de listas guardadas como strings
                if "clean_tokens" in self._df.columns:
                    import ast
                    self._df["clean_tokens"] = self._df["clean_tokens"].apply(
                        lambda x: ast.literal_eval(x) if isinstance(x, str) else []
                    )
                console.print(f"[green]✓ Dataset cargado: {len(self._df):,} artículos[/green]")
                return self._df

        # 2. Cargar desde estructura de carpetas
        self._df = self._load_from_folders()

        # 3. Guardar cache (Parquet preferido, CSV como fallback)
        try:
            self._df.to_parquet(parquet_cache, index=False)
            console.print(f"[green]✓ Cache Parquet guardado en {parquet_cache}[/green]")
        except ImportError:
            self._df.to_csv(csv_cache, index=False)
            console.print(f"[yellow]pyarrow no disponible — cache guardado como CSV en {csv_cache}[/yellow]")
            console.print("[yellow]Instalá pyarrow para mejor rendimiento: pip install pyarrow[/yellow]")

        return self._df

    def _load_from_folders(self) -> pd.DataFrame:
        """Lee artículos y resúmenes desde la estructura de carpetas de Kaggle."""
        base = self.cfg.dataset_path
        articles_base = base / "BBC News Summary" / "News Articles"
        summaries_base = base / "BBC News Summary" / "Summaries"

        if not articles_base.exists():
            raise FileNotFoundError(
                f"No se encontró la carpeta de artículos en:\n{articles_base}\n"
                "Asegurate de haber ejecutado la celda de descarga del dataset."
            )

        records = []
        for category in self.CATEGORIES:
            art_dir = articles_base / category
            sum_dir = summaries_base / category

            if not art_dir.exists():
                console.print(f"[yellow]⚠ Categoría no encontrada: {category}[/yellow]")
                continue

            for art_file in sorted(art_dir.glob("*.txt")):
                filename = art_file.stem
                sum_file = sum_dir / art_file.name

                article_text = self._read_file(art_file)
                summary_text = self._read_file(sum_file) if sum_file.exists() else ""

                # El título original suele ser la primera línea del artículo
                original_title = self._extract_title(article_text)

                records.append({
                    "category": category,
                    "filename": filename,
                    "article": article_text,
                    "summary": summary_text,
                    "original_title": original_title,
                })

        df = pd.DataFrame(records)

        # Feature engineering básico
        df["article_len"] = df["article"].str.len()
        df["summary_len"] = df["summary"].str.len()
        df["title_len"] = df["original_title"].str.len()
        df["article_word_count"] = df["article"].str.split().str.len()
        df["summary_word_count"] = df["summary"].str.split().str.len()

        console.print(
            f"[green]✓ Dataset cargado desde carpetas: {len(df):,} artículos "
            f"en {df['category'].nunique()} categorías[/green]"
        )
        return df

    @staticmethod
    def _read_file(path: Path) -> str:
        """Lee un archivo de texto con fallback de encoding."""
        for encoding in ["utf-8", "latin-1", "cp1252"]:
            try:
                return path.read_text(encoding=encoding).strip()
            except UnicodeDecodeError:
                continue
        return ""

    @staticmethod
    def _extract_title(text: str) -> str:
        """Extrae el título (primera línea no vacía del artículo)."""
        for line in text.splitlines():
            line = line.strip()
            if line:
                return line
        return ""

    # ------------------------------------------------------------------
    # Exploración
    # ------------------------------------------------------------------

    def describe(self) -> None:
        """Imprime un resumen detallado del dataset en formato tabla."""
        df = self.load()

        console.print("\n[bold blue]═══ Descripción del Dataset BBC News Summary ═══[/bold blue]\n")

        # Info general
        table = Table(title="Resumen General", show_header=True, header_style="bold magenta")
        table.add_column("Métrica", style="cyan")
        table.add_column("Valor", style="green")

        table.add_row("Total de artículos", f"{len(df):,}")
        table.add_row("Categorías", str(df["category"].nunique()))
        table.add_row("Columnas", str(len(df.columns)))
        table.add_row("Valores nulos (article)", str(df["article"].isna().sum()))
        table.add_row("Valores nulos (summary)", str(df["summary"].isna().sum()))
        table.add_row("Long. promedio artículo (chars)", f"{df['article_len'].mean():.0f}")
        table.add_row("Long. promedio resumen (chars)", f"{df['summary_len'].mean():.0f}")
        table.add_row("Long. promedio título (chars)", f"{df['title_len'].mean():.0f}")
        console.print(table)

        # Por categoría
        cat_table = Table(title="\nDistribución por Categoría", show_header=True, header_style="bold magenta")
        cat_table.add_column("Categoría", style="cyan")
        cat_table.add_column("Artículos", style="green")
        cat_table.add_column("% Total", style="yellow")
        cat_table.add_column("Palabras promedio (artículo)", style="blue")

        for cat, grp in df.groupby("category"):
            pct = len(grp) / len(df) * 100
            cat_table.add_row(
                str(cat),
                str(len(grp)),
                f"{pct:.1f}%",
                f"{grp['article_word_count'].mean():.0f}",
            )
        console.print(cat_table)

    def get_sample(self, n: int = 5, category: str | None = None, seed: int = 42) -> pd.DataFrame:
        """Retorna una muestra aleatoria del dataset."""
        df = self.load()
        if category:
            df = df[df["category"] == category]
        return df.sample(n=min(n, len(df)), random_state=seed)

    @property
    def df(self) -> pd.DataFrame:
        """Acceso directo al DataFrame (carga si es necesario)."""
        return self.load()


if __name__ == "__main__":
    loader = DataLoader()
    loader.describe()
    print(loader.get_sample(3)[["category", "original_title"]].to_string())
