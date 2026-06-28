"""
preprocessor.py — Preprocesamiento de texto para el pipeline NLP

Responsabilidades:
  - Limpieza: lowercasing, puntuación, caracteres especiales
  - Tokenización con NLTK
  - Eliminación de stopwords
  - Lematización con WordNetLemmatizer
  - Extracción de las primeras N oraciones como input del modelo
  - Análisis de frecuencia de palabras clave
"""

from __future__ import annotations

import re
import string
from collections import Counter
from functools import lru_cache
from typing import List

import nltk
import pandas as pd
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize
from rich.console import Console
from tqdm import tqdm

from src.config import Config

console = Console()

# Descargar recursos NLTK necesarios (solo si no están presentes)
def _download_nltk_resources() -> None:
    resources = [
        ("tokenizers/punkt", "punkt"),
        ("tokenizers/punkt_tab", "punkt_tab"),
        ("corpora/stopwords", "stopwords"),
        ("corpora/wordnet", "wordnet"),
        ("taggers/averaged_perceptron_tagger", "averaged_perceptron_tagger"),
        ("taggers/averaged_perceptron_tagger_eng", "averaged_perceptron_tagger_eng"),
    ]
    for path, name in resources:
        try:
            nltk.data.find(path)
        except LookupError:
            console.print(f"[yellow]Descargando recurso NLTK: {name}...[/yellow]")
            nltk.download(name, quiet=True)


_download_nltk_resources()


class Preprocessor:
    """
    Pipeline de preprocesamiento de texto para artículos de noticias.

    Aplica las siguientes transformaciones en orden:
        1. Extracción de N oraciones iniciales (resumen de input)
        2. Lowercasing
        3. Remoción de URLs, emails, caracteres especiales
        4. Tokenización por palabra
        5. Remoción de stopwords
        6. Lematización

    Ejemplo:
        pp = Preprocessor()
        df = pp.fit_transform(df)
        # df tiene nuevas columnas: input_summary, clean_tokens, clean_text
    """

    def __init__(self) -> None:
        self.cfg = Config.get_instance()
        self._lemmatizer = WordNetLemmatizer()
        self._stop_words = set(stopwords.words("english"))
        self._summary_sentences = self.cfg.summary_sentences

    # ------------------------------------------------------------------
    # Pipeline principal
    # ------------------------------------------------------------------

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aplica el pipeline completo de preprocesamiento al DataFrame.

        Agrega columnas:
            - input_summary: primeras N oraciones (input al modelo)
            - clean_tokens: lista de tokens limpios y lematizados
            - clean_text: tokens limpios como string (para WordCloud etc.)
            - keyword_freq: dict con frecuencia de palabras (top 20)

        Args:
            df: DataFrame con columna 'article'

        Returns:
            DataFrame con columnas adicionales.
        """
        console.print(f"[cyan]Preprocesando {len(df):,} artículos...[/cyan]")

        tqdm.pandas(desc="Extrayendo resúmenes")
        df = df.copy()
        df["input_summary"] = df["article"].progress_apply(self.extract_summary)

        tqdm.pandas(desc="Limpiando y tokenizando")
        df["clean_tokens"] = df["article"].progress_apply(self.tokenize_and_clean)

        df["clean_text"] = df["clean_tokens"].apply(lambda tokens: " ".join(tokens))

        console.print("[green]✓ Preprocesamiento completado[/green]")
        return df

    # ------------------------------------------------------------------
    # Métodos individuales
    # ------------------------------------------------------------------

    def extract_summary(self, text: str) -> str:
        """
        Extrae las primeras N oraciones del texto como resumen de input.

        Args:
            text: Texto completo del artículo.

        Returns:
            String con las primeras N oraciones.
        """
        if not text or not isinstance(text, str):
            return ""
        sentences = sent_tokenize(text)
        # Saltar la primera línea si es el título (muy corta)
        if sentences and len(sentences[0].split()) <= 8:
            sentences = sentences[1:]
        selected = sentences[: self._summary_sentences]
        return " ".join(selected).strip()

    def tokenize_and_clean(self, text: str) -> List[str]:
        """
        Limpia, tokeniza y lematiza el texto.

        Pasos:
            1. Lowercase
            2. Remoción de URLs y emails
            3. Remoción de puntuación y dígitos
            4. Tokenización NLTK
            5. Remoción de stopwords y tokens cortos
            6. Lematización

        Args:
            text: Texto a procesar.

        Returns:
            Lista de tokens limpios y lematizados.
        """
        if not text or not isinstance(text, str):
            return []

        # 1. Lowercase
        text = text.lower()

        # 2. Remover URLs y emails
        text = re.sub(r"https?://\S+|www\.\S+", "", text)
        text = re.sub(r"\S+@\S+", "", text)

        # 3. Remover puntuación y dígitos
        text = re.sub(r"[^\w\s]", " ", text)
        text = re.sub(r"\d+", "", text)

        # 4. Tokenizar
        tokens = word_tokenize(text)

        # 5. Remover stopwords y tokens cortos (< 3 chars)
        tokens = [
            t for t in tokens
            if t not in self._stop_words and len(t) > 2
        ]

        # 6. Lematizar
        tokens = [self._lemmatizer.lemmatize(t) for t in tokens]

        return tokens

    def get_keyword_frequencies(
        self,
        df: pd.DataFrame,
        top_n: int = 20,
        by_category: bool = False,
    ) -> dict[str, Counter]:
        """
        Calcula la frecuencia de palabras clave.

        Args:
            df: DataFrame con columna 'clean_tokens'.
            top_n: Número de palabras top a retornar.
            by_category: Si True, retorna frecuencias por categoría.

        Returns:
            Dict con categoría (o 'global') como clave y Counter como valor.
        """
        if "clean_tokens" not in df.columns:
            raise ValueError("El DataFrame debe tener la columna 'clean_tokens'. Ejecutá fit_transform primero.")

        result: dict[str, Counter] = {}

        if by_category and "category" in df.columns:
            for cat, grp in df.groupby("category"):
                all_tokens = [t for tokens in grp["clean_tokens"] for t in tokens]
                result[str(cat)] = Counter(all_tokens).most_common(top_n)
        else:
            all_tokens = [t for tokens in df["clean_tokens"] for t in tokens]
            result["global"] = Counter(all_tokens).most_common(top_n)

        return result

    def clean_title(self, title: str) -> str:
        """Limpia un título para comparación (ROUGE)."""
        return " ".join(self.tokenize_and_clean(title))


if __name__ == "__main__":
    # Test rápido
    sample_text = """
    Scientists discover new planet in distant galaxy.
    The discovery was made by astronomers using the Hubble Space Telescope.
    The planet, named Kepler-452b, orbits a star similar to our Sun.
    Researchers believe the planet could potentially support life.
    Further studies are planned for next year.
    """
    pp = Preprocessor()
    summary = pp.extract_summary(sample_text)
    tokens = pp.tokenize_and_clean(sample_text)
    print(f"Input summary ({len(summary.split())} palabras):")
    print(summary)
    print(f"\nTokens limpios ({len(tokens)} tokens):")
    print(tokens[:15])
