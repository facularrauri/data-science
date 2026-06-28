"""
visualizer.py — Visualizaciones del proyecto

Responsabilidades:
  - WordCloud por categoría y global
  - Distribución de longitudes (artículo, resumen, título)
  - Top-N palabras por categoría (gráfico de barras)
  - Comparación de métricas ROUGE por estrategia (heatmap + barras)
  - Distribución de similitud entre títulos generados y originales
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from wordcloud import WordCloud

from src.config import Config

# Estilo visual consistente
plt.style.use("dark_background")
PALETTE = ["#6C63FF", "#FF6584", "#43B89C", "#F9C74F", "#F8961E"]
CATEGORY_COLORS = {
    "business": "#6C63FF",
    "entertainment": "#FF6584",
    "politics": "#43B89C",
    "sport": "#F9C74F",
    "tech": "#F8961E",
}


class Visualizer:
    """
    Generador de visualizaciones para el análisis del dataset y resultados.

    Todas las figuras se guardan en cfg.figures_path y también se muestran
    en el notebook (plt.show()).

    Ejemplo:
        viz = Visualizer()
        viz.plot_category_distribution(df)
        viz.plot_wordcloud(df, category="sport")
        viz.plot_rouge_comparison(results_df)
    """

    def __init__(self) -> None:
        self.cfg = Config.get_instance()
        self._setup_fonts()

    def _setup_fonts(self) -> None:
        """Configura la tipografía para las visualizaciones."""
        plt.rcParams.update({
            "font.family": "DejaVu Sans",
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "figure.facecolor": "#1a1a2e",
            "axes.facecolor": "#16213e",
            "axes.edgecolor": "#444",
            "grid.color": "#333",
            "text.color": "#e0e0e0",
            "axes.labelcolor": "#e0e0e0",
            "xtick.color": "#e0e0e0",
            "ytick.color": "#e0e0e0",
        })

    def _save(self, fig: plt.Figure, filename: str) -> Path:
        """Guarda la figura en alta resolución."""
        path = self.cfg.figures_path / filename
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        return path

    # ------------------------------------------------------------------
    # 1. Distribución por categoría
    # ------------------------------------------------------------------

    def plot_category_distribution(self, df: pd.DataFrame) -> plt.Figure:
        """Gráfico de barras con cantidad de artículos por categoría."""
        counts = df["category"].value_counts()

        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.bar(
            counts.index,
            counts.values,
            color=[CATEGORY_COLORS.get(c, PALETTE[i % len(PALETTE)]) for i, c in enumerate(counts.index)],
            width=0.6,
            zorder=2,
        )

        # Etiquetas sobre las barras
        for bar, val in zip(bars, counts.values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 2,
                str(val),
                ha="center",
                va="bottom",
                fontsize=11,
                fontweight="bold",
                color="#e0e0e0",
            )

        ax.set_title("Distribución de Artículos por Categoría", pad=15, fontweight="bold")
        ax.set_xlabel("Categoría")
        ax.set_ylabel("Número de artículos")
        ax.grid(axis="y", alpha=0.3, zorder=1)
        ax.set_axisbelow(True)

        fig.tight_layout()
        self._save(fig, "category_distribution.png")
        return fig

    # ------------------------------------------------------------------
    # 2. Distribución de longitudes
    # ------------------------------------------------------------------

    def plot_length_distributions(self, df: pd.DataFrame) -> plt.Figure:
        """Histogramas de longitud de artículo, resumen y título (en palabras)."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle("Distribución de Longitudes (en palabras)", fontsize=15, fontweight="bold")

        cols = {
            "Artículos": "article_word_count",
            "Resúmenes": "summary_word_count",
            "Títulos": "title_len",
        }
        colors = [PALETTE[0], PALETTE[1], PALETTE[2]]

        for ax, (label, col), color in zip(axes, cols.items(), colors):
            if col not in df.columns:
                ax.text(0.5, 0.5, f"Sin datos:\n{col}", ha="center", va="center", transform=ax.transAxes)
                continue

            data = df[col].dropna()
            ax.hist(data, bins=30, color=color, alpha=0.85, edgecolor="none")
            ax.axvline(data.mean(), color="#ffffff", linestyle="--", linewidth=1.5, label=f"Media: {data.mean():.0f}")
            ax.axvline(data.median(), color="#F9C74F", linestyle=":", linewidth=1.5, label=f"Mediana: {data.median():.0f}")
            ax.set_title(label, fontweight="bold")
            ax.set_xlabel("Palabras")
            ax.set_ylabel("Frecuencia")
            ax.legend(fontsize=9)
            ax.grid(alpha=0.2)

        fig.tight_layout()
        self._save(fig, "length_distributions.png")
        return fig

    # ------------------------------------------------------------------
    # 3. WordCloud
    # ------------------------------------------------------------------

    def plot_wordcloud(
        self,
        df: pd.DataFrame,
        category: str | None = None,
        max_words: int = 100,
    ) -> plt.Figure:
        """
        Genera WordCloud de palabras clave.

        Args:
            df: DataFrame con columna 'clean_text'.
            category: Si se provee, filtra por categoría. None = global.
            max_words: Máximo de palabras en el cloud.
        """
        if "clean_text" not in df.columns:
            raise ValueError("El DataFrame debe tener 'clean_text'. Ejecutá Preprocessor.fit_transform() primero.")

        if category:
            sub_df = df[df["category"] == category]
            title = f"WordCloud — {category.capitalize()}"
            color = CATEGORY_COLORS.get(category, PALETTE[0])
        else:
            sub_df = df
            title = "WordCloud — Global"
            color = PALETTE[0]

        text = " ".join(sub_df["clean_text"].dropna().tolist())

        # Colormap basado en el color de la categoría
        wc = WordCloud(
            width=1200,
            height=600,
            background_color="#1a1a2e",
            max_words=max_words,
            colormap="viridis",
            prefer_horizontal=0.8,
            min_font_size=8,
            max_font_size=120,
        ).generate(text)

        fig, ax = plt.subplots(figsize=(14, 7))
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        ax.set_title(title, fontsize=16, fontweight="bold", pad=15)
        fig.patch.set_facecolor("#1a1a2e")

        suffix = f"_{category}" if category else "_global"
        self._save(fig, f"wordcloud{suffix}.png")
        return fig

    def plot_all_wordclouds(self, df: pd.DataFrame) -> None:
        """Genera WordClouds para todas las categorías."""
        categories = df["category"].unique().tolist()
        for cat in categories:
            self.plot_wordcloud(df, category=cat)
            plt.show()

    # ------------------------------------------------------------------
    # 4. Top palabras por categoría
    # ------------------------------------------------------------------

    def plot_top_keywords(
        self,
        keyword_freqs: dict,
        top_n: int = 15,
    ) -> plt.Figure:
        """
        Gráfico de barras horizontales con las top palabras por categoría.

        Args:
            keyword_freqs: Resultado de Preprocessor.get_keyword_frequencies(by_category=True).
            top_n: Número de palabras top a mostrar.
        """
        categories = [k for k in keyword_freqs.keys() if k != "global"]
        n_cats = len(categories)

        if n_cats == 0:
            return plt.figure()

        fig, axes = plt.subplots(1, n_cats, figsize=(5 * n_cats, 8))
        if n_cats == 1:
            axes = [axes]

        fig.suptitle(f"Top {top_n} Palabras Clave por Categoría", fontsize=15, fontweight="bold")

        for ax, cat in zip(axes, categories):
            freqs = keyword_freqs.get(cat, [])[:top_n]
            if not freqs:
                continue

            words, counts = zip(*freqs)
            color = CATEGORY_COLORS.get(cat, PALETTE[0])

            ax.barh(list(reversed(words)), list(reversed(counts)), color=color, alpha=0.85)
            ax.set_title(cat.capitalize(), fontweight="bold")
            ax.set_xlabel("Frecuencia")
            ax.grid(axis="x", alpha=0.2)

        fig.tight_layout()
        self._save(fig, "top_keywords_by_category.png")
        return fig

    # ------------------------------------------------------------------
    # 5. Comparación ROUGE
    # ------------------------------------------------------------------

    def plot_rouge_comparison(self, results_df: pd.DataFrame) -> plt.Figure:
        """
        Gráfico de barras agrupadas comparando ROUGE F1 por estrategia.

        Args:
            results_df: DataFrame resultado de Evaluator.evaluate().
        """
        strategies = ["formal", "impactful", "seo"]
        rouge_types = ["rouge1", "rouge2", "rougeL"]
        available_strategies = [s for s in strategies if f"rougeL_{s}_f1" in results_df.columns]

        if not available_strategies:
            raise ValueError("No se encontraron métricas ROUGE en el DataFrame.")

        # Construir datos para la figura
        data = []
        for strategy in available_strategies:
            for rouge_type in rouge_types:
                col = f"{rouge_type}_{strategy}_f1"
                if col in results_df.columns:
                    data.append({
                        "strategy": strategy,
                        "rouge_type": rouge_type.upper().replace("ROUGEL", "ROUGE-L"),
                        "f1_mean": results_df[col].mean(),
                        "f1_std": results_df[col].std(),
                    })

        plot_df = pd.DataFrame(data)

        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(rouge_types))
        width = 0.25

        for i, strategy in enumerate(available_strategies):
            sub = plot_df[plot_df["strategy"] == strategy]
            means = sub["f1_mean"].values
            stds = sub["f1_std"].values
            offset = (i - len(available_strategies) / 2 + 0.5) * width

            bars = ax.bar(
                x + offset,
                means,
                width=width * 0.9,
                label=strategy.capitalize(),
                color=PALETTE[i],
                alpha=0.85,
                zorder=2,
            )
            ax.errorbar(
                x + offset,
                means,
                yerr=stds,
                fmt="none",
                color="white",
                capsize=4,
                linewidth=1.5,
                zorder=3,
            )

        ax.set_title("Comparación de Métricas ROUGE F1 por Estrategia de Prompt", fontweight="bold", pad=15)
        ax.set_ylabel("ROUGE F1 Score")
        ax.set_xticks(x)
        ax.set_xticklabels(["ROUGE-1", "ROUGE-2", "ROUGE-L"], fontsize=12)
        ax.legend(title="Estrategia", fontsize=10)
        ax.grid(axis="y", alpha=0.3, zorder=1)
        ax.set_axisbelow(True)
        ax.set_ylim(0, min(1.0, plot_df["f1_mean"].max() * 1.4))

        fig.tight_layout()
        self._save(fig, "rouge_comparison.png")
        return fig

    def plot_rouge_heatmap(self, results_df: pd.DataFrame) -> plt.Figure:
        """
        Heatmap de ROUGE-L F1 por estrategia y categoría.

        Args:
            results_df: DataFrame resultado de Evaluator.evaluate().
        """
        strategies = ["formal", "impactful", "seo"]
        available_strategies = [s for s in strategies if f"rougeL_{s}_f1" in results_df.columns]

        if "category" not in results_df.columns or not available_strategies:
            raise ValueError("Faltan columnas 'category' o métricas ROUGE.")

        heatmap_data = {}
        for strategy in available_strategies:
            col = f"rougeL_{strategy}_f1"
            heatmap_data[strategy] = results_df.groupby("category")[col].mean()

        heatmap_df = pd.DataFrame(heatmap_data)

        fig, ax = plt.subplots(figsize=(8, 5))

        cmap = LinearSegmentedColormap.from_list("custom", ["#16213e", "#6C63FF", "#43B89C"])
        sns.heatmap(
            heatmap_df,
            annot=True,
            fmt=".4f",
            cmap=cmap,
            ax=ax,
            linewidths=0.5,
            linecolor="#1a1a2e",
            annot_kws={"size": 10},
            cbar_kws={"label": "ROUGE-L F1"},
        )

        ax.set_title("ROUGE-L F1 por Categoría y Estrategia de Prompt", fontweight="bold", pad=15)
        ax.set_xlabel("Estrategia")
        ax.set_ylabel("Categoría")
        ax.set_xticklabels([s.capitalize() for s in available_strategies], rotation=0)

        fig.tight_layout()
        self._save(fig, "rouge_heatmap.png")
        return fig

    # ------------------------------------------------------------------
    # 6. Longitud de títulos: generado vs. original
    # ------------------------------------------------------------------

    def plot_title_length_comparison(self, df: pd.DataFrame) -> plt.Figure:
        """
        Compara la longitud de los títulos generados vs. los originales.

        Args:
            df: DataFrame con original_title y title_* generados.
        """
        strategies = ["formal", "impactful", "seo"]
        available = [s for s in strategies if f"title_{s}" in df.columns]

        all_cols = {"Original": "original_title"}
        all_cols.update({s.capitalize(): f"title_{s}" for s in available})

        lengths = {}
        for label, col in all_cols.items():
            if col in df.columns:
                lengths[label] = df[col].dropna().apply(lambda x: len(x.split()))

        fig, ax = plt.subplots(figsize=(10, 5))

        for i, (label, data) in enumerate(lengths.items()):
            color = "#ffffff" if label == "Original" else PALETTE[i - 1]
            ax.hist(
                data,
                bins=20,
                alpha=0.6,
                color=color,
                label=f"{label} (μ={data.mean():.1f})",
            )

        ax.set_title("Distribución de Longitud de Títulos (palabras)", fontweight="bold")
        ax.set_xlabel("Número de palabras")
        ax.set_ylabel("Frecuencia")
        ax.legend(fontsize=10)
        ax.grid(alpha=0.2)

        fig.tight_layout()
        self._save(fig, "title_length_comparison.png")
        return fig


if __name__ == "__main__":
    # Demo con datos sintéticos
    import numpy as np
    np.random.seed(42)

    n = 100
    categories = np.random.choice(["business", "sport", "tech", "politics", "entertainment"], n)
    df = pd.DataFrame({
        "category": categories,
        "article_word_count": np.random.randint(100, 800, n),
        "summary_word_count": np.random.randint(30, 150, n),
        "title_len": np.random.randint(5, 80, n),
        "clean_text": [" ".join(np.random.choice(["economy", "market", "sport", "technology", "politics", "government", "company", "player", "team", "innovation"], 20)) for _ in range(n)],
        "original_title": ["Title " + str(i) for i in range(n)],
        "title_formal": ["Formal title " + str(i) for i in range(n)],
        "title_impactful": ["Impactful title " + str(i) for i in range(n)],
        "title_seo": ["SEO title " + str(i) for i in range(n)],
        "rouge1_formal_f1": np.random.uniform(0.1, 0.5, n),
        "rouge2_formal_f1": np.random.uniform(0.05, 0.3, n),
        "rougeL_formal_f1": np.random.uniform(0.1, 0.45, n),
        "rouge1_impactful_f1": np.random.uniform(0.1, 0.4, n),
        "rouge2_impactful_f1": np.random.uniform(0.03, 0.25, n),
        "rougeL_impactful_f1": np.random.uniform(0.08, 0.38, n),
        "rouge1_seo_f1": np.random.uniform(0.15, 0.45, n),
        "rouge2_seo_f1": np.random.uniform(0.07, 0.28, n),
        "rougeL_seo_f1": np.random.uniform(0.12, 0.42, n),
    })

    viz = Visualizer()
    viz.plot_category_distribution(df)
    plt.show()
    viz.plot_length_distributions(df)
    plt.show()
    viz.plot_rouge_comparison(df)
    plt.show()
    print("✓ Visualizaciones generadas")
