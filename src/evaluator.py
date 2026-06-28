"""
evaluator.py — Evaluación cuantitativa y cualitativa de títulos generados

Responsabilidades:
  - Calcular métricas ROUGE-1, ROUGE-2 y ROUGE-L
  - Comparar títulos generados vs. originales
  - Generar análisis estadístico por estrategia de prompt
  - Seleccionar ejemplos para análisis cualitativo
  - Exportar resultados a CSV
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd
from rich.console import Console
from rich.table import Table
from rouge_score import rouge_scorer

from src.config import Config

console = Console()


class Evaluator:
    """
    Evaluador de títulos generados vs. títulos originales.

    Usa métricas ROUGE (Recall-Oriented Understudy for Gisting Evaluation)
    para comparación cuantitativa:
      - ROUGE-1: Overlap de unigramas
      - ROUGE-2: Overlap de bigramas
      - ROUGE-L: Longest Common Subsequence

    Ejemplo:
        evaluator = Evaluator()
        results_df = evaluator.evaluate(df_with_titles)
        evaluator.print_summary(results_df)
        evaluator.save_results(results_df)
    """

    STRATEGIES = ["formal", "impactful", "seo"]
    ROUGE_TYPES = ["rouge1", "rouge2", "rougeL"]

    def __init__(self) -> None:
        self.cfg = Config.get_instance()
        self._scorer = rouge_scorer.RougeScorer(
            self.ROUGE_TYPES,
            use_stemmer=True,
        )

    # ------------------------------------------------------------------
    # Evaluación principal
    # ------------------------------------------------------------------

    def evaluate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calcula métricas ROUGE para todos los títulos generados.

        Para cada estrategia y cada métrica ROUGE, agrega columnas:
            rouge1_formal_precision, rouge1_formal_recall, rouge1_formal_f1,
            rouge2_formal_precision, ...
            rougeL_formal_f1, etc.

        Args:
            df: DataFrame con columnas original_title, title_formal,
                title_impactful, title_seo.

        Returns:
            DataFrame enriquecido con todas las métricas ROUGE.
        """
        available_strategies = [
            s for s in self.STRATEGIES if f"title_{s}" in df.columns
        ]

        if not available_strategies:
            raise ValueError(
                "No se encontraron columnas de títulos generados. "
                "Asegurate de haber ejecutado generate_batch() primero."
            )

        result_df = df.copy()

        for strategy in available_strategies:
            col_generated = f"title_{strategy}"
            console.print(f"[cyan]Calculando ROUGE para estrategia: {strategy}...[/cyan]")

            scores_per_row = result_df.apply(
                lambda row: self._compute_rouge(
                    generated=str(row.get(col_generated, "")),
                    reference=str(row.get("original_title", "")),
                ),
                axis=1,
            )

            # Expandir dict de scores en columnas
            for rouge_type in self.ROUGE_TYPES:
                result_df[f"{rouge_type}_{strategy}_precision"] = scores_per_row.apply(
                    lambda s: s.get(rouge_type, {}).get("precision", 0.0)
                )
                result_df[f"{rouge_type}_{strategy}_recall"] = scores_per_row.apply(
                    lambda s: s.get(rouge_type, {}).get("recall", 0.0)
                )
                result_df[f"{rouge_type}_{strategy}_f1"] = scores_per_row.apply(
                    lambda s: s.get(rouge_type, {}).get("f1", 0.0)
                )

        console.print("[green]✓ Métricas ROUGE calculadas[/green]")
        return result_df

    def _compute_rouge(self, generated: str, reference: str) -> dict:
        """
        Calcula ROUGE entre un título generado y el original.

        Args:
            generated: Título generado por el modelo.
            reference: Título original del dataset.

        Returns:
            Dict con scores por tipo de ROUGE.
        """
        if not generated or generated in ("[ERROR]", "[BLOCKED]") or not reference:
            return {rt: {"precision": 0.0, "recall": 0.0, "f1": 0.0} for rt in self.ROUGE_TYPES}

        scores = self._scorer.score(reference, generated)
        return {
            rt: {
                "precision": scores[rt].precision,
                "recall": scores[rt].recall,
                "f1": scores[rt].fmeasure,
            }
            for rt in self.ROUGE_TYPES
        }

    # ------------------------------------------------------------------
    # Análisis estadístico
    # ------------------------------------------------------------------

    def summary_stats(self, results_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calcula estadísticas resumidas de ROUGE por estrategia.

        Args:
            results_df: DataFrame resultado de evaluate().

        Returns:
            DataFrame con media, std y mediana de cada métrica por estrategia.
        """
        available_strategies = [
            s for s in self.STRATEGIES if f"title_{s}" in results_df.columns
        ]

        rows = []
        for strategy in available_strategies:
            row = {"strategy": strategy}
            for rouge_type in self.ROUGE_TYPES:
                col_f1 = f"{rouge_type}_{strategy}_f1"
                if col_f1 in results_df.columns:
                    row[f"{rouge_type}_f1_mean"] = results_df[col_f1].mean()
                    row[f"{rouge_type}_f1_std"] = results_df[col_f1].std()
                    row[f"{rouge_type}_f1_median"] = results_df[col_f1].median()
            rows.append(row)

        return pd.DataFrame(rows).round(4)

    def print_summary(self, results_df: pd.DataFrame) -> None:
        """Imprime la tabla de resumen ROUGE en consola."""
        stats = self.summary_stats(results_df)

        console.print("\n[bold blue]═══ Resultados de Evaluación ROUGE ═══[/bold blue]\n")

        table = Table(
            title="ROUGE F1 Scores por Estrategia de Prompt",
            show_header=True,
            header_style="bold magenta",
        )
        table.add_column("Estrategia", style="cyan", min_width=12)
        table.add_column("ROUGE-1 F1", style="green")
        table.add_column("ROUGE-2 F1", style="yellow")
        table.add_column("ROUGE-L F1", style="blue")

        for _, row in stats.iterrows():
            table.add_row(
                str(row["strategy"]).upper(),
                f"{row.get('rouge1_f1_mean', 0):.4f} ± {row.get('rouge1_f1_std', 0):.4f}",
                f"{row.get('rouge2_f1_mean', 0):.4f} ± {row.get('rouge2_f1_std', 0):.4f}",
                f"{row.get('rougeL_f1_mean', 0):.4f} ± {row.get('rougeL_f1_std', 0):.4f}",
            )
        console.print(table)

        # Mejor estrategia
        if "rouge1_f1_mean" in stats.columns:
            best_idx = stats["rouge1_f1_mean"].idxmax()
            best = stats.loc[best_idx, "strategy"]
            console.print(
                f"\n[green]Mejor estrategia (ROUGE-1 F1): [bold]{best.upper()}[/bold][/green]"
            )

    def get_qualitative_examples(
        self,
        results_df: pd.DataFrame,
        n_examples: int = 10,
        seed: int = 42,
    ) -> pd.DataFrame:
        """
        Selecciona ejemplos para análisis cualitativo.

        Selecciona una muestra representativa que incluya:
          - 3 ejemplos de alto ROUGE-L (buenos)
          - 3 ejemplos de bajo ROUGE-L (malos)
          - 4 ejemplos aleatorios

        Args:
            results_df: DataFrame con métricas ROUGE.
            n_examples: Total de ejemplos a retornar.
            seed: Semilla aleatoria.

        Returns:
            DataFrame con los ejemplos seleccionados y sus métricas.
        """
        available_strategies = [
            s for s in self.STRATEGIES if f"title_{s}" in results_df.columns
        ]
        if not available_strategies:
            return results_df.sample(n=min(n_examples, len(results_df)), random_state=seed)

        # Usar la primera estrategia disponible como referencia para selección
        ref_strategy = available_strategies[0]
        col = f"rougeL_{ref_strategy}_f1"

        if col not in results_df.columns:
            return results_df.sample(n=min(n_examples, len(results_df)), random_state=seed)

        df_sorted = results_df.sort_values(col, ascending=False).reset_index(drop=True)

        n_top = n_examples // 3
        n_bottom = n_examples // 3
        n_random = n_examples - n_top - n_bottom

        top = df_sorted.head(n_top)
        bottom = df_sorted.tail(n_bottom)
        middle = df_sorted.iloc[n_top:-n_bottom]
        random_sample = middle.sample(n=min(n_random, len(middle)), random_state=seed)

        qualitative = pd.concat([top, random_sample, bottom]).drop_duplicates()

        # Columnas relevantes para el análisis
        cols = (
            ["category", "original_title"]
            + [f"title_{s}" for s in available_strategies if f"title_{s}" in results_df.columns]
            + [f"rougeL_{s}_f1" for s in available_strategies if f"rougeL_{s}_f1" in results_df.columns]
            + ["input_summary"]
        )
        cols = [c for c in cols if c in qualitative.columns]
        return qualitative[cols].reset_index(drop=True)

    # ------------------------------------------------------------------
    # Exportación
    # ------------------------------------------------------------------

    def save_results(
        self,
        results_df: pd.DataFrame,
        filename: str = "evaluation_results.csv",
    ) -> Path:
        """
        Guarda los resultados de evaluación en CSV.

        Args:
            results_df: DataFrame con métricas.
            filename: Nombre del archivo.

        Returns:
            Path del archivo guardado.
        """
        output_path = self.cfg.outputs_path / filename
        results_df.to_csv(output_path, index=False)
        console.print(f"[green]✓ Resultados guardados en {output_path}[/green]")
        return output_path

    def save_generated_titles(
        self,
        df: pd.DataFrame,
        filename: str = "generated_titles.csv",
    ) -> Path:
        """
        Guarda los títulos generados en CSV (solo columnas relevantes).

        Returns:
            Path del archivo guardado.
        """
        available_strategies = [
            s for s in self.STRATEGIES if f"title_{s}" in df.columns
        ]
        cols = (
            ["category", "filename", "original_title", "input_summary"]
            + [f"title_{s}" for s in available_strategies]
        )
        cols = [c for c in cols if c in df.columns]

        output_path = self.cfg.outputs_path / filename
        df[cols].to_csv(output_path, index=False)
        console.print(f"[green]✓ Títulos generados guardados en {output_path}[/green]")
        return output_path


if __name__ == "__main__":
    # Test con datos sintéticos
    import numpy as np
    np.random.seed(42)

    sample_data = {
        "category": ["business", "sport", "tech"],
        "filename": ["001", "002", "003"],
        "original_title": [
            "Bank of England raises interest rates to 15-year high",
            "England win World Cup after penalty shootout",
            "Apple unveils new iPhone with revolutionary AI features",
        ],
        "input_summary": ["summary 1", "summary 2", "summary 3"],
        "title_formal": [
            "Bank of England increases interest rates to highest level since 2008",
            "England defeats opponent in World Cup final",
            "Apple announces new smartphone with artificial intelligence capabilities",
        ],
        "title_impactful": [
            "Breaking: Interest Rates Hit 15-Year High as Bank Acts on Inflation",
            "Shock! England Win World Cup in Penalty Drama",
            "Apple's New iPhone Will Change Everything You Know About AI",
        ],
        "title_seo": [
            "Bank of England Interest Rate Hike 2024: What It Means for Mortgages",
            "England World Cup Win: Penalty Shootout Victory Highlights",
            "Apple iPhone AI Features: Complete Guide to New Capabilities",
        ],
    }
    df = pd.DataFrame(sample_data)

    evaluator = Evaluator()
    results = evaluator.evaluate(df)
    evaluator.print_summary(results)
    examples = evaluator.get_qualitative_examples(results, n_examples=3)
    print("\nEjemplos cualitativos:")
    print(examples[["original_title", "title_formal"]].to_string())
