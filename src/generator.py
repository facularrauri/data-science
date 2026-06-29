"""
generator.py — Integración con la API de Gemini (Patrón Facade)

Responsabilidades:
  - Proveer una interfaz simple sobre google-genai (Facade)
  - Generar titulares a partir de prompts pre-construidos
  - Manejar rate limiting, errores y reintentos
  - Procesar datasets completos en batch con barra de progreso

SDK: google-genai (sucesor de google-generativeai)
Patrón Facade:
    El cliente solo llama a generate_title() o generate_batch(),
    sin conocer los detalles de configuración de la API de Gemini.
"""

from __future__ import annotations

import time

import pandas as pd
from google import genai
from google.genai import types
from rich.console import Console
from tqdm import tqdm

from src.config import Config
from src.prompt_builder import BuiltPrompt, PromptBuilder

console = Console()


class GeminiGenerator:
    """
    Facade sobre la API de Google Gemini para generación de títulos.

    Usa el SDK google-genai (sucesor de google-generativeai).

    Encapsula:
      - Autenticación y configuración del modelo
      - Construcción y envío de prompts
      - Manejo de errores y rate limiting (backoff exponencial)
      - Generación en batch con progreso

    Ejemplo:
        gen = GeminiGenerator()
        title = gen.generate_title(summary="...", strategy="formal")

        # Batch completo
        df_with_titles = gen.generate_batch(df, strategies=["formal", "impactful"])
    """

    # Rate limiting conservador para Gemini Flash (15 RPM free tier)
    _REQUESTS_PER_MINUTE = 12
    _MIN_DELAY = 60.0 / _REQUESTS_PER_MINUTE  # ~5 segundos entre requests

    def __init__(self) -> None:
        self.cfg = Config.get_instance()
        self._client: genai.Client | None = None
        self._prompt_builder = PromptBuilder()
        self._request_count = 0
        self._last_request_time: float = 0.0
        self._configure_api()

    def _configure_api(self) -> None:
        """Configura el cliente de Gemini con la API key."""
        self._client = genai.Client(api_key=self.cfg.gemini_api_key)
        console.print(
            f"[cyan]API Gemini configurada. Modelo: "
            f"[bold]{self.cfg.gemini_model}[/bold][/cyan]"
        )

    @property
    def client(self) -> genai.Client:
        """Cliente Gemini configurado."""
        if self._client is None:
            self._configure_api()
        return self._client

    # ------------------------------------------------------------------
    # Generación individual
    # ------------------------------------------------------------------

    def generate_title(
        self,
        summary: str,
        strategy: str = "formal",
        category: str = "",
        context_articles: list[str] | None = None,
        max_retries: int = 3,
    ) -> str:
        """
        Genera un único título para un artículo.

        Args:
            summary: Resumen del artículo (primeras N oraciones).
            strategy: Nombre de la estrategia de prompting.
            category: Categoría del artículo (para contexto en el prompt).
            context_articles: Artículos similares de ChromaDB (RAG).
            max_retries: Número máximo de reintentos ante errores de API.

        Returns:
            Título generado como string limpio.
        """
        self._prompt_builder.set_strategy(strategy)
        built = self._prompt_builder.build(
            summary=summary,
            category=category,
            context_articles=context_articles,
        )
        return self._call_api(built.prompt, max_retries=max_retries)

    def generate_from_built_prompt(
        self,
        built_prompt: BuiltPrompt,
        max_retries: int = 3,
    ) -> str:
        """
        Genera un título a partir de un BuiltPrompt ya construido.

        Args:
            built_prompt: Resultado de PromptBuilder.build().
            max_retries: Número máximo de reintentos.

        Returns:
            Título generado como string.
        """
        return self._call_api(built_prompt.prompt, max_retries=max_retries)

    # ------------------------------------------------------------------
    # Generación en batch
    # ------------------------------------------------------------------

    def generate_batch(
        self,
        df: pd.DataFrame,
        strategies: list[str] | None = None,
        summary_column: str = "input_summary",
        n_samples: int | None = None,
        delay_between_requests: float | None = None,
    ) -> pd.DataFrame:
        """
        Genera títulos para múltiples artículos con múltiples estrategias.

        Para cada artículo y cada estrategia, llama a la API de Gemini
        y agrega el resultado como nueva columna al DataFrame.

        Args:
            df: DataFrame con artículos preprocesados.
            strategies: Lista de estrategias a usar. Default: todas.
            summary_column: Columna con el resumen de input.
            n_samples: Si se provee, usa solo los primeros N artículos (para pruebas).
            delay_between_requests: Segundos entre requests (default: calculado por RPM).

        Returns:
            DataFrame con nuevas columnas: title_formal, title_impactful, title_seo.
        """
        strategies = strategies or list(PromptBuilder.AVAILABLE_STRATEGIES.keys())
        delay = delay_between_requests or self._MIN_DELAY

        result_df = df.copy()
        if n_samples:
            result_df = result_df.head(n_samples)

        console.print(
            f"\n[cyan]Generando títulos para {len(result_df):,} artículos "
            f"con {len(strategies)} estrategias...[/cyan]"
        )

        for strategy in strategies:
            col_name = f"title_{strategy}"
            console.print(f"\n[bold]Estrategia: {strategy.upper()}[/bold]")
            titles = []

            for idx, row in tqdm(
                result_df.iterrows(),
                total=len(result_df),
                desc=f"  {strategy}",
            ):
                summary = row.get(summary_column, "")
                category = row.get("category", "")

                if not summary:
                    titles.append("")
                    continue

                title = self.generate_title(
                    summary=summary,
                    strategy=strategy,
                    category=category,
                    max_retries=3,
                )
                titles.append(title)

                # Rate limiting
                self._rate_limit(delay)

            result_df[col_name] = titles

        console.print("\n[green]✓ Generación completada[/green]")
        return result_df

    # ------------------------------------------------------------------
    # Métodos internos
    # ------------------------------------------------------------------

    def _call_api(self, prompt: str, max_retries: int = 3) -> str:
        """
        Llama a la API de Gemini con backoff exponencial ante errores.

        Args:
            prompt: Prompt completo a enviar.
            max_retries: Número máximo de reintentos.

        Returns:
            Texto generado, limpio.
        """
        for attempt in range(max_retries):
            try:
                response = self.client.models.generate_content(
                    model=self.cfg.gemini_model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        max_output_tokens=self.cfg.gemini_max_tokens,
                        temperature=self.cfg.gemini_temperature,
                        top_p=0.9,
                        top_k=40,
                    ),
                )
                text = response.text.strip()
                text = self._clean_output(text)
                self._request_count += 1
                return text

            except Exception as e:
                error_msg = str(e)
                wait_time = 2 ** attempt * 5  # 5s, 10s, 20s

                # Cuota DIARIA agotada (limit: 0) — no tiene sentido reintentar
                if "limit: 0" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
                    console.print(
                        f"[red bold]\n CUOTA AGOTADA para el modelo '{self.cfg.gemini_model}'[/red bold]"
                    )
                    console.print(
                        "[yellow]Opciones para continuar:[/yellow]\n"
                        "  1. Crear una nueva API Key en: https://aistudio.google.com/apikey\n"
                        "  2. Cambiar GEMINI_MODEL en el archivo .env (ej: gemini-1.5-flash)\n"
                        "  3. Esperar al día siguiente (cuota diaria se resetea a medianoche UTC)"
                    )
                    return "[QUOTA_EXHAUSTED]"

                # Rate limit temporal (RPM) — reintenta con delay de la API
                elif "429" in error_msg or "rate" in error_msg.lower():
                    # Extraer el retryDelay sugerido por la API si está disponible
                    import re
                    delay_match = re.search(r'retry.*?(\d+)s', error_msg, re.IGNORECASE)
                    api_delay = int(delay_match.group(1)) + 1 if delay_match else wait_time
                    effective_delay = max(wait_time, api_delay)
                    console.print(
                        f"[yellow]Rate limit (RPM). Esperando {effective_delay}s "
                        f"(intento {attempt+1}/{max_retries})...[/yellow]"
                    )
                    time.sleep(effective_delay)

                elif "blocked" in error_msg.lower() or "safety" in error_msg.lower():
                    console.print("[red]Contenido bloqueado por safety filters.[/red]")
                    return "[BLOCKED]"

                elif "401" in error_msg or "403" in error_msg or "API_KEY" in error_msg:
                    console.print(
                        "[red bold]API Key inválida o sin permisos.[/red bold]\n"
                        "Verificá GEMINI_API_KEY en el archivo .env\n"
                        "Nueva key en: https://aistudio.google.com/apikey"
                    )
                    return "[AUTH_ERROR]"

                else:
                    console.print(f"[red]Error API (intento {attempt+1}): {error_msg[:120]}[/red]")
                    if attempt < max_retries - 1:
                        time.sleep(wait_time)

        return "[ERROR]"

    @staticmethod
    def _clean_output(text: str) -> str:
        """
        Limpia el output del modelo removiendo prefijos comunes.

        El modelo a veces agrega "Headline:", "Title:", "HEADLINE:" etc.
        """
        prefixes_to_remove = [
            "headline:", "title:", "seo headline:", "seo title:",
            "**headline:**", "**title:**", "here is", "here's",
        ]
        text_lower = text.lower()
        for prefix in prefixes_to_remove:
            if text_lower.startswith(prefix):
                text = text[len(prefix):].strip()
                text_lower = text.lower()

        # Remover comillas si envuelven todo el título
        if text.startswith('"') and text.endswith('"'):
            text = text[1:-1].strip()
        if text.startswith("'") and text.endswith("'"):
            text = text[1:-1].strip()

        # Remover asteriscos de markdown
        text = text.replace("**", "").strip()

        return text

    def _rate_limit(self, delay: float) -> None:
        """Aplica rate limiting entre requests."""
        elapsed = time.time() - self._last_request_time
        if elapsed < delay:
            time.sleep(delay - elapsed)
        self._last_request_time = time.time()

    @property
    def request_count(self) -> int:
        """Número total de requests realizados a la API."""
        return self._request_count


class HuggingFaceGenerator:
    """
    Generador local usando modelos de Hugging Face.

    No usa API externa, por lo que evita cuotas y rate limits. Descarga el modelo
    la primera vez que se ejecuta y luego lo reutiliza desde la cache local.
    """

    def __init__(self) -> None:
        self.cfg = Config.get_instance()
        self._tokenizer = None
        self._model = None
        self._device = None
        self._prompt_builder = PromptBuilder()
        self._request_count = 0
        self._configure_model()

    def _configure_model(self) -> None:
        """Carga el modelo local de Hugging Face."""
        try:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
            import torch
        except ImportError as exc:
            raise ImportError(
                "Falta instalar transformers para usar Hugging Face. "
                "Ejecutá: pip install transformers accelerate"
            ) from exc

        requested_device = self.cfg.huggingface_device.lower()
        self._device = torch.device(
            "cuda" if requested_device == "cuda" and torch.cuda.is_available() else "cpu"
        )

        console.print(
            f"[cyan]Modelo Hugging Face local: "
            f"[bold]{self.cfg.huggingface_model}[/bold] ({self._device})[/cyan]"
        )
        self._tokenizer = AutoTokenizer.from_pretrained(self.cfg.huggingface_model)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(self.cfg.huggingface_model)
        self._model.to(self._device)
        self._model.eval()

    @property
    def tokenizer(self):
        """Tokenizer local configurado."""
        if self._tokenizer is None:
            self._configure_model()
        return self._tokenizer

    @property
    def model(self):
        """Modelo local configurado."""
        if self._model is None:
            self._configure_model()
        return self._model

    def generate_title(
        self,
        summary: str,
        strategy: str = "formal",
        category: str = "",
        context_articles: list[str] | None = None,
        max_retries: int = 1,
    ) -> str:
        """Genera un único título para un artículo."""
        self._prompt_builder.set_strategy(strategy)
        built = self._prompt_builder.build(
            summary=summary,
            category=category,
            context_articles=context_articles,
        )
        return self._call_model(built.prompt, max_retries=max_retries)

    def generate_from_built_prompt(
        self,
        built_prompt: BuiltPrompt,
        max_retries: int = 1,
    ) -> str:
        """Genera un título a partir de un BuiltPrompt ya construido."""
        return self._call_model(built_prompt.prompt, max_retries=max_retries)

    def generate_batch(
        self,
        df: pd.DataFrame,
        strategies: list[str] | None = None,
        summary_column: str = "input_summary",
        n_samples: int | None = None,
        delay_between_requests: float | None = None,
    ) -> pd.DataFrame:
        """
        Genera títulos para múltiples artículos con múltiples estrategias.

        Mantiene la misma interfaz que GeminiGenerator para poder cambiar de
        proveedor sin reescribir el notebook.
        """
        strategies = strategies or list(PromptBuilder.AVAILABLE_STRATEGIES.keys())

        result_df = df.copy()
        if n_samples:
            result_df = result_df.head(n_samples)

        console.print(
            f"\n[cyan]Generando títulos locales para {len(result_df):,} artículos "
            f"con {len(strategies)} estrategias...[/cyan]"
        )

        for strategy in strategies:
            col_name = f"title_{strategy}"
            console.print(f"\n[bold]Estrategia: {strategy.upper()}[/bold]")
            titles = []

            for _, row in tqdm(
                result_df.iterrows(),
                total=len(result_df),
                desc=f"  {strategy}",
            ):
                summary = row.get(summary_column, "")
                category = row.get("category", "")

                if not summary:
                    titles.append("")
                    continue

                title = self.generate_title(
                    summary=summary,
                    strategy=strategy,
                    category=category,
                    max_retries=1,
                )
                titles.append(title)

                if delay_between_requests:
                    time.sleep(delay_between_requests)

            result_df[col_name] = titles

        console.print("\n[green]✓ Generación local completada[/green]")
        return result_df

    def _call_model(self, prompt: str, max_retries: int = 1) -> str:
        """Ejecuta el modelo local con manejo simple de errores."""
        import torch

        for attempt in range(max_retries):
            try:
                inputs = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                )
                inputs = {key: value.to(self._device) for key, value in inputs.items()}

                with torch.no_grad():
                    output_ids = self.model.generate(
                        **inputs,
                        max_new_tokens=self.cfg.huggingface_max_tokens,
                        do_sample=self.cfg.huggingface_temperature > 0,
                        temperature=self.cfg.huggingface_temperature,
                    )

                text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
                self._request_count += 1
                return GeminiGenerator._clean_output(text)
            except Exception as exc:
                console.print(
                    f"[red]Error Hugging Face (intento {attempt+1}): "
                    f"{str(exc)[:120]}[/red]"
                )

        return "[ERROR]"

    @property
    def request_count(self) -> int:
        """Número total de generaciones locales realizadas."""
        return self._request_count


def create_generator(provider: str | None = None) -> GeminiGenerator | HuggingFaceGenerator:
    """
    Crea el generador configurado.

    Providers:
      - gemini: API de Google Gemini
      - huggingface: modelo local vía transformers
    """
    cfg = Config.get_instance()
    selected = (provider or cfg.generator_provider).lower()

    if selected == "gemini":
        return GeminiGenerator()
    if selected in {"huggingface", "hf", "local"}:
        return HuggingFaceGenerator()

    raise ValueError(
        f"Proveedor generativo desconocido: {selected!r}. "
        "Usá 'gemini' o 'huggingface'."
    )


if __name__ == "__main__":
    gen = create_generator()

    summary = (
        "The Bank of England has raised interest rates by 0.25 percentage points "
        "to 5.25%, the highest level in 15 years, as it continues to battle inflation. "
        "The decision has raised concerns about the impact on mortgage holders."
    )

    print("=== Test de generación de títulos ===\n")
    for strategy in ["formal", "impactful", "seo"]:
        title = gen.generate_title(summary=summary, strategy=strategy, category="business")
        print(f"[{strategy.upper()}]: {title}")
    
    print(f"\nTotal requests: {gen.request_count}")
