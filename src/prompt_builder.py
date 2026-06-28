"""
prompt_builder.py — Diseño de prompts con Patrón Strategy

Responsabilidades:
  - Definir variantes de prompts para generación de títulos
  - Implementar el patrón Strategy para intercambiar estilos de prompting
  - Proveer un PromptBuilder que construye el prompt final con contexto

Estrategias implementadas:
  1. FormalPromptStrategy     → Titular periodístico serio y objetivo
  2. ImpactfulPromptStrategy  → Titular llamativo, viral, con gancho
  3. SEOPromptStrategy        → Titular optimizado para buscadores web

Patrón Strategy:
    strategy = ImpactfulPromptStrategy()
    builder = PromptBuilder(strategy=strategy)
    prompt = builder.build(summary="Texto del artículo...")
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


# ===========================================================================
# Interfaz base (Strategy)
# ===========================================================================


class PromptStrategy(ABC):
    """
    Interfaz abstracta para estrategias de construcción de prompts.

    Cada estrategia define un estilo diferente de prompt para guiar
    al modelo generativo hacia un tipo distinto de título.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Nombre identificador de la estrategia."""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """Descripción del estilo de título que genera."""
        ...

    @abstractmethod
    def build(self, summary: str, category: str = "", context: str = "") -> str:
        """
        Construye el prompt completo.

        Args:
            summary: Resumen del artículo (primeras N oraciones).
            category: Categoría del artículo (opcional, para contexto).
            context: Artículos similares recuperados de ChromaDB (opcional).

        Returns:
            Prompt completo listo para enviar al modelo.
        """
        ...


# ===========================================================================
# Estrategia 1: Formal / Periodístico
# ===========================================================================


class FormalPromptStrategy(PromptStrategy):
    """
    Genera titulares periodísticos formales y objetivos.

    Estilo: BBC, Reuters, The Guardian.
    Características: Informativo, neutral, sin sensacionalismo.
    """

    @property
    def name(self) -> str:
        return "formal"

    @property
    def description(self) -> str:
        return "Titular periodístico formal, objetivo e informativo"

    def build(self, summary: str, category: str = "", context: str = "") -> str:
        cat_hint = f" de la sección {category.upper()}" if category else ""

        context_block = ""
        if context:
            context_block = f"""
EJEMPLOS DE REFERENCIA (artículos similares del archivo):
{context}

"""

        return f"""You are a professional journalist writing for a prestigious news outlet like the BBC or Reuters.

Your task is to write a single, clear, and informative headline{cat_hint} based on the following article summary.

RULES:
- Write exactly ONE headline (no alternatives, no explanations)
- Maximum 12 words
- Use active voice when possible
- Be factual and neutral — no sensationalism
- Do not use clickbait or emotional manipulation
- Do not add quotes, colons introducing lists, or bullet points
- Write in English
{context_block}
ARTICLE SUMMARY:
{summary}

HEADLINE:"""


# ===========================================================================
# Estrategia 2: Impactante / Viral
# ===========================================================================


class ImpactfulPromptStrategy(PromptStrategy):
    """
    Genera titulares llamativos, con gancho emocional y alto impacto.

    Estilo: BuzzFeed, Daily Mail, Upworthy.
    Características: Urgencia, curiosidad, gancho emocional.
    """

    @property
    def name(self) -> str:
        return "impactful"

    @property
    def description(self) -> str:
        return "Titular llamativo, con gancho emocional y alto impacto"

    def build(self, summary: str, category: str = "", context: str = "") -> str:
        context_block = ""
        if context:
            context_block = f"""
REFERENCE EXAMPLES (similar articles for inspiration):
{context}

"""

        return f"""You are a viral content editor who creates attention-grabbing headlines that make people want to click and read.

Your task is to write a single, impactful and compelling headline based on the following article summary.

RULES:
- Write exactly ONE headline (no alternatives, no explanations)
- Maximum 12 words
- Use power words that create urgency or curiosity (e.g., "Reveals", "Shocking", "Breaking", "First Ever")
- Include a hook that makes readers want to know more
- Can use numbers if relevant (e.g., "5 reasons why...")
- Do not add quotes, colons introducing lists, or bullet points
- Write in English
{context_block}
ARTICLE SUMMARY:
{summary}

HEADLINE:"""


# ===========================================================================
# Estrategia 3: SEO / Buscadores
# ===========================================================================


class SEOPromptStrategy(PromptStrategy):
    """
    Genera titulares optimizados para motores de búsqueda (SEO).

    Estilo: Search Engine Journal, HubSpot Blog.
    Características: Palabras clave al inicio, claridad, longitud óptima (50-60 chars).
    """

    @property
    def name(self) -> str:
        return "seo"

    @property
    def description(self) -> str:
        return "Titular optimizado para SEO con palabras clave relevantes"

    def build(self, summary: str, category: str = "", context: str = "") -> str:
        context_block = ""
        if context:
            context_block = f"""
REFERENCE EXAMPLES (similar articles):
{context}

"""

        return f"""You are an SEO specialist and content strategist. Your task is to write a search-engine-optimized headline for a news article.

RULES:
- Write exactly ONE headline (no alternatives, no explanations)
- Between 50 and 60 characters total (optimal for Google search results)
- Put the most important keyword at the BEGINNING of the headline
- Be descriptive and specific — avoid vague terms
- Include the main topic/subject clearly
- Do not add quotes, colons introducing lists, or bullet points
- Write in English
{context_block}
ARTICLE SUMMARY:
{summary}

SEO HEADLINE:"""


# ===========================================================================
# PromptBuilder — Orquestador
# ===========================================================================


@dataclass
class BuiltPrompt:
    """Resultado de la construcción de un prompt."""
    strategy_name: str
    strategy_description: str
    summary: str
    category: str
    prompt: str

    def __repr__(self) -> str:
        preview = self.prompt[:200].replace("\n", " ")
        return (
            f"BuiltPrompt(\n"
            f"  strategy={self.strategy_name!r},\n"
            f"  category={self.category!r},\n"
            f"  prompt_preview={preview!r}...\n"
            f")"
        )


class PromptBuilder:
    """
    Construye prompts usando la estrategia configurada (Patrón Strategy).

    Permite cambiar la estrategia en runtime sin modificar el código cliente.

    Ejemplo:
        builder = PromptBuilder(strategy=FormalPromptStrategy())
        prompt = builder.build(summary="...", category="sport")

        # Cambiar estrategia en runtime
        builder.set_strategy(ImpactfulPromptStrategy())
        prompt2 = builder.build(summary="...", category="sport")
    """

    # Registro de estrategias disponibles
    AVAILABLE_STRATEGIES: dict[str, type[PromptStrategy]] = {
        "formal": FormalPromptStrategy,
        "impactful": ImpactfulPromptStrategy,
        "seo": SEOPromptStrategy,
    }

    def __init__(self, strategy: PromptStrategy | str = "formal") -> None:
        if isinstance(strategy, str):
            self._strategy = self._resolve_strategy(strategy)
        else:
            self._strategy = strategy

    def set_strategy(self, strategy: PromptStrategy | str) -> "PromptBuilder":
        """Cambia la estrategia activa (retorna self para encadenamiento)."""
        if isinstance(strategy, str):
            self._strategy = self._resolve_strategy(strategy)
        else:
            self._strategy = strategy
        return self

    def build(
        self,
        summary: str,
        category: str = "",
        context_articles: list[str] | None = None,
    ) -> BuiltPrompt:
        """
        Construye el prompt completo con la estrategia activa.

        Args:
            summary: Resumen del artículo.
            category: Categoría del artículo.
            context_articles: Artículos similares de ChromaDB para contexto (RAG).

        Returns:
            BuiltPrompt con el prompt y metadatos.
        """
        context = ""
        if context_articles:
            context = "\n".join(
                f"- {art}" for art in context_articles[:3]
            )

        prompt = self._strategy.build(
            summary=summary,
            category=category,
            context=context,
        )

        return BuiltPrompt(
            strategy_name=self._strategy.name,
            strategy_description=self._strategy.description,
            summary=summary,
            category=category,
            prompt=prompt,
        )

    def build_all_strategies(
        self,
        summary: str,
        category: str = "",
        context_articles: list[str] | None = None,
    ) -> dict[str, BuiltPrompt]:
        """
        Construye prompts con TODAS las estrategias disponibles.

        Útil para comparar resultados de diferentes estilos.

        Returns:
            Dict {strategy_name: BuiltPrompt}.
        """
        results = {}
        for name, strategy_cls in self.AVAILABLE_STRATEGIES.items():
            builder = PromptBuilder(strategy=strategy_cls())
            results[name] = builder.build(summary, category, context_articles)
        return results

    def _resolve_strategy(self, name: str) -> PromptStrategy:
        if name not in self.AVAILABLE_STRATEGIES:
            available = list(self.AVAILABLE_STRATEGIES.keys())
            raise ValueError(
                f"Estrategia desconocida: {name!r}. "
                f"Disponibles: {available}"
            )
        return self.AVAILABLE_STRATEGIES[name]()

    @property
    def strategy(self) -> PromptStrategy:
        return self._strategy

    @property
    def strategy_name(self) -> str:
        return self._strategy.name


if __name__ == "__main__":
    summary = (
        "The Bank of England has raised interest rates by 0.25 percentage points "
        "to 5.25%, the highest level in 15 years, as it continues to battle "
        "inflation which remains well above its 2% target. The decision was "
        "widely expected by markets but has raised concerns about the impact "
        "on mortgage holders and the broader economy."
    )

    builder = PromptBuilder()

    print("=== Comparando las 3 estrategias ===\n")
    all_prompts = builder.build_all_strategies(summary, category="business")

    for name, built in all_prompts.items():
        print(f"[{name.upper()}] — {built.strategy_description}")
        print(f"Prompt preview: {built.prompt[:300]}...")
        print()
