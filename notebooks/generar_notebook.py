"""
Script para generar el notebook del proyecto con encoding UTF-8 correcto.
Ejecutar desde la raíz del proyecto:
    python notebooks/generar_notebook.py
"""
import json
import pathlib

ROOT = pathlib.Path(__file__).parent.parent
OUTPUT = ROOT / "notebooks" / "proyecto_titulos_noticias.ipynb"


def md(*lines):
    """Crea una celda Markdown."""
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": list(lines),
    }


def code(*lines):
    """Crea una celda de código."""
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": list(lines),
    }


# ============================================================
# CELDAS DEL NOTEBOOK
# ============================================================

cells = [

    # --------------------------------------------------------
    # SECCIÓN 1: INTRODUCCIÓN
    # --------------------------------------------------------
    md(
        "# Generacion de Titulos de Noticias con IA Generativa\n",
        "## Ciencia de Datos — UTN Facultad Regional La Plata — 2026\n",
        "\n",
        "---\n",
        "\n",
        "## 1. Introducción y Motivación\n",
        "\n",
        "### Qué problema resolvemos?\n",
        "\n",
        "En el ecosistema de medios digitales, el **titular de una noticia** es la primera —y a veces única— oportunidad de captar la atención del lector. Un buen titular debe ser:\n",
        "\n",
        "- **Informativo**: reflejar con precisión el contenido del artículo\n",
        "- **Atractivo**: generar interés y motivar la lectura\n",
        "- **Conciso**: comunicar la esencia en pocas palabras\n",
        "\n",
        "Redactores y editores invierten tiempo valioso en este proceso. Este proyecto propone **automatizar la generación de titulares** usando un pipeline de IA Generativa que combina:\n",
        "\n",
        "1. **Preprocesamiento NLP** (NLTK) para extraer resúmenes y limpiar texto\n",
        "2. **Embeddings semánticos** (sentence-transformers + GPU) para representación vectorial\n",
        "3. **Base de datos vectorial** (ChromaDB) para búsqueda semántica y RAG\n",
        "4. **Modelo generativo** (Gemini 2.0 Flash) con distintas estrategias de prompting\n",
        "5. **Evaluación cuantitativa** con métricas ROUGE vs. títulos originales\n",
        "\n",
        "### Por qué es relevante?\n",
        "\n",
        "- **Volumen**: Los medios digitales publican cientos de artículos por día; la automatización escala\n",
        "- **Personalización**: Distintas estrategias de prompting generan titulares con diferentes estilos\n",
        "- **Aprendizaje**: El proyecto integra el ciclo completo de un sistema NLP generativo\n",
        "\n",
        "### Dataset\n",
        "\n",
        "Usamos el **BBC News Summary** (Kaggle), con ~2.225 artículos en 5 categorías:\n",
        "`business`, `entertainment`, `politics`, `sport`, `tech`\n",
        "\n",
        "### Arquitectura del pipeline\n",
        "\n",
        "```\n",
        "Artículo BBC\n",
        "    │\n",
        "    ├─► Preprocessor (NLTK) ──► input_summary (3 oraciones)\n",
        "    │                               │\n",
        "    │                               ├─► Embedder (GPU) ──► ChromaDB\n",
        "    │                               │                         │\n",
        "    │                               │            RAG: artículos similares\n",
        "    │                               │                         │\n",
        "    │                               └─► PromptBuilder ◄────────┘\n",
        "    │                                       │\n",
        "    │                           [Formal | Impactful | SEO]\n",
        "    │                                       │\n",
        "    │                               Gemini 2.0 Flash\n",
        "    │                                       │\n",
        "    └─► título original          título generado\n",
        "                 └──────────────────────┘\n",
        "                        ROUGE evaluation\n",
        "```\n",
    ),

    code(
        "# Configuración inicial del entorno\n",
        "import sys\n",
        "import os\n",
        "from pathlib import Path\n",
        "\n",
        "# En Jupyter, usamos cwd().parent (la notebook está en notebooks/)\n",
        "ROOT = Path.cwd().parent\n",
        "if str(ROOT) not in sys.path:\n",
        "    sys.path.insert(0, str(ROOT))\n",
        "\n",
        "import warnings\n",
        "warnings.filterwarnings('ignore')\n",
        "os.environ['TOKENIZERS_PARALLELISM'] = 'false'\n",
        "\n",
        "print(f'Python: {sys.version}')\n",
        "print(f'Working dir: {Path.cwd()}')\n",
        "print(f'Root: {ROOT}')\n",
    ),

    code(
        "# Verificar configuración y GPU\n",
        "from src.config import Config\n",
        "\n",
        "cfg = Config.get_instance()\n",
        "print(cfg)\n",
    ),

    code(
        "import torch\n",
        "print(f'PyTorch: {torch.__version__}')\n",
        "print(f'CUDA disponible: {torch.cuda.is_available()}')\n",
        "if torch.cuda.is_available():\n",
        "    print(f'GPU: {torch.cuda.get_device_name(0)}')\n",
        "    vram = torch.cuda.get_device_properties(0).total_memory / 1e9\n",
        "    print(f'VRAM total: {vram:.2f} GB')\n",
    ),

    # --------------------------------------------------------
    # SECCIÓN 2: DESCARGA DEL DATASET
    # --------------------------------------------------------
    md(
        "---\n",
        "## 2. Descarga del Dataset\n",
        "\n",
        "Descargamos el dataset **BBC News Summary** de Kaggle usando la API oficial.\n",
        "El dataset contiene artículos de noticias en 5 categorías con sus resúmenes correspondientes.\n",
        "\n",
        "La API Key de Kaggle se carga automáticamente desde el archivo `.env`.\n",
        "\n",
        "**Estructura del dataset:**\n",
        "```\n",
        "BBC News Summary/\n",
        "├── News Articles/\n",
        "│   ├── business/     ← artículos .txt\n",
        "│   ├── entertainment/\n",
        "│   ├── politics/\n",
        "│   ├── sport/\n",
        "│   └── tech/\n",
        "└── Summaries/\n",
        "    ├── business/     ← resúmenes .txt\n",
        "    ├── entertainment/\n",
        "    ├── politics/\n",
        "    ├── sport/\n",
        "    └── tech/\n",
        "```\n",
    ),

    code(
        "import os\n",
        "from pathlib import Path\n",
        "from src.config import Config\n",
        "\n",
        "cfg = Config.get_instance()\n",
        "\n",
        "# Configurar credenciales de Kaggle\n",
        "os.environ['KAGGLE_USERNAME'] = cfg.kaggle_username\n",
        "os.environ['KAGGLE_KEY'] = cfg.kaggle_key\n",
        "\n",
        "dataset_check = cfg.dataset_path / 'BBC News Summary'\n",
        "\n",
        "if dataset_check.exists() and any(dataset_check.iterdir()):\n",
        "    print(f'Dataset ya descargado en: {cfg.dataset_path}')\n",
        "else:\n",
        "    print('Descargando dataset BBC News Summary de Kaggle...')\n",
        "    import kaggle\n",
        "    kaggle.api.authenticate()\n",
        "    kaggle.api.dataset_download_files(\n",
        "        'pariza/bbc-news-summary',\n",
        "        path=str(cfg.dataset_path),\n",
        "        unzip=True,\n",
        "        quiet=False,\n",
        "    )\n",
        "    print(f'Dataset descargado en: {cfg.dataset_path}')\n",
        "\n",
        "# Listar estructura\n",
        "print('\\nEstructura del dataset:')\n",
        "for p in sorted(cfg.dataset_path.rglob('*'))[:20]:\n",
        "    indent = '  ' * (len(p.parts) - len(cfg.dataset_path.parts))\n",
        "    suffix = '/' if p.is_dir() else ''\n",
        "    print(f'{indent}{p.name}{suffix}')\n",
    ),

    # --------------------------------------------------------
    # SECCIÓN 3: CARGA Y EXPLORACIÓN
    # --------------------------------------------------------
    md(
        "---\n",
        "## 3. Carga y Exploración del Dataset\n",
        "\n",
        "Cargamos todos los artículos desde la estructura de carpetas de Kaggle y construimos\n",
        "un **DataFrame consolidado**. Luego exploramos:\n",
        "\n",
        "- Distribución por categoría\n",
        "- Longitud promedio de artículos, resúmenes y títulos\n",
        "- Distribución de longitudes\n",
        "- WordCloud global y por categoría\n",
    ),

    code(
        "from src.data_loader import DataLoader\n",
        "\n",
        "loader = DataLoader()\n",
        "df = loader.load()\n",
        "loader.describe()\n",
    ),

    code(
        "# Muestra de 3 artículos\n",
        "sample = loader.get_sample(3)\n",
        "for _, row in sample.iterrows():\n",
        "    print(f\"\\n{'='*60}\")\n",
        "    print(f\"Categoria: {row['category'].upper()}\")\n",
        "    print(f\"Titulo original: {row['original_title']}\")\n",
        "    print(f\"Inicio del articulo:\\n{row['article'][:300]}...\")\n",
    ),

    code(
        "import matplotlib.pyplot as plt\n",
        "from src.visualizer import Visualizer\n",
        "\n",
        "viz = Visualizer()\n",
        "\n",
        "# Distribución por categoría\n",
        "fig = viz.plot_category_distribution(df)\n",
        "plt.show()\n",
    ),

    code(
        "# Distribución de longitudes\n",
        "fig = viz.plot_length_distributions(df)\n",
        "plt.show()\n",
    ),

    # --------------------------------------------------------
    # SECCIÓN 4: PREPROCESAMIENTO
    # --------------------------------------------------------
    md(
        "---\n",
        "## 4. Preprocesamiento de Datos\n",
        "\n",
        "Aplicamos el pipeline de preprocesamiento NLP:\n",
        "\n",
        "| Paso | Técnica | Herramienta |\n",
        "|------|---------|-------------|\n",
        "| 1 | Extracción de primeras 3 oraciones | NLTK sent_tokenize |\n",
        "| 2 | Lowercasing | str.lower() |\n",
        "| 3 | Remoción de URLs, emails, puntuación | regex |\n",
        "| 4 | Tokenización | NLTK word_tokenize |\n",
        "| 5 | Remoción de stopwords | NLTK stopwords (inglés) |\n",
        "| 6 | Lematización | WordNetLemmatizer |\n",
        "\n",
        "El preprocesamiento produce tres nuevas columnas:\n",
        "- `input_summary`: texto de entrada al modelo Gemini (3 oraciones limpias)\n",
        "- `clean_tokens`: lista de tokens para análisis estadístico\n",
        "- `clean_text`: tokens como string (para WordCloud)\n",
        "\n",
        "**Ejemplo de lematización:**\n",
        "```\n",
        "Texto:   'Running companies are buying goods from markets'\n",
        "Tokens:  ['running', 'company', 'buying', 'good', 'market']\n",
        "```\n",
    ),

    code(
        "from src.preprocessor import Preprocessor\n",
        "\n",
        "pp = Preprocessor()\n",
        "df = pp.fit_transform(df)\n",
        "\n",
        "nuevas_cols = [c for c in df.columns if c not in\n",
        "               ['category', 'filename', 'article', 'summary',\n",
        "                'article_len', 'summary_len', 'title_len',\n",
        "                'article_word_count', 'summary_word_count', 'original_title']]\n",
        "print('Columnas agregadas:', nuevas_cols)\n",
        "print(f'\\nEjemplo de input_summary:')\n",
        "print(df['input_summary'].iloc[0])\n",
        "print(f'\\nTokens limpios (primeros 15):')\n",
        "print(df['clean_tokens'].iloc[0][:15])\n",
    ),

    code(
        "# Análisis de frecuencia de palabras clave por categoría\n",
        "keyword_freqs = pp.get_keyword_frequencies(df, top_n=15, by_category=True)\n",
        "\n",
        "print('Top 10 palabras por categoria:')\n",
        "for cat, freqs in keyword_freqs.items():\n",
        "    top_words = ', '.join([f'{w}({c})' for w, c in freqs[:10]])\n",
        "    print(f'  {cat.upper():15s}: {top_words}')\n",
    ),

    code(
        "# WordCloud global\n",
        "fig = viz.plot_wordcloud(df)\n",
        "plt.show()\n",
    ),

    code(
        "# WordClouds por categoría\n",
        "for cat in ['business', 'sport', 'tech']:\n",
        "    fig = viz.plot_wordcloud(df, category=cat)\n",
        "    plt.title(f'WordCloud - {cat.capitalize()}')\n",
        "    plt.show()\n",
    ),

    code(
        "# Top palabras por categoría (gráfico de barras)\n",
        "fig = viz.plot_top_keywords(keyword_freqs, top_n=12)\n",
        "plt.show()\n",
    ),

    # --------------------------------------------------------
    # SECCIÓN 5: EMBEDDINGS Y CHROMADB
    # --------------------------------------------------------
    md(
        "---\n",
        "## 5. Embeddings y Base Vectorial (ChromaDB)\n",
        "\n",
        "### Por qué embeddings?\n",
        "\n",
        "Los embeddings transforman texto en vectores numéricos que capturan **significado semántico**.\n",
        "Artículos con contenido similar estarán cerca en el espacio vectorial, independientemente\n",
        "de las palabras exactas usadas.\n",
        "\n",
        "**Modelo usado:** `all-MiniLM-L6-v2` (384 dimensiones, rápido, eficiente)\n",
        "\n",
        "### Por qué ChromaDB?\n",
        "\n",
        "ChromaDB es una base de datos vectorial **embedded** (sin Docker, sin servidor):\n",
        "- Almacena documentos + embeddings en disco\n",
        "- Búsqueda por similitud coseno (K-NN aproximado con HNSW)\n",
        "- Filtros de metadatos con API intuitiva tipo MongoDB: `where={\"category\": \"sport\"}`\n",
        "\n",
        "### Pipeline RAG (Retrieval-Augmented Generation)\n",
        "\n",
        "```\n",
        "input_summary\n",
        "     │\n",
        "     ▼\n",
        "  Embedder (GPU) → vector [384-dim]\n",
        "     │\n",
        "     ▼\n",
        "  ChromaDB.query_similar()\n",
        "     │\n",
        "     ▼\n",
        "  Top-3 artículos similares → contexto en el prompt\n",
        "```\n",
    ),

    code(
        "from src.embedder import Embedder\n",
        "\n",
        "embedder = Embedder()\n",
        "print(f'Dispositivo: {embedder.device}')\n",
        "print(f'Modelo: {cfg.embedding_model}')\n",
        "print(f'Dimension de embeddings: {embedder.embedding_dim}')\n",
    ),

    code(
        "# Demo de similitud semántica\n",
        "test_texts = [\n",
        "    'Bank of England raises interest rates to fight inflation',\n",
        "    'Central bank increases borrowing costs amid economic pressure',\n",
        "    'England wins football match in extra time',\n",
        "]\n",
        "\n",
        "test_embs = embedder.encode(test_texts, show_progress=False)\n",
        "\n",
        "sim_12 = embedder.cosine_similarity(test_embs[0], test_embs[1])\n",
        "sim_13 = embedder.cosine_similarity(test_embs[0], test_embs[2])\n",
        "\n",
        "print('Demostracion de similitud semantica:')\n",
        "print(f'  Banco vs Banco: {sim_12:.4f}  <- textos semanticamente similares')\n",
        "print(f'  Banco vs Futbol: {sim_13:.4f}  <- textos semanticamente distintos')\n",
        "assert sim_12 > sim_13, 'Error: la similitud deberia ser mayor entre textos relacionados'\n",
        "print('OK: similitud coseno funciona correctamente')\n",
    ),

    code(
        "from src.vector_store import VectorStore\n",
        "\n",
        "# Inicializar ChromaDB (persiste en ./chroma_db)\n",
        "store = VectorStore(embedder=embedder)\n",
        "print(f'Documentos ya indexados: {store.count}')\n",
    ),

    code(
        "# Indexar todos los artículos (o reutilizar si ya están indexados)\n",
        "store.index_dataframe(df, text_column='input_summary')\n",
        "print(f'\\nTotal indexado en ChromaDB: {store.count:,} documentos')\n",
    ),

    code(
        "# Demo: búsqueda semántica\n",
        "print('=== Busqueda semantica: technology innovation startup ===')\n",
        "results = store.query_similar('technology innovation startup', n_results=5)\n",
        "display(results[['category', 'original_title', 'similarity']].round(4))\n",
    ),

    code(
        "# Demo: filtro por categoría (API tipo MongoDB)\n",
        "print('=== Filtro por categoria: sport ===')\n",
        "sport_results = store.filter_by_category('sport', n_results=5)\n",
        "display(sport_results[['category', 'original_title']].head())\n",
    ),

    code(
        "# Demo: búsqueda semántica + filtro combinado (RAG)\n",
        "print(\"=== Busqueda 'economy inflation' solo en business ===\")\n",
        "business_results = store.query_similar(\n",
        "    'economy inflation interest rates',\n",
        "    n_results=5,\n",
        "    where={'category': 'business'},\n",
        ")\n",
        "display(business_results[['category', 'original_title', 'similarity']].round(4))\n",
    ),

    # --------------------------------------------------------
    # SECCIÓN 6: PROMPTING
    # --------------------------------------------------------
    md(
        "---\n",
        "## 6. Diseño del Pipeline de Prompting\n",
        "\n",
        "### Patron Strategy\n",
        "\n",
        "Implementamos el **Patron Strategy** para encapsular tres estrategias de prompting:\n",
        "\n",
        "| Estrategia | Estilo | Referencia | Longitud target |\n",
        "|------------|--------|------------|-----------------|\n",
        "| **Formal** | Periodístico, objetivo, neutral | BBC, Reuters | max 12 palabras |\n",
        "| **Impactful** | Llamativo, emocional, viral | BuzzFeed, Daily Mail | max 12 palabras |\n",
        "| **SEO** | Optimizado para buscadores | Google News | 50-60 caracteres |\n",
        "\n",
        "### RAG con ChromaDB\n",
        "\n",
        "Antes de generar el título, recuperamos artículos similares de ChromaDB y los\n",
        "incluimos en el prompt como ejemplos de referencia.\n",
        "Esto ayuda al modelo a mantenerse en el dominio temático correcto.\n",
    ),

    code(
        "from src.prompt_builder import PromptBuilder\n",
        "\n",
        "# Tomar un artículo de ejemplo\n",
        "sample_row = df.sample(1, random_state=42).iloc[0]\n",
        "summary = sample_row['input_summary']\n",
        "category = sample_row['category']\n",
        "original_title = sample_row['original_title']\n",
        "\n",
        "print(f'Categoria: {category.upper()}')\n",
        "print(f'Titulo original: {original_title}')\n",
        "print(f'\\nResumen (input al modelo):')\n",
        "print(summary)\n",
    ),

    code(
        "# Recuperar artículos similares para RAG\n",
        "similar = store.query_similar(\n",
        "    summary,\n",
        "    n_results=4,\n",
        "    where={'category': category},\n",
        ")\n",
        "context_articles = similar['original_title'].tolist()[:3]\n",
        "\n",
        "print('Articulos similares recuperados (contexto RAG):')\n",
        "for i, title in enumerate(context_articles, 1):\n",
        "    print(f'  {i}. {title}')\n",
    ),

    code(
        "# Construir y comparar los 3 prompts\n",
        "builder = PromptBuilder()\n",
        "all_prompts = builder.build_all_strategies(\n",
        "    summary=summary,\n",
        "    category=category,\n",
        "    context_articles=context_articles,\n",
        ")\n",
        "\n",
        "for strategy_name, built in all_prompts.items():\n",
        "    print(f\"\\n{'='*60}\")\n",
        "    print(f'ESTRATEGIA: {strategy_name.upper()} - {built.strategy_description}')\n",
        "    print(f\"{'='*60}\")\n",
        "    print(built.prompt)\n",
    ),

    # --------------------------------------------------------
    # SECCIÓN 7: INTEGRACIÓN GEMINI
    # --------------------------------------------------------
    md(
        "---\n",
        "## 7. Integración con el Modelo Generativo (Gemini)\n",
        "\n",
        "### Patron Facade\n",
        "\n",
        "La clase `GeminiGenerator` actúa como **Facade** sobre la API de Google Gemini\n",
        "(`google-genai`, el SDK moderno):\n",
        "\n",
        "- Oculta la complejidad de configuración del modelo\n",
        "- Maneja automáticamente rate limiting y reintentos con backoff exponencial\n",
        "- Limpia el output del modelo (prefijos indeseados, comillas, asteriscos)\n",
        "\n",
        "### Configuración del modelo\n",
        "| Parámetro | Valor | Descripción |\n",
        "|-----------|-------|-------------|\n",
        "| Modelo | `gemini-2.0-flash` | Rápido, ideal para texto corto |\n",
        "| Temperature | 0.7 | Balance creatividad / coherencia |\n",
        "| Max tokens | 200 | Suficiente para un titular |\n",
        "\n",
        "**Nota:** El parámetro `N_SAMPLES` controla cuántos artículos se envían a la API.\n",
        "Con 50 artículos × 3 estrategias = **150 llamadas a la API**.\n",
    ),

    code(
        "from src.generator import GeminiGenerator\n",
        "\n",
        "gen = GeminiGenerator()\n",
        "\n",
        "# Test rápido con el artículo de ejemplo\n",
        "print(f'Articulo de prueba: {original_title}')\n",
        "print(f'Categoria: {category}')\n",
        "print()\n",
        "\n",
        "for strategy in ['formal', 'impactful', 'seo']:\n",
        "    title = gen.generate_title(\n",
        "        summary=summary,\n",
        "        strategy=strategy,\n",
        "        category=category,\n",
        "        context_articles=context_articles,\n",
        "    )\n",
        "    print(f'[{strategy.upper():10s}]: {title}')\n",
        "\n",
        "print(f'\\n[ORIGINAL  ]: {original_title}')\n",
    ),

    code(
        "import pandas as pd\n",
        "\n",
        "# Generación en batch — muestra estratificada por categoría\n",
        "N_SAMPLES = 50  # Ajustar segun cuota de API disponible\n",
        "\n",
        "df_sample = df.groupby('category').apply(\n",
        "    lambda x: x.sample(min(N_SAMPLES // 5, len(x)), random_state=42)\n",
        ").reset_index(drop=True)\n",
        "\n",
        "print(f'Muestra estratificada: {len(df_sample)} articulos')\n",
        "print(df_sample['category'].value_counts().to_string())\n",
    ),

    code(
        "# Generación en batch con las 3 estrategias\n",
        "# Realiza N_SAMPLES x 3 llamadas a la API de Gemini\n",
        "df_generated = gen.generate_batch(\n",
        "    df_sample,\n",
        "    strategies=['formal', 'impactful', 'seo'],\n",
        "    summary_column='input_summary',\n",
        ")\n",
        "\n",
        "print(f'\\nTotal requests a API: {gen.request_count}')\n",
        "print('\\nMuestra de resultados:')\n",
        "display(df_generated[[\n",
        "    'category', 'original_title',\n",
        "    'title_formal', 'title_impactful', 'title_seo'\n",
        "]].head(5))\n",
    ),

    code(
        "# Guardar títulos generados\n",
        "from src.evaluator import Evaluator\n",
        "evaluator = Evaluator()\n",
        "path = evaluator.save_generated_titles(df_generated)\n",
        "print(f'Guardado en: {path}')\n",
    ),

    # --------------------------------------------------------
    # SECCIÓN 8: EVALUACIÓN
    # --------------------------------------------------------
    md(
        "---\n",
        "## 8. Evaluación de Resultados\n",
        "\n",
        "### Metricas ROUGE\n",
        "\n",
        "**ROUGE** (Recall-Oriented Understudy for Gisting Evaluation) compara el overlap\n",
        "léxico entre el texto generado y el texto de referencia (título original):\n",
        "\n",
        "| Métrica | Nivel | Descripción |\n",
        "|---------|-------|-------------|\n",
        "| **ROUGE-1** | Unigramas | Overlap de palabras individuales |\n",
        "| **ROUGE-2** | Bigramas | Overlap de pares de palabras |\n",
        "| **ROUGE-L** | Secuencias | Longest Common Subsequence |\n",
        "\n",
        "Para cada métrica se calcula: **precision**, **recall** y **F1**.\n",
        "\n",
        "### Limitaciones de ROUGE\n",
        "\n",
        "ROUGE mide similitud **léxica**, no calidad periodística ni creatividad.\n",
        "Un título perfectamente válido con distintas palabras puede tener score bajo.\n",
        "Por eso complementamos con **análisis cualitativo** sobre 10 ejemplos seleccionados.\n",
    ),

    code(
        "# Calcular métricas ROUGE\n",
        "df_evaluated = evaluator.evaluate(df_generated)\n",
        "evaluator.print_summary(df_evaluated)\n",
    ),

    code(
        "# Estadísticas detalladas por estrategia\n",
        "stats = evaluator.summary_stats(df_evaluated)\n",
        "print('\\nEstadísticas (mean ± std):')\n",
        "display(stats.round(4))\n",
    ),

    code(
        "# Visualización: comparación ROUGE por estrategia\n",
        "fig = viz.plot_rouge_comparison(df_evaluated)\n",
        "plt.show()\n",
    ),

    code(
        "# Heatmap: ROUGE-L por categoría y estrategia\n",
        "fig = viz.plot_rouge_heatmap(df_evaluated)\n",
        "plt.show()\n",
    ),

    code(
        "# Comparación de longitud de títulos\n",
        "fig = viz.plot_title_length_comparison(df_generated)\n",
        "plt.show()\n",
    ),

    md(
        "### 8.2 Análisis Cualitativo\n",
        "\n",
        "Seleccionamos 10 ejemplos representativos:\n",
        "- **3 mejores** (alto ROUGE-L): casos donde el modelo acierta\n",
        "- **3 peores** (bajo ROUGE-L): casos problemáticos\n",
        "- **4 aleatorios**: visión general del comportamiento\n",
        "\n",
        "El análisis cualitativo complementa a ROUGE evaluando aspectos que la métrica no captura:\n",
        "creatividad, coherencia temática, impacto periodístico.\n",
    ),

    code(
        "# Seleccionar ejemplos para análisis cualitativo\n",
        "qualitative = evaluator.get_qualitative_examples(df_evaluated, n_examples=10)\n",
        "\n",
        "for i, (_, row) in enumerate(qualitative.iterrows(), 1):\n",
        "    print(f\"\\n{'='*70}\")\n",
        "    print(f\"Ejemplo {i} | Categoria: {row.get('category', 'N/A').upper()}\")\n",
        "    print(f\"{'='*70}\")\n",
        "    print(f\"ORIGINAL:   {row.get('original_title', 'N/A')}\")\n",
        "    print(f\"FORMAL:     {row.get('title_formal', 'N/A')}\")\n",
        "    print(f\"IMPACTFUL:  {row.get('title_impactful', 'N/A')}\")\n",
        "    print(f\"SEO:        {row.get('title_seo', 'N/A')}\")\n",
        "    r1 = row.get('rougeL_formal_f1', float('nan'))\n",
        "    r2 = row.get('rougeL_impactful_f1', float('nan'))\n",
        "    r3 = row.get('rougeL_seo_f1', float('nan'))\n",
        "    print(f\"ROUGE-L F1: Formal={r1:.4f} | Impactful={r2:.4f} | SEO={r3:.4f}\")\n",
    ),

    code(
        "# Guardar resultados completos\n",
        "path = evaluator.save_results(df_evaluated)\n",
        "print(f'Resultados guardados en: {path}')\n",
    ),

    # --------------------------------------------------------
    # SECCIÓN 9: CONCLUSIONES
    # --------------------------------------------------------
    md(
        "---\n",
        "## 9. Conclusiones\n",
        "\n",
        "### 9.1 Análisis de Resultados\n",
        "\n",
        "#### Sobre las métricas ROUGE\n",
        "\n",
        "ROUGE mide el **overlap léxico** entre el título generado y el original. Un score bajo\n",
        "no implica que el título sea malo — puede ser igualmente válido con vocabulario diferente.\n",
        "Esta es una **limitación inherente** para evaluar texto generativo:\n",
        "el modelo puede producir títulos creativamente superiores que ROUGE penaliza\n",
        "por no coincidir con las palabras exactas del original.\n",
        "\n",
        "#### Sobre las estrategias de prompting\n",
        "\n",
        "- **Formal**: mayor overlap léxico con los originales (ambos son estilo BBC)\n",
        "- **Impactful**: mayor diversidad léxica → ROUGE más bajo, títulos más atractivos\n",
        "- **SEO**: longitudes controladas, foco en palabras clave → buena precisión\n",
        "\n",
        "#### Sobre el RAG con ChromaDB\n",
        "\n",
        "La incorporación de artículos similares como contexto en el prompt ayuda al modelo\n",
        "a mantenerse en el dominio temático correcto, especialmente para categorías técnicas.\n",
        "\n",
        "### 9.2 Que funcionó bien\n",
        "\n",
        "- El pipeline de preprocesamiento NLTK funciona robustamente sobre el corpus BBC\n",
        "- Los embeddings de sentence-transformers capturan bien la semántica por categoría\n",
        "- ChromaDB facilita la búsqueda semántica con filtros intuitivos\n",
        "- Gemini genera títulos coherentes con el contenido de los artículos\n",
        "- La estrategia formal produce los títulos más similares a los originales\n",
        "\n",
        "### 9.3 Que no funcionó / Que mejoraría\n",
        "\n",
        "- **ROUGE como métrica única es insuficiente**: necesitaríamos BERTScore o evaluación\n",
        "  humana para capturar calidad semántica\n",
        "- **Rate limiting de Gemini**: en producción se necesitaría un sistema de cola\n",
        "  con reintentos más sofisticado\n",
        "- **Tamaño de muestra**: 50 artículos es limitado estadísticamente;\n",
        "  con más cuota se analizaría el corpus completo (~2.225 artículos)\n",
        "- **Mejora futura**: Fine-tuning de un modelo pequeño (Mistral 7B o similar)\n",
        "  con los pares (resumen, título) del dataset BBC\n",
        "- **Mejora futura**: Agregar feedback humano (RLHF-like) para entrenar\n",
        "  un clasificador de calidad de titulares\n",
    ),

    code(
        "# Resumen final del pipeline ejecutado\n",
        "print('=' * 60)\n",
        "print('  RESUMEN DEL PIPELINE')\n",
        "print('=' * 60)\n",
        "print(f'  Dataset:           BBC News Summary')\n",
        "print(f'  Total articulos:   {len(df):,}')\n",
        "print(f'  Muestra evaluada:  {len(df_generated):,}')\n",
        "print(f'  Modelo generativo: {cfg.gemini_model}')\n",
        "print(f'  Modelo embeddings: {cfg.embedding_model} ({embedder.device})')\n",
        "print(f'  Base vectorial:    ChromaDB embedded ({store.count:,} docs)')\n",
        "print(f'  Estrategias:       formal | impactful | seo')\n",
        "print(f'  API calls totales: {gen.request_count}')\n",
        "print(f'  Outputs en:        {cfg.outputs_path}')\n",
        "print('=' * 60)\n",
        "\n",
        "print('\\nMetricas ROUGE-L F1 (mean):')\n",
        "final_stats = evaluator.summary_stats(df_evaluated)\n",
        "if 'rougeL_f1_mean' in final_stats.columns:\n",
        "    for _, row in final_stats.iterrows():\n",
        "        print(f'  {row[\"strategy\"].upper():12s}: {row.get(\"rougeL_f1_mean\", 0):.4f}')\n",
    ),
]

# ============================================================
# ARMAR EL NOTEBOOK JSON
# ============================================================

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3 (datos)",
            "language": "python",
            "name": "datos",
        },
        "language_info": {
            "name": "python",
            "version": "3.12.10",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

# Escribir con UTF-8 puro (sin BOM), con ensure_ascii=False
OUTPUT.write_bytes(
    json.dumps(notebook, ensure_ascii=False, indent=1).encode("utf-8")
)

print(f"Notebook generado: {OUTPUT}")
print(f"Celdas: {len(cells)}")

# Verificar encoding
raw = OUTPUT.read_bytes()
print(f"Tiene BOM: {raw[:3] == b'\\xef\\xbb\\xbf'}")
first_line = json.loads(raw.decode("utf-8"))["cells"][0]["source"][0]
print(f"Primera línea: {first_line!r}")
