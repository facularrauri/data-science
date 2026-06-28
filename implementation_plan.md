# Generación de Títulos de Noticias con IA Generativa

## Descripción General

Proyecto académico de Ciencia de Datos (UTN - FRLP) que implementa un pipeline completo para generar titulares periodísticos usando el dataset BBC News Summary, embeddings locales con GPU (sentence-transformers), ChromaDB como base vectorial y la API de Gemini como modelo generativo. Se evalúa con métricas ROUGE vs. los títulos originales.

---

## Arquitectura del Proyecto

```
datos/
├── .venv/                          # Virtualenv existente
├── .env                            # API Keys (gitignored)
├── data/
│   └── bbc-news-summary/           # Dataset de Kaggle
│       ├── BBC News Summary/
│       │   ├── News Articles/      # Artículos por categoría
│       │   └── Summaries/          # Resúmenes por categoría
│       └── bbc-news-summary.csv    # CSV consolidado (si existe)
├── src/                            # Módulos Python reutilizables
│   ├── __init__.py
│   ├── config.py                   # Configuración centralizada (Singleton)
│   ├── data_loader.py              # Carga y exploración del dataset
│   ├── preprocessor.py             # Limpieza, tokenización, lematización
│   ├── embedder.py                 # Embeddings locales (sentence-transformers GPU)
│   ├── vector_store.py             # ChromaDB: indexado y consultas
│   ├── prompt_builder.py           # Diseño y variantes de prompts
│   ├── generator.py                # Integración con Gemini API
│   ├── evaluator.py                # Métricas ROUGE + análisis cualitativo
│   └── visualizer.py              # WordCloud, distribuciones, plots
├── notebooks/
│   └── proyecto_titulos_noticias.ipynb   # Notebook principal
├── outputs/
│   ├── generated_titles.csv        # Títulos generados
│   ├── evaluation_results.csv      # Métricas ROUGE
│   └── figures/                    # Gráficos exportados
├── chroma_db/                      # Base vectorial persistente (ChromaDB)
└── requirements.txt
```

---

## Patrones de Diseño Aplicados

| Patrón | Módulo | Uso |
|--------|--------|-----|
| **Singleton** | `config.py` | Una sola instancia de configuración global |
| **Strategy** | `prompt_builder.py` | Intercambiar estrategias de prompting (formal, impactante, SEO) |
| **Facade** | `generator.py` | Interfaz simple sobre la API de Gemini |
| **Pipeline / Chain** | Notebook | Orquestación secuencial de etapas |
| **Repository** | `vector_store.py` | Abstracción de ChromaDB como repositorio |

---

## Módulos y Responsabilidades

### `src/config.py` — Singleton de Configuración
- Lee variables de entorno desde `.env`
- Expone rutas, modelos, parámetros de generación

### `src/data_loader.py` — Carga de Datos
- Descarga/lectura del dataset BBC
- Construcción de DataFrame con columnas: `category`, `article`, `summary`, `original_title`
- Estadísticas básicas (shape, nulls, distribuciones)

### `src/preprocessor.py` — Preprocesamiento
- Lowercasing, remoción de puntuación y stopwords
- Tokenización y lematización (NLTK + WordNetLemmatizer)
- Extracción de primeras N oraciones como resumen de input
- Frecuencia de palabras clave por categoría

### `src/embedder.py` — Embeddings Locales con GPU
- Modelo: `all-MiniLM-L6-v2` o `paraphrase-multilingual-mpnet-base-v2`
- Usa `sentence-transformers` con `device='cuda'`
- Batch encoding para eficiencia

### `src/vector_store.py` — ChromaDB (Repository Pattern)
- Colección persistente en `./chroma_db/`
- Métodos: `add_documents()`, `query_similar()`, `filter_by_category()`
- API tipo `where={"category": "sport"}` (similar a MongoDB)

### `src/prompt_builder.py` — Strategy Pattern para Prompts
- **PromptStrategy** (ABC)
  - `FormalPromptStrategy`: titular periodístico serio
  - `ImpactfulPromptStrategy`: titular llamativo/viral
  - `SEOPromptStrategy`: optimizado para búsqueda web
- `PromptBuilder`: selecciona estrategia y construye prompt final

### `src/generator.py` — Facade sobre Gemini
- Cliente de `google.generativeai`
- Método `generate_title(summary, strategy)` → str
- Manejo de rate limiting y errores

### `src/evaluator.py` — Evaluación
- Métricas: ROUGE-1, ROUGE-2, ROUGE-L (`rouge-score`)
- Análisis cualitativo de 10 ejemplos seleccionados
- Exportación de resultados a CSV

### `src/visualizer.py` — Visualizaciones
- WordCloud por categoría
- Distribución de longitudes de artículos/resúmenes/títulos
- Top-N palabras por categoría (barras)
- Heatmap de scores ROUGE por estrategia de prompt

---

## Estructura de la Notebook

La notebook importa desde `src/` y actúa como orquestador. Cada celda de Markdown explica la sección.

1. **Introducción y Motivación** — contexto del problema
2. **Carga y Exploración** — `DataLoader` + visualizaciones
3. **Preprocesamiento** — `Preprocessor` pipeline
4. **Diseño del Pipeline de Prompting** — 3 estrategias comparadas
5. **Embeddings y Base Vectorial** — indexado + búsqueda semántica
6. **Integración con Gemini** — generación de títulos en batch
7. **Evaluación** — ROUGE + análisis cualitativo
8. **Conclusiones**

---

## Stack Tecnológico

| Librería | Versión | Uso |
|----------|---------|-----|
| `pandas` | latest | DataFrame principal |
| `nltk` | latest | Tokenización, lemmatización, stopwords |
| `sentence-transformers` | latest | Embeddings locales con GPU |
| `chromadb` | latest | Base vectorial embedded |
| `google-generativeai` | latest | API Gemini |
| `rouge-score` | latest | Métricas de evaluación |
| `wordcloud` | latest | Nube de palabras |
| `matplotlib` / `seaborn` | latest | Visualizaciones |
| `python-dotenv` | latest | Carga de .env |
| `kaggle` | latest | Descarga del dataset |
| `tqdm` | latest | Barras de progreso |
| `torch` | CUDA | Backend GPU para sentence-transformers |

---

## Decisiones de Diseño

> [!IMPORTANT]
> La API Key de Gemini y el Kaggle Token se guardan en `.env` (nunca en el notebook).

> [!NOTE]
> ChromaDB en modo **embedded** (sin Docker): persiste en `./chroma_db/`. Ideal para notebooks académicos. La API usa filtros tipo `where={"category": "sport"}` similar a MongoDB.

> [!TIP]
> Con GPU disponible, `sentence-transformers` usará CUDA automáticamente para encoding en batch → embeddings de todos los artículos en segundos.

---

## Plan de Ejecución

1. Crear `.env` con las API Keys
2. Instalar dependencias (`requirements.txt`)
3. Crear estructura de directorios y módulos `src/`
4. Descargar dataset de Kaggle
5. Crear la notebook con todas las secciones
6. Verificar ejecución end-to-end

---

## Verificación

- Cada módulo `src/*.py` tendrá un bloque `if __name__ == "__main__"` para prueba independiente
- La notebook corre de inicio a fin sin errores
- Los outputs se guardan en `outputs/`
