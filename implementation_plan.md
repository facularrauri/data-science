# Generación de Títulos de Noticias con IA Generativa

## Descripción General

Proyecto académico de Ciencia de Datos (UTN - FRLP) que implementa un pipeline completo para generar titulares periodísticos usando el dataset **BBC News Summary**, preprocesamiento NLP, embeddings semánticos, ChromaDB como base vectorial, RAG liviano y un backend generativo configurable.

El modelo generativo puede seleccionarse desde `.env` con `GENERATOR_PROVIDER`:

- `gemini`: usa Google Gemini vía API.
- `huggingface`: usa un modelo local de Hugging Face con `transformers`.

Los resultados se evalúan con métricas ROUGE frente a los títulos originales del dataset y se complementan con análisis cualitativo.

---

## Arquitectura del Proyecto

```
data-science/
├── .venv/                          # Virtualenv local (gitignored)
├── .env                            # Credenciales y configuración (gitignored)
├── .gitignore
├── README.md
├── implementation_plan.md
├── INFORME_TECNICO_BASE.md         # Base para redactar el informe técnico
├── requirements.txt
├── data/
│   └── bbc-news-summary/           # Dataset de Kaggle (gitignored)
├── src/                            # Módulos Python reutilizables
│   ├── __init__.py
│   ├── config.py                   # Configuración centralizada (Singleton)
│   ├── data_loader.py              # Carga y exploración del dataset
│   ├── preprocessor.py             # Limpieza, tokenización, lematización
│   ├── embedder.py                 # Embeddings semánticos con GPU/CPU
│   ├── vector_store.py             # ChromaDB: indexado y consultas
│   ├── prompt_builder.py           # Diseño y variantes de prompts
│   ├── generator.py                # Gemini / Hugging Face (Facade)
│   ├── evaluator.py                # Métricas ROUGE + análisis cualitativo
│   └── visualizer.py               # WordCloud, distribuciones, plots
├── notebooks/
│   └── proyecto_titulos_noticias.ipynb
├── outputs/
│   ├── generated_titles.csv        # Títulos generados
│   ├── evaluation_results.csv      # Métricas ROUGE
│   └── figures/                    # Gráficos exportados
└── chroma_db/                      # Base vectorial persistente (gitignored)
```

---

## Patrones de Diseño Aplicados

| Patrón | Módulo | Uso |
|--------|--------|-----|
| **Singleton** | `config.py` | Una sola instancia de configuración global |
| **Strategy** | `prompt_builder.py` | Intercambiar estrategias de prompting: formal, impactante y SEO |
| **Facade** | `generator.py` | Interfaz común para Gemini y Hugging Face |
| **Repository** | `vector_store.py` | Abstracción de ChromaDB como repositorio vectorial |
| **Pipeline / Chain** | Notebook | Orquestación secuencial de carga, EDA, NLP, RAG, generación y evaluación |

---

## Módulos y Responsabilidades

### `src/config.py` — Singleton de Configuración

- Lee variables de entorno desde `.env`.
- Expone rutas del proyecto, modelos, parámetros de generación y parámetros de embeddings.
- Permite seleccionar el proveedor generativo con `GENERATOR_PROVIDER`.
- Solo exige `GEMINI_API_KEY` cuando `GENERATOR_PROVIDER=gemini`.

### `src/data_loader.py` — Carga de Datos

- Lee el dataset BBC News Summary desde la estructura descargada por Kaggle.
- Construye un DataFrame consolidado con columnas como `category`, `filename`, `article`, `summary` y `original_title`.
- Guarda/cachea el dataset en `outputs/` cuando corresponde.
- Provee descripciones básicas del corpus.

### `src/preprocessor.py` — Preprocesamiento

- Extrae `input_summary` a partir de las primeras oraciones del artículo.
- Aplica lowercasing, limpieza con regex, tokenización, remoción de stopwords y lematización.
- Genera `clean_tokens` y `clean_text`.
- Calcula frecuencias de palabras por categoría.

### `src/embedder.py` — Embeddings Semánticos

- Usa `sentence-transformers`, por defecto `all-MiniLM-L6-v2`.
- Intenta usar CUDA si está disponible y cae a CPU si no lo está.
- Genera embeddings en batch y normalizados para similitud coseno.

### `src/vector_store.py` — ChromaDB (Repository Pattern)

- Usa ChromaDB embedded y persistente en `./chroma_db/`.
- Indexa el DataFrame con `index_dataframe()`.
- Consulta artículos similares con `query_similar()`.
- Filtra por categoría con `filter_by_category()`.
- Soporta filtros tipo `where={"category": "sport"}`.

### `src/prompt_builder.py` — Strategy Pattern para Prompts

- Define una interfaz `PromptStrategy`.
- Implementa tres estrategias:
  - `FormalPromptStrategy`: titular periodístico serio, neutral e informativo.
  - `ImpactfulPromptStrategy`: titular llamativo, con gancho emocional.
  - `SEOPromptStrategy`: titular optimizado para buscadores.
- `PromptBuilder` construye prompts con resumen, categoría y contexto RAG opcional.

### `src/generator.py` — Facade Generativo

- Expone `create_generator()` para elegir backend según `GENERATOR_PROVIDER`.
- `GeminiGenerator` usa `google-genai`, API key, rate limiting y reintentos.
- `HuggingFaceGenerator` usa `transformers`, `AutoTokenizer` y `AutoModelForSeq2SeqLM`.
- Ambos backends exponen:
  - `generate_title()`
  - `generate_batch()`
  - `request_count`

### `src/evaluator.py` — Evaluación

- Calcula ROUGE-1, ROUGE-2 y ROUGE-L con `rouge-score`.
- Agrega columnas de precision, recall y F1 por estrategia.
- Imprime resumen estadístico por estrategia.
- Selecciona ejemplos para análisis cualitativo evitando problemas con columnas no hashables como listas.
- Exporta resultados a CSV.

### `src/visualizer.py` — Visualizaciones

- Distribución de categorías.
- Distribución de longitudes de artículos, resúmenes y títulos.
- WordCloud global y por categoría.
- Top-N palabras por categoría.
- Comparación de métricas ROUGE y heatmaps de resultados.

---

## Estructura de la Notebook

La notebook `notebooks/proyecto_titulos_noticias.ipynb` importa desde `src/` y actúa como orquestador del proyecto.

1. **Introducción y Motivación** — problema, dataset y arquitectura del pipeline configurable.
2. **Descarga del Dataset** — configuración de Kaggle desde `.env`.
3. **Carga y Exploración** — `DataLoader` y primeras estadísticas.
4. **Preprocesamiento** — generación de `input_summary`, `clean_tokens` y `clean_text`.
5. **Embeddings y Base Vectorial** — embeddings con `sentence-transformers` e indexado en ChromaDB.
6. **Diseño del Pipeline de Prompting** — comparación de prompts formal, impactante y SEO.
7. **Integración con Modelo Generativo** — `create_generator()` para Gemini o Hugging Face.
8. **Evaluación** — métricas ROUGE, estadísticas y análisis cualitativo.
9. **Conclusiones** — lectura crítica de resultados, limitaciones y mejoras futuras.

---

## Uso de RAG con ChromaDB

El proyecto implementa un RAG liviano:

1. Se indexan los `input_summary` en ChromaDB.
2. Para un artículo de ejemplo, se consulta `query_similar()` filtrando por categoría.
3. Se toman títulos similares como `context_articles`.
4. Esos ejemplos se inyectan en el prompt mediante `PromptBuilder`.

En la notebook, el RAG se usa explícitamente en el ejemplo individual de generación. El método `generate_batch()` automatiza la generación masiva por estrategia, pero no consulta ChromaDB por cada fila en la versión actual.

---

## Stack Tecnológico

| Librería | Uso |
|----------|-----|
| `pandas`, `numpy`, `pyarrow` | Manipulación, cache y análisis de datos |
| `nltk` | Tokenización, stopwords y lematización |
| `sentence-transformers` | Embeddings semánticos |
| `torch`, `torchvision` | Backend CPU/GPU para modelos |
| `chromadb` | Base vectorial embedded |
| `google-genai` | API moderna de Google Gemini |
| `transformers`, `accelerate` | Modelos Hugging Face locales |
| `rouge-score` | Métricas ROUGE |
| `matplotlib`, `seaborn`, `wordcloud` | Visualizaciones |
| `python-dotenv` | Carga de variables `.env` |
| `kaggle` | Descarga del dataset |
| `tqdm`, `rich` | Progreso y salida formateada |
| `jupyter`, `ipykernel`, `ipywidgets` | Ejecución notebook |

---

## Variables de Entorno

### Gemini

```env
GENERATOR_PROVIDER=gemini
GEMINI_API_KEY=tu_api_key
GEMINI_MODEL=gemini-1.5-flash
GEMINI_MAX_TOKENS=200
GEMINI_TEMPERATURE=0.7
```

### Hugging Face local

```env
GENERATOR_PROVIDER=huggingface
HUGGINGFACE_MODEL=google/flan-t5-base
HUGGINGFACE_DEVICE=cpu
HUGGINGFACE_MAX_TOKENS=64
HUGGINGFACE_TEMPERATURE=0.7
```

Para `google/flan-t5-base` no hace falta token de Hugging Face. Solo sería necesario para modelos privados, gated o para usar una API online.

### Kaggle

```env
KAGGLE_USERNAME=tu_usuario
KAGGLE_KEY=tu_kaggle_key
```

---

## Decisiones de Diseño

> [!IMPORTANT]
> `.env` queda fuera de git. Allí se guardan credenciales y selección del proveedor generativo.

> [!NOTE]
> ChromaDB corre en modo embedded y persiste en `./chroma_db/`, sin Docker ni servidor.

> [!TIP]
> Hugging Face local permite ejecutar la generación sin cuota externa. Gemini puede ser más rápido o potente, pero depende de API key y límites de uso.

> [!WARNING]
> ROUGE mide overlap léxico, no calidad editorial completa. Por eso se complementa con análisis cualitativo.

---

## Plan de Ejecución

1. Crear y activar `.venv`.
2. Instalar dependencias con `pip install -r requirements.txt`.
3. Crear `.env` con Kaggle y el proveedor generativo elegido.
4. Ejecutar la notebook desde el inicio.
5. Descargar/cargar el dataset BBC News Summary.
6. Ejecutar EDA y preprocesamiento.
7. Indexar embeddings en ChromaDB.
8. Revisar prompts y ejemplo RAG.
9. Ejecutar generación con `create_generator()`.
10. Generar batch de títulos para la muestra estratificada.
11. Calcular ROUGE y análisis cualitativo.
12. Guardar CSVs y figuras en `outputs/`.
13. Usar `INFORME_TECNICO_BASE.md` como base para el informe final.

---

## Verificación

- `src/config.py`, `src/generator.py` y `src/evaluator.py` compilan con `py_compile`.
- No hay errores de linter en los módulos modificados.
- La notebook contempla Gemini y Hugging Face en sus textos principales.
- `README.md` refleja el backend configurable y el informe base.
- Los resultados generados se guardan en `outputs/generated_titles.csv` y `outputs/evaluation_results.csv`.

---

## Entregables del Proyecto

- Notebook principal: `notebooks/proyecto_titulos_noticias.ipynb`.
- Código modular en `src/`.
- Resultados en `outputs/`.
- Documentación general en `README.md`.
- Plan técnico en `implementation_plan.md`.
- Base para informe formal en `INFORME_TECNICO_BASE.md`.
