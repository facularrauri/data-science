# 🗞️ Generación de Títulos de Noticias con IA Generativa

**UTN Facultad Regional La Plata — Ciencia de Datos 2026**

Pipeline completo para generar titulares periodísticos usando el dataset BBC News Summary,
embeddings semánticos, ChromaDB, RAG y un backend generativo configurable:
Google Gemini vía API o Hugging Face local.

---

## 🚀 Inicio Rápido

```powershell
# 1. Activar el entorno virtual
.\.venv\Scripts\Activate.ps1

# 2. Instalar/actualizar dependencias
pip install -r requirements.txt

# 3. Configurar .env
# Elegir GENERATOR_PROVIDER=gemini o GENERATOR_PROVIDER=huggingface

# 4. Abrir Jupyter Lab
jupyter lab notebooks/proyecto_titulos_noticias.ipynb
```

## 📁 Estructura del Proyecto

```
datos/
├── .env                    # API Keys (no commitear)
├── requirements.txt        # Dependencias
├── INFORME_TECNICO_BASE.md # Base para redactar el informe técnico
├── src/                    # Módulos Python
│   ├── config.py           # Singleton de configuración
│   ├── data_loader.py      # Carga del dataset BBC
│   ├── preprocessor.py     # NLP: tokenización, lematización
│   ├── embedder.py         # Embeddings semánticos (GPU)
│   ├── vector_store.py     # ChromaDB (Repository Pattern)
│   ├── prompt_builder.py   # Estrategias de prompting (Strategy Pattern)
│   ├── generator.py        # Gemini / Hugging Face (Facade Pattern)
│   ├── evaluator.py        # Métricas ROUGE
│   └── visualizer.py       # Visualizaciones
├── notebooks/
│   └── proyecto_titulos_noticias.ipynb  # Notebook principal
├── data/                   # Dataset BBC (ignorado por git)
├── outputs/                # Resultados generados
│   ├── generated_titles.csv
│   ├── evaluation_results.csv
│   └── figures/            # Gráficos exportados
└── chroma_db/              # Base vectorial (ignorada por git)
```

## 🏗️ Arquitectura y Patrones de Diseño

| Patrón | Módulo | Descripción |
|--------|--------|-------------|
| **Singleton** | `config.py` | Configuración única en toda la app |
| **Strategy** | `prompt_builder.py` | 3 estilos de prompt intercambiables |
| **Facade** | `generator.py` | Interfaz común para Gemini o Hugging Face |
| **Repository** | `vector_store.py` | Abstracción de ChromaDB |
| **Pipeline** | Notebook | Orquestación secuencial |

## ⚙️ Stack Tecnológico

| Componente | Tecnología |
|------------|------------|
| Dataset | BBC News Summary (Kaggle) |
| NLP | NLTK (tokenización, stopwords, lematización) |
| Embeddings | sentence-transformers + GPU/CPU |
| Base vectorial | ChromaDB embedded |
| Modelo generativo | Google Gemini API o Hugging Face local |
| Evaluación | ROUGE-1, ROUGE-2, ROUGE-L |
| Visualización | Matplotlib, Seaborn, WordCloud |

## 📊 Estrategias de Prompting

1. **Formal**: Titular periodístico serio al estilo BBC/Reuters
2. **Impactful**: Titular llamativo, viral, con gancho emocional
3. **SEO**: Optimizado para motores de búsqueda (50-60 chars)

## 🔧 Instalación de PyTorch con CUDA

Para usar GPU (recomendado):

```powershell
# RTX 3090 con CUDA 12.x
.\.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

## 📝 Variables de Entorno (.env)

### Opción A: Gemini

```env
GENERATOR_PROVIDER=gemini
GEMINI_API_KEY=tu_api_key_aqui
GEMINI_MODEL=gemini-1.5-flash
GEMINI_MAX_TOKENS=200
GEMINI_TEMPERATURE=0.7

KAGGLE_USERNAME=tu_usuario
KAGGLE_KEY=tu_kaggle_key
```

Gemini requiere API key y puede estar sujeto a cuota/rate limit según el proyecto de Google AI Studio.

### Opción B: Hugging Face local

```env
GENERATOR_PROVIDER=huggingface
HUGGINGFACE_MODEL=google/flan-t5-base
HUGGINGFACE_DEVICE=cpu
HUGGINGFACE_MAX_TOKENS=64
HUGGINGFACE_TEMPERATURE=0.7

KAGGLE_USERNAME=tu_usuario
KAGGLE_KEY=tu_kaggle_key
```

Para modelos públicos como `google/flan-t5-base` no hace falta token de Hugging Face. El modelo se descarga la primera vez y luego se usa desde la cache local.

## 🧠 Backend Generativo

El notebook usa `create_generator()` desde `src/generator.py`. Esta función lee `GENERATOR_PROVIDER` desde `.env` y crea el generador correspondiente:

- `gemini`: usa `GeminiGenerator` con `google-genai`, rate limiting y reintentos.
- `huggingface`: usa `HuggingFaceGenerator` con `transformers`, `AutoTokenizer` y `AutoModelForSeq2SeqLM`.

Ambos backends exponen la misma interfaz:

- `generate_title()`
- `generate_batch()`
- `request_count`

Esto permite cambiar de proveedor sin modificar el pipeline del notebook.

