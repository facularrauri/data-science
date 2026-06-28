# 🗞️ Generación de Títulos de Noticias con IA Generativa

**UTN Facultad Regional La Plata — Ciencia de Datos 2026**

Pipeline completo para generar titulares periodísticos usando el dataset BBC News Summary,
embeddings locales con GPU, ChromaDB y la API de Gemini.

---

## 🚀 Inicio Rápido

```powershell
# 1. Activar el entorno virtual
.\.venv\Scripts\Activate.ps1

# 2. Abrir Jupyter Lab
jupyter lab notebooks/proyecto_titulos_noticias.ipynb
```

## 📁 Estructura del Proyecto

```
datos/
├── .env                    # API Keys (no commitear)
├── requirements.txt        # Dependencias
├── src/                    # Módulos Python
│   ├── config.py           # Singleton de configuración
│   ├── data_loader.py      # Carga del dataset BBC
│   ├── preprocessor.py     # NLP: tokenización, lematización
│   ├── embedder.py         # Embeddings semánticos (GPU)
│   ├── vector_store.py     # ChromaDB (Repository Pattern)
│   ├── prompt_builder.py   # Estrategias de prompting (Strategy Pattern)
│   ├── generator.py        # Integración Gemini (Facade Pattern)
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
| **Facade** | `generator.py` | API simplificada sobre Gemini |
| **Repository** | `vector_store.py` | Abstracción de ChromaDB |
| **Pipeline** | Notebook | Orquestación secuencial |

## ⚙️ Stack Tecnológico

| Componente | Tecnología |
|------------|------------|
| Dataset | BBC News Summary (Kaggle) |
| NLP | NLTK (tokenización, stopwords, lematización) |
| Embeddings | sentence-transformers + GPU (CUDA) |
| Base vectorial | ChromaDB embedded |
| Modelo generativo | Google Gemini 2.0 Flash |
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

```env
GEMINI_API_KEY=tu_api_key_aqui
KAGGLE_USERNAME=tu_usuario
KAGGLE_KEY=tu_kaggle_key
```
