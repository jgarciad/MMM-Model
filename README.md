# MELI DS Case — Hierarchical Forecasting & Causal Inference

## Estructura del Proyecto

```
meli_final/
├── src/                                        ← código modular (importable)
│   ├── data_pipeline/
│   │   ├── __init__.py
│   │   ├── load_and_clean.py                  ← carga, limpieza, agregación mensual
│   │   └── ecommerce_orders_hierarchical.csv  ← COLOCAR AQUÍ EL CSV
│   ├── feature_transformers/
│   │   ├── __init__.py
│   │   └── eda_hierarchical.py                ← EDA, ranking de nodos, masking, contribution_table
│   └── model_training/
│       ├── __init__.py
│       ├── hierarchical_forecast.py           ← Prophet + grid search + Bottom-Up (con cache)
│       ├── model_benchmark.py                 ← benchmark Prophet vs. 5 modelos alternativos
│       ├── causal_inference.py                ← elasticidad + modelo Synthetic Control
├── notebooks/
│   └── InformeTecnico.ipynb            ← informe técnico completo (importa src/)
│   └── ResumenEjecutivo.pptx            ← explicación vista negocio
├── plots/                                     ← generado automáticamente
├── outputs/                                   ← generado automáticamente (incluye caches)
├── main.py                                    ← orquestador — ejecutar desde aquí
└── README.md
```

---

## Configuración del Entorno (`meli_env`)

###  venv + pip

```bash
python -m venv meli_env
source meli_env/bin/activate        # Mac/Linux
# meli_env\Scripts\activate         # Windows

pip install pandas numpy matplotlib scipy statsmodels 
pip install prophet

# En caso de que marque error la ejecución, es posible que sea porque no se tiene Stan y se ejecuta una sola vez lo siguiente:
python -c "import cmdstanpy; cmdstanpy.install_cmdstan()"
```

### Registrar el kernel en Jupyter

```bash
conda activate meli_env
pip install ipykernel
python -m ipykernel install --user --name meli_env --display-name "Python 3 (meli_env)"
```

---

## Ejecución

### Script de producción

El pipeline tiene 5 pasos  y soporta los siguientes flags:

| Comando | Qué hace | Tiempo estimado |
|---------|----------|-----------------|
| `python main.py` | Carga todos los caches | ~segundos |
| `python main.py --force-retrain` | Re-entrena Prophet con grid search | ~90 min |
| `python main.py --force-benchmark` | Re-corre el benchmark de modelos | ~1 min |
| `python main.py --force-all` | Re-entrena Prophet **y** re-corre el benchmark | ~91 min |

> Los caches están incluidos en `outputs/` para evaluación inmediata sin re-entrenar.
> Usar `--force-all` solo para reproducir el entrenamiento desde cero. El tiempo estimado de 90 minutos  varia según las especificaciones de la computadora.

### Flujo del pipeline

```
[1/5] load_and_clean.py        → limpieza y agregación mensual
[2/5] eda_hierarchical.py      → ranking de nodos, masking, plots 01-03
[3/5] hierarchical_forecast.py → Prophet + hiperparámetros + plots 04
[4/5] model_benchmark.py       → benchmark vs. alternativos + plots 15-17
[5/5] causal_inference.py      → elasticidad + SC + plots 05-06
```

### Outputs generados

**Plots:**

| Archivo | Descripción |
|---------|-------------|
| `plots/01_masking_effect.png` | Efecto masking — total → región → subcategoría |
| `plots/02_node_heatmap.png` | Heatmap % cambio post Jul-2023 por nodo |
| `plots/03_problem_node_deep_dive.png` | Problem Node vs. mejor nodo |
| `plots/04_forecast.png` | Forecast 6M Prophet con IC 95% |
| `plots/05_elasticity.png` | Elasticidad precio-demanda (log-log) |
| `plots/06a_sc_diagnostics.png` | Resultados de ajuste del modelo SC |
| `plots/06b_sc_prospective.png` | Proyección estimada |
| `plots/15_model_benchmark.png` | sMAPE por modelo — comparativa completa |
| `plots/16_metrics_summary_table.png` | Tabla resumen: mejor modelo por serie + agregados |
| `plots/17_business_projection.png` | Proyección 6M sin intervención |

**CSVs y JSONs en `outputs/`:**

| Archivo | Descripción |
|---------|-------------|
| `node_ranking.csv` | Ranking de los 10 nodos por % cambio post Jul-2023 |
| `forecast_subcategory_region.csv` | Forecast Prophet por nodo |
| `forecast_region.csv` | Forecast reconciliado por región |
| `forecast_total.csv` | Forecast total plataforma |
| `train_results.csv` | Hiperparámetros seleccionados por nodo (Prophet) |
| `model_evaluation.csv` | RMSE, MAPE, IC Prophet vs. baselines |
| `benchmark_full.csv` | **Cache del benchmark** — métricas de todos los modelos por serie |
| `benchmark_summary.csv` | Resumen: mejor modelo + agregados North/South/Total |
| `benchmark_forecast_6m.csv` | Forecast 6M del mejor modelo por serie |
| `contribution_table.csv` | Contribución de cada nodo al cambio total de revenue |
| `executive_summary.json` | Métricas clave del análisis completo |

### Informe Técnico

Se detalla de manera técnica los análisis realizados así como los modelos ajustados propuestos. 

```bash
 Abrir: notebooks/MELI_Informe_Tecnico.ipynb
 Kernel: Python 3 (meli_env)
 Kernel → Restart & Run All
```

---

## Hallazgos Principales

| # | Hallazgo | Evidencia |
|---|----------|-----------|
| 1 | **Problem Node:** North · Electronics · Smartphones | Caída −44.2% post Jul-2023 (único nodo >15%) |
| 2 | **Efecto Masking:** 30% | Decay $390K absorbido por crecimiento de otros 9 nodos |
| 3 | **Precio no es la causa** | Elasticidad p=0.365 (no sig.), precio estable $800 ± $15 |
| 4 | **Prophet no siempre es el mejor** | Seasonal Naive gana en Pharma/Decor; trend_month gana en Smartphones |
| 5 | **Benchmark total sMAPE: 8.3%** | Seleccionando el mejor modelo por serie vía holdout |
| 6 | **Descuento plano ineficiente** | 8/10 nodos sanos sacrificados sin necesidad |

---

## Arquitectura del Forecasting

### Por qué Bottom-Up

Con 10 series atómicas, Bottom-Up es preferible porque:
- Preserva la dinámica del Problem Node (Top-Down la diluye al redistribuir por proporciones)
- La reconciliación por suma directa es exacta — no introduce error adicional
- MinT requeriría estimar la matriz de covarianzas W con solo 36 meses — pero costo - beneficio

### Benchmark de Modelos

Se evaluaron 6 modelos en holdout de 6 meses. Criterio de selección: `sMAPE → WAPE → RMSE`.

| Modelo | Descripción | Cuándo gana |
|--------|-------------|-------------|
| `prophet_hw` | Prophet con grid search de hiperparámetros | Series con estacionalidad pronunciada |
| `snaive` | Mismo mes del año anterior | Series estables sin drift |
| `holt_linear` | Holt doble con tendencia amortiguada | Series con tendencia suave |
| `holt_winters_add` | HW aditivo | Series con estacionalidad moderada |
| `holt_winters_mul` | HW multiplicativo | Electronics con picos crecientes |
| `trend_month` | Tendencia lineal + dummies mensuales (OLS) | Series con deterioro lineal sostenido |

**Métricas finales bottom-up (holdout 6 meses):**

| Nivel | sMAPE | WAPE |
|-------|-------|------|
| Total plataforma | 8.3% | 8.7% |
| North | 17.1% | 18.1% |
| South | 8.2% | 7.7% |

### Sistema de Cache

El entrenamiento de Prophet tarda ~90 min. El proyecto tiene cache en tres niveles:

```
outputs/train_results.csv      ← hiperparámetros Prophet
outputs/forecast_*.csv         ← forecasts Prophet
outputs/benchmark_full.csv     ← métricas benchmark de todos los modelos
```


### Prophet — Hiperparámetros por Categoría

| Categoría | `seasonality_mode` | `changepoint_prior_scale` | `seasonality_prior_scale` |
|-----------|-------------------|--------------------------|--------------------------|
| Electronics | `multiplicative` | [0.05, 0.15, 0.30] | [5, 10, 15] |
| Pharma | `additive` | [0.01, 0.05, 0.10] | [2, 5, 10] |
| Home | `additive` | [0.05, 0.10, 0.20] | [3, 7, 10] |

Optimización: Grid Search × CV temporal expanding window
(`initial=548d`, `period=91d`, `horizon=91d`, métrica: RMSE promedio de folds).

### Causalidad

Se estima el contrafactual usando Synthetic Control donde se determina como regla de éxito:

* Acum 3M > 88u  -> efecto positivo (80% conf.)
* Acum 6M > 169u  -> efecto positivo (80% conf.)


---

## Resumen Ejecutivo 
Dentro de la carpeta Notebooks se encuentra una presentación ejecutiva orientado a mostrar resultados para el negocio. 
