# Classificação de Renda — Adult Census Income

![Python](https://img.shields.io/badge/Python-3.14-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.9-F7931E?logo=scikitlearn&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-3.0-150458?logo=pandas&logoColor=white)
![status](https://img.shields.io/badge/status-concluído-success)

Pipeline de ciência de dados que compara quatro algoritmos de classificação
para prever se a renda anual de uma pessoa ultrapassa US$ 50 mil, a partir
de variáveis demográficas e de trabalho do [censo dos EUA de 1994
(dataset Adult / Census Income)](https://archive.ics.uci.edu/dataset/2/adult).

## Sumário

- [Visão geral](#visão-geral)
- [Resultados](#resultados)
- [Estrutura do projeto](#estrutura-do-projeto)
- [Como executar](#como-executar)
- [Metodologia](#metodologia)
- [Stack técnica](#stack-técnica)
- [Autor](#autor)

## Visão geral

O pipeline cobre o fluxo completo de um projeto de classificação:
diagnóstico e limpeza de dados, engenharia de atributos, treino e avaliação
comparativa de modelos. O destaque técnico é a etapa de qualidade de dados:
em vez de corrigir erros de digitação a partir de uma lista fixada à mão, um
módulo dedicado (`verificador_paises.py`) valida cada valor da coluna de
país contra uma base de referência real (`pycountry`), usando comparação
aproximada de texto para detectar e corrigir grafias inválidas de forma
automática e auditável.

## Resultados

Avaliação no conjunto de teste (9.035 registros, classes desbalanceadas:
75,2% / 24,8%):

| Modelo | Acurácia | Precisão | Recall | F1 | AUC-ROC |
|---|---|---|---|---|---|
| DummyClassifier (baseline) | 0,752 | 0,000 | 0,000 | 0,000 | 0,500 |
| Logistic Regression | 0,845 | 0,732 | 0,588 | 0,652 | 0,901 |
| Decision Tree | 0,851 | 0,770 | 0,571 | 0,655 | 0,897 |
| **Random Forest** | **0,852** | **0,811** | 0,527 | 0,639 | **0,908** |

Discussão completa dos resultados, trade-offs entre precisão e recall, e
análise de limitações estão no relatório técnico do projeto (entregue à
parte, fora deste repositório).

## Estrutura do projeto

```
.
├── dados/            # Base bruta (UCI) e base tratada (adult_limpo.csv)
├── limpeza/          # Pipeline de qualidade de dados (diagnóstico → limpeza → auditoria)
│   ├── 01_verificacao_qualidade.py
│   ├── 02_limpeza_dados.py
│   ├── 03_verificacao_profunda.py
│   ├── comum.py
│   └── verificador_paises.py
├── modelagem/        # Análise exploratória, treino e avaliação dos modelos
│   ├── 01_analise_exploratoria.py
│   └── 02_modelos_e_avaliacao.py
├── figuras/          # Gráficos gerados (PNG)
├── resultados/       # Métricas geradas (CSV)
└── requirements.txt
```

## Como executar

```bash
# 1. Instalar dependências
pip install -r requirements.txt

# 2. Pipeline de dados
cd limpeza
python 01_verificacao_qualidade.py   # diagnóstico da base bruta (read-only)
python 02_limpeza_dados.py           # limpeza + correção automática de ortografia
python 03_verificacao_profunda.py    # valida a limpeza e audita relações entre colunas

# 3. Modelagem
cd ../modelagem
python 01_analise_exploratoria.py
python 02_modelos_e_avaliacao.py     # treina e avalia os 4 modelos
```

Cada etapa é independente e lê o artefato gerado pela anterior, então pode
ser reexecutada isoladamente.

## Metodologia

- **Pré-processamento:** `ColumnTransformer` com padronização (`StandardScaler`)
  para variáveis numéricas e one-hot encoding para categóricas, encapsulado
  em `Pipeline` para evitar vazamento de dados entre treino e teste.
- **Split:** 80/20 estratificado por classe, `random_state` fixo para
  reprodutibilidade.
- **Validação:** validação cruzada estratificada (5 folds) no treino +
  avaliação única no conjunto de teste.
- **Métricas:** acurácia, precisão, recall, F1 e AUC-ROC — escolhidas pelo
  desbalanceamento de classes do dataset.
- **Qualidade de dados:** detecção automática (não hardcoded) de erros de
  ortografia por similaridade de texto contra uma base de países real,
  checagens de consistência entre colunas, faixas numéricas plausíveis e
  outliers via IQR.

## Stack técnica

Python · pandas · NumPy · scikit-learn · Matplotlib · Seaborn · pycountry

## Autor

**Pedro Lucas Francisco**
