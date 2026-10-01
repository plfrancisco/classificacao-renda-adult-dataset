# Classificação de renda na base Adult (Census Income)

Comparação de quatro algoritmos de classificação para prever se a renda
anual de uma pessoa é maior que 50 mil dólares, usando a base [Adult
(Census Income)](https://archive.ics.uci.edu/dataset/2/adult), do UCI
Machine Learning Repository. Projeto da disciplina de Aplicação de
Ciência de Dados.

## Resultados (conjunto de teste)

| Modelo | Acurácia | Precisão | Recall | F1 | AUC-ROC |
|---|---|---|---|---|---|
| DummyClassifier | 0,752 | 0,000 | 0,000 | 0,000 | 0,500 |
| LogisticRegression | 0,845 | 0,732 | 0,588 | 0,652 | 0,901 |
| DecisionTreeClassifier | 0,851 | 0,770 | 0,571 | 0,655 | 0,897 |
| RandomForestClassifier | 0,852 | 0,811 | 0,527 | 0,639 | 0,908 |

Metodologia, critérios de comparação e discussão completa estão no
relatório entregue separadamente (não incluído neste repositório).

## Estrutura do repositório

```
dados/            Base original do UCI (adult.data, adult.test, adult.names)
                  e a base limpa gerada por limpeza/02_limpeza_dados.py
                  (adult_limpo.csv).
limpeza/          Scripts de diagnóstico, limpeza e auditoria dos dados.
modelagem/        Scripts de análise exploratória, geração e avaliação
                  dos modelos.
figuras/          Gráficos gerados pelos scripts (PNG).
resultados/       Tabelas de métricas geradas pelos scripts (CSV).
requirements.txt  Dependências Python do projeto.
```

## Como reproduzir

```bash
pip install -r requirements.txt

cd limpeza
python 01_verificacao_qualidade.py   # diagnóstico da base original (não corrige nada)
python 02_limpeza_dados.py           # limpeza + correção de ortografia -> gera dados/adult_limpo.csv
python 03_verificacao_profunda.py    # confirma a limpeza e audita mais a fundo

cd ../modelagem
python 01_analise_exploratoria.py    # gera as figuras 1 e 2
python 02_modelos_e_avaliacao.py     # treina e avalia os quatro modelos
```

Cada script é independente e pode ser executado isoladamente, desde que os
anteriores já tenham sido rodados ao menos uma vez (cada um lê o resultado
salvo pelo anterior).

## Metodologia, em resumo

- **Algoritmos comparados:** DummyClassifier (referência), LogisticRegression,
  DecisionTreeClassifier e RandomForestClassifier, todos com o mesmo
  pré-processamento (padronização das variáveis numéricas e one-hot encoding
  das categóricas) e a mesma divisão treino/teste estratificada (80%/20%,
  `random_state = 42`).
- **Avaliação:** validação cruzada estratificada de 5 partições no treino,
  seguida de avaliação única no teste, com as métricas acurácia, precisão,
  recall, F1 e AUC-ROC (a base é desbalanceada: 75,2% `<=50K`, 24,8% `>50K`).
- **Limpeza de dados:** remoção de valores ausentes e duplicatas, e correção
  de erros de ortografia em `native-country`, detectados automaticamente por
  comparação com a lista de países da biblioteca `pycountry` (ver
  `limpeza/verificador_paises.py`) -- não por uma lista de erros fixada de
  antemão.

## Fonte dos dados

BECKER, B.; KOHAVI, R. **Adult**. UCI Machine Learning Repository, 1996.
<https://archive.ics.uci.edu/dataset/2/adult>.

## Autor

Pedro Lucas Francisco
