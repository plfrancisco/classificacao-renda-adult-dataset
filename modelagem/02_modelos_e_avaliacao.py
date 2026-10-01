"""Geração e avaliação dos quatro modelos de classificação na base Adult.

Parte da base já limpa (ver ../limpeza/02_limpeza_dados.py), prepara os
dados para a modelagem, treina os quatro modelos comparados e avalia cada
um por validação cruzada e no conjunto de teste.

Modelos comparados: DummyClassifier (referência), LogisticRegression,
DecisionTreeClassifier e RandomForestClassifier.

Saídas:
    ../figuras/: matrizes de confusão, curva ROC e boxplots da validação
        cruzada.
    ../resultados/: tabelas de métricas em CSV, usadas no relatório.

Uso:
    python 02_modelos_e_avaliacao.py
"""

from pathlib import Path
from typing import Dict, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

SEED = 42

PASTA_ATIVIDADE = Path(__file__).resolve().parent.parent
PASTA_FIGURAS = PASTA_ATIVIDADE / "figuras"
PASTA_RESULTADOS = PASTA_ATIVIDADE / "resultados"
PASTA_FIGURAS.mkdir(exist_ok=True)
PASTA_RESULTADOS.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid")

# Nome exibido no relatório -> nome aceito pelo parâmetro `scoring` do
# scikit-learn. Todas as métricas são calculadas em relação à classe
# positiva (">50K"), a classe minoritária e a mais relevante para o problema.
METRICAS: Dict[str, str] = {
    "Acurácia": "accuracy",
    "Precisão": "precision",
    "Recall": "recall",
    "F1": "f1",
    "AUC-ROC": "roc_auc",
}


def preparar_dados() -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, ColumnTransformer]:
    """Carrega a base limpa e separa treino/teste para a modelagem.

    Descarta `education` (duplica `education-num`, mesmo atributo como
    texto e como código numérico) e `fnlwgt` (peso estatístico do censo,
    sem relação direta com a renda da pessoa). A divisão treino/teste é
    estratificada por `income`, para preservar a proporção de pessoas com
    renda alta nos dois conjuntos, já que a base é desbalanceada (ver
    01_analise_exploratoria.py).

    Returns:
        Tupla `(X_treino, X_teste, y_treino, y_teste, pre_processamento)`,
        em que `pre_processamento` é o `ColumnTransformer` que padroniza as
        colunas numéricas e aplica one-hot encoding às categóricas.
    """
    df = pd.read_csv(PASTA_ATIVIDADE / "dados" / "adult_limpo.csv")
    df_modelo = df.drop(columns=["education", "fnlwgt"])

    colunas_categoricas = (
        df_modelo.select_dtypes(include="object").columns.drop("income").tolist()
    )
    colunas_numericas = df_modelo.select_dtypes(include="number").columns.tolist()

    X = df_modelo.drop(columns="income")
    y = (df_modelo["income"] == ">50K").astype(int)

    X_treino, X_teste, y_treino, y_teste = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )

    pre_processamento = ColumnTransformer(
        [
            ("num", StandardScaler(), colunas_numericas),
            ("cat", OneHotEncoder(handle_unknown="ignore"), colunas_categoricas),
        ]
    )

    print("Treino:", X_treino.shape, "| Teste:", X_teste.shape)
    print(
        "Proporção da classe positiva (>50K) -- treino:",
        round(y_treino.mean(), 3),
        "| teste:",
        round(y_teste.mean(), 3),
    )
    return X_treino, X_teste, y_treino, y_teste, pre_processamento


def montar_pipelines(
    pre_processamento: ColumnTransformer,
) -> Tuple[Dict[str, BaseEstimator], Dict[str, Pipeline]]:
    """Monta os quatro modelos comparados, cada um dentro de um Pipeline.

    Embutir o pré-processamento no Pipeline garante que o ajuste do
    `StandardScaler` e do `OneHotEncoder` use só os dados de treino em cada
    fold da validação cruzada, sem vazar informação do fold de validação.

    Args:
        pre_processamento: `ColumnTransformer` compartilhado por todos os
            modelos (ver `preparar_dados`).

    Returns:
        Tupla `(modelos, pipelines)`: `modelos` mapeia o nome de cada
        algoritmo ao estimador "puro" (usado para relatar os hiperparâmetros
        no texto do relatório); `pipelines` mapeia o mesmo nome ao Pipeline
        completo (pré-processamento + modelo), usado no treino e na
        avaliação.
    """
    modelos: Dict[str, BaseEstimator] = {
        "DummyClassifier": DummyClassifier(strategy="most_frequent"),
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=SEED),
        "DecisionTreeClassifier": DecisionTreeClassifier(max_depth=10, random_state=SEED),
        "RandomForestClassifier": RandomForestClassifier(
            n_estimators=200, max_depth=10, random_state=SEED
        ),
    }
    pipelines = {
        nome: Pipeline([("pre", pre_processamento), ("modelo", modelo)])
        for nome, modelo in modelos.items()
    }
    return modelos, pipelines


def validacao_cruzada(
    pipelines: Dict[str, Pipeline], X_treino: pd.DataFrame, y_treino: pd.Series
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Executa a validação cruzada estratificada de 5 partições.

    A estratificação mantém, em cada partição, a mesma proporção das
    classes `<=50K`/`>50K` do conjunto de treino completo -- importante por
    a base ser desbalanceada.

    Args:
        pipelines: Pipelines a avaliar, uma por modelo (ver `montar_pipelines`).
        X_treino: Variáveis preditoras do conjunto de treino.
        y_treino: Alvo do conjunto de treino.

    Returns:
        Tupla `(cv_folds, tabela_cv)`: `cv_folds` tem uma linha por
        combinação de modelo e partição (usada nos boxplots); `tabela_cv`
        tem a média e o desvio-padrão de cada métrica por modelo (usada no
        relatório).
    """
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    linhas = []
    for nome, pipe in pipelines.items():
        resultado = cross_validate(pipe, X_treino, y_treino, cv=skf, scoring=METRICAS)
        for fold in range(skf.get_n_splits()):
            linha = {"Modelo": nome, "Fold": fold + 1}
            # cross_validate nomeia as colunas de retorno como "test_<chave>",
            # onde <chave> é a chave do dicionário METRICAS (ex.: "test_Acurácia").
            for rotulo in METRICAS:
                linha[rotulo] = resultado[f"test_{rotulo}"][fold]
            linhas.append(linha)

    cv_folds = pd.DataFrame(linhas)
    colunas = list(METRICAS)
    tabela_cv = cv_folds.groupby("Modelo")[colunas].agg(["mean", "std"]).round(3)
    tabela_cv = tabela_cv.loc[list(pipelines)]
    return cv_folds, tabela_cv


def avaliacao_teste(
    pipelines: Dict[str, Pipeline],
    X_treino: pd.DataFrame,
    y_treino: pd.Series,
    X_teste: pd.DataFrame,
    y_teste: pd.Series,
) -> Tuple[pd.DataFrame, Dict[str, "pd.Series[int]"], Dict[str, "pd.Series[float]"]]:
    """Treina cada modelo em todo o treino e avalia uma única vez no teste.

    Args:
        pipelines: Pipelines a treinar e avaliar, uma por modelo.
        X_treino: Variáveis preditoras do conjunto de treino.
        y_treino: Alvo do conjunto de treino.
        X_teste: Variáveis preditoras do conjunto de teste (não usado em
            nenhum ajuste até esta chamada).
        y_teste: Alvo do conjunto de teste.

    Returns:
        Tupla `(tabela_teste, previsoes, probabilidades)`: `tabela_teste`
        tem as métricas de cada modelo no teste; `previsoes` mapeia o nome
        do modelo ao rótulo previsto (0/1) de cada pessoa do teste, usado
        nas matrizes de confusão; `probabilidades` mapeia o nome do modelo
        à probabilidade prevista da classe `>50K`, usada na curva ROC.
    """
    linhas = []
    previsoes, probabilidades = {}, {}
    for nome, pipe in pipelines.items():
        pipe.fit(X_treino, y_treino)
        predicao = pipe.predict(X_teste)
        previsoes[nome] = predicao

        # O DummyClassifier("most_frequent") não implementa predict_proba de
        # forma informativa para a AUC-ROC; nesse caso, a própria previsão
        # (sempre 0) é usada como "probabilidade", o que resulta,
        # corretamente, em AUC-ROC = 0.5 (separação aleatória).
        if hasattr(pipe, "predict_proba"):
            probabilidade = pipe.predict_proba(X_teste)[:, 1]
        else:
            probabilidade = predicao.astype(float)
        probabilidades[nome] = probabilidade

        linhas.append(
            {
                "Modelo": nome,
                "Acurácia": accuracy_score(y_teste, predicao),
                "Precisão": precision_score(y_teste, predicao, zero_division=0),
                "Recall": recall_score(y_teste, predicao),
                "F1": f1_score(y_teste, predicao),
                "AUC-ROC": roc_auc_score(y_teste, probabilidade),
            }
        )

    tabela_teste = pd.DataFrame(linhas).set_index("Modelo").round(3).loc[list(pipelines)]
    return tabela_teste, previsoes, probabilidades


def gerar_figuras(
    cv_folds: pd.DataFrame,
    pipelines: Dict[str, Pipeline],
    y_teste: pd.Series,
    previsoes: Dict[str, "pd.Series[int]"],
    probabilidades: Dict[str, "pd.Series[float]"],
) -> None:
    """Gera e salva, em PASTA_FIGURAS, os três gráficos usados no relatório.

    Args:
        cv_folds: Métricas por partição da validação cruzada (ver
            `validacao_cruzada`), usadas nos boxplots.
        pipelines: Pipelines treinados, na ordem em que devem aparecer nos
            gráficos.
        y_teste: Alvo real do conjunto de teste.
        previsoes: Rótulo previsto por modelo, para as matrizes de confusão.
        probabilidades: Probabilidade prevista da classe `>50K` por modelo,
            para a curva ROC.
    """
    fig, eixos = plt.subplots(1, 5, figsize=(20, 4))
    for eixo, metrica in zip(eixos, METRICAS):
        sns.boxplot(data=cv_folds, x="Modelo", y=metrica, order=list(pipelines), ax=eixo, color="#4C72B0")
        eixo.set_title(metrica)
        eixo.set_xlabel("")
        eixo.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(PASTA_FIGURAS / "fig3_validacao_cruzada.png", dpi=200)
    plt.close(fig)

    fig, eixos = plt.subplots(1, 4, figsize=(18, 4))
    for eixo, nome in zip(eixos, pipelines):
        matriz = confusion_matrix(y_teste, previsoes[nome])
        ConfusionMatrixDisplay(matriz, display_labels=["<=50K", ">50K"]).plot(ax=eixo, colorbar=False)
        eixo.set_title(nome)
    fig.tight_layout()
    fig.savefig(PASTA_FIGURAS / "fig4_matrizes_confusao.png", dpi=200)
    plt.close(fig)

    fig, eixo = plt.subplots(figsize=(6, 6))
    for nome in pipelines:
        RocCurveDisplay.from_predictions(y_teste, probabilidades[nome], name=nome, ax=eixo)
    eixo.plot([0, 1], [0, 1], "k--", lw=1, label="Classificador aleatório")
    eixo.set_title("Curva ROC no conjunto de teste")
    eixo.legend()
    fig.tight_layout()
    fig.savefig(PASTA_FIGURAS / "fig5_curva_roc.png", dpi=200)
    plt.close(fig)


def main() -> None:
    """Executa o fluxo completo: prepara os dados, treina, avalia, gera as
    figuras e salva as tabelas de resultado em PASTA_RESULTADOS.
    """
    X_treino, X_teste, y_treino, y_teste, pre_processamento = preparar_dados()
    _, pipelines = montar_pipelines(pre_processamento)

    cv_folds, tabela_cv = validacao_cruzada(pipelines, X_treino, y_treino)
    print("\nMétricas na validação cruzada (média ± desvio-padrão):")
    print(tabela_cv)

    tabela_teste, previsoes, probabilidades = avaliacao_teste(
        pipelines, X_treino, y_treino, X_teste, y_teste
    )
    print("\nMétricas no conjunto de teste:")
    print(tabela_teste)

    gerar_figuras(cv_folds, pipelines, y_teste, previsoes, probabilidades)

    tabela_cv.to_csv(PASTA_RESULTADOS / "tabela_validacao_cruzada.csv")
    tabela_teste.to_csv(PASTA_RESULTADOS / "tabela_teste.csv")
    cv_folds.to_csv(PASTA_RESULTADOS / "validacao_cruzada_por_fold.csv", index=False)
    print(f"\nTabelas salvas em: {PASTA_RESULTADOS}")
    print(f"Figuras salvas em: {PASTA_FIGURAS}")


if __name__ == "__main__":
    main()
