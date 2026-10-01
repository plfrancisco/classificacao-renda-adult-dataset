"""Análise exploratória da base Adult, já limpa (ver ../limpeza/02_limpeza_dados.py).

Gera as figuras usadas no relatório (em ../figuras/) e imprime no terminal
os números citados no texto (distribuição do alvo, tipos de variável etc.).

Uso:
    python 01_analise_exploratoria.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

PASTA_ATIVIDADE = Path(__file__).resolve().parent.parent
PASTA_FIGURAS = PASTA_ATIVIDADE / "figuras"
PASTA_FIGURAS.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid")


def plotar_distribuicao_do_alvo(df: pd.DataFrame) -> None:
    """Salva um gráfico de barras com a contagem de cada classe de renda.

    Args:
        df: Base limpa, com a coluna `income`.
    """
    contagem = df["income"].value_counts()

    fig, eixo = plt.subplots(figsize=(5, 4))
    sns.barplot(x=contagem.index, y=contagem.values, ax=eixo, color="#4C72B0")
    eixo.set_xlabel("Renda (income)")
    eixo.set_ylabel("Número de pessoas")
    for i, valor in enumerate(contagem.values):
        eixo.text(i, valor, f"{valor}\n({valor / len(df):.1%})", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(PASTA_FIGURAS / "fig1_distribuicao_income.png", dpi=200)
    plt.close(fig)


def plotar_idade_e_horas_por_renda(df: pd.DataFrame) -> None:
    """Salva os histogramas de idade e de horas trabalhadas, por classe de
    renda, para visualizar se essas variáveis separam bem as duas classes.

    Args:
        df: Base limpa, com as colunas `age`, `hours-per-week` e `income`.
    """
    fig, eixos = plt.subplots(1, 2, figsize=(10, 4))
    sns.histplot(data=df, x="age", hue="income", kde=True, ax=eixos[0], bins=30)
    sns.histplot(data=df, x="hours-per-week", hue="income", kde=True, ax=eixos[1], bins=30)
    eixos[0].set_title("Idade por classe de renda")
    eixos[1].set_title("Horas trabalhadas por semana por classe de renda")
    fig.tight_layout()
    fig.savefig(PASTA_FIGURAS / "fig2_idade_horas_por_renda.png", dpi=200)
    plt.close(fig)


def main() -> None:
    """Carrega a base limpa, imprime o diagnóstico textual e gera as duas
    figuras da análise exploratória.
    """
    df = pd.read_csv(PASTA_ATIVIDADE / "dados" / "adult_limpo.csv")
    print(f"Base: {df.shape[0]} linhas, {df.shape[1]} colunas")
    print(f"Valores ausentes: {df.isna().sum().sum()} (a limpeza já removeu os ausentes)")

    contagem = df["income"].value_counts()
    proporcao = df["income"].value_counts(normalize=True)
    print("\nDistribuição do alvo (income):")
    print(contagem)
    print(proporcao.round(3))

    plotar_distribuicao_do_alvo(df)
    plotar_idade_e_horas_por_renda(df)

    colunas_categoricas = df.select_dtypes(include="object").columns.drop("income").tolist()
    colunas_numericas = df.select_dtypes(include="number").columns.tolist()
    print("\nVariáveis categóricas:", colunas_categoricas)
    print("Variáveis numéricas:", colunas_numericas)

    print(f"\nFiguras salvas em: {PASTA_FIGURAS}")


if __name__ == "__main__":
    main()
