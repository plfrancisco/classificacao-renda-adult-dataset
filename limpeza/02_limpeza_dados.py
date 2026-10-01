"""
Passo 2 — Limpeza da base Adult (UCI Machine Learning Repository).

Parte dos problemas identificados no passo 1 (01_verificacao_qualidade.py) e
aplica as correções:

1. Junta `adult.data` (treino) e `adult.test` (teste) em uma única tabela
   (a divisão treino/teste usada na modelagem é feita de forma própria e
   controlada, e não pela divisão original do UCI).
2. Padroniza a coluna alvo `income` (o arquivo de teste traz os rótulos com
   um ponto final, ex.: ">50K.").
3. Remove linhas com valores ausentes (marcados como "?" nas colunas
   `workclass`, `occupation` e `native-country`).
4. Remove linhas duplicadas.
5. Corrige os erros de grafia em `native-country`, usando uma comparação
   real contra a lista de países do pycountry (ver verificador_paises.py) --
   e não uma lista de erros conhecidos de antemão.
6. Salva o resultado em ../dados/adult_limpo.csv.

O passo 3 (03_verificacao_profunda.py) confirma que essas correções
funcionaram e faz uma auditoria mais aprofundada sobre o resultado.

Execução:
    python 02_limpeza_dados.py
"""

import pandas as pd

from comum import PASTA_DADOS, carregar_bruto
from verificador_paises import ReferenciaPaises, auditar, sugerir_correcoes


def relatar_ausentes(df: pd.DataFrame) -> None:
    """Imprime, por coluna, a quantidade e o percentual de valores ausentes.

    Args:
        df: DataFrame a diagnosticar (antes da limpeza).
    """
    ausentes = df.isna().sum()
    ausentes = ausentes[ausentes > 0]
    print("Valores ausentes por coluna (antes da limpeza):")
    for coluna, quantidade in ausentes.items():
        print(f"  - {coluna}: {quantidade} ({100 * quantidade / len(df):.1f}%)")


def remover_ausentes_e_duplicatas(df: pd.DataFrame) -> pd.DataFrame:
    """Remove linhas com valores ausentes e duplicatas exatas.

    Decisão de projeto: as três colunas com valores ausentes (`workclass`,
    `occupation`, `native-country`) se sobrepõem bastante -- quem não
    informa `workclass` também costuma não informar `occupation` -- então a
    remoção elimina cerca de 7% das linhas, não a soma das três.

    Args:
        df: Base concatenada, ainda sem limpeza (ver `carregar_bruto`).

    Returns:
        Cópia do DataFrame sem linhas ausentes/duplicadas, com índice
        reiniciado e espaços residuais removidos das colunas de texto.
    """
    n_antes = len(df)
    df = df.dropna().copy()
    n_depois_na = len(df)

    df = df.drop_duplicates()
    n_depois_dup = len(df)

    print(f"\nLinhas antes da limpeza: {n_antes}")
    print(f"Linhas após remover ausentes: {n_depois_na} "
          f"(-{n_antes - n_depois_na}, {100 * (n_antes - n_depois_na) / n_antes:.1f}%)")
    print(f"Linhas após remover duplicatas: {n_depois_dup} "
          f"(-{n_depois_na - n_depois_dup})")

    colunas_texto = df.select_dtypes(include=["object", "str"]).columns
    for coluna in colunas_texto:
        df[coluna] = df[coluna].str.strip()

    return df.reset_index(drop=True)


def corrigir_nomes_de_pais(df: pd.DataFrame) -> pd.DataFrame:
    """Corrige os erros de grafia em `native-country`.

    Compara cada valor contra a lista real de países do pycountry (ver
    `verificador_paises.py`). Só são corrigidos automaticamente os valores
    classificados como `"possivel_erro"` (grafia parecida o bastante com um
    único país para ser, com confiança, um erro de digitação, ex.:
    `"Columbia"` -> `"Colombia"`). Valores `"ambiguo"` (podem ser mais de um
    país, ex.: `"South"`) ou `"nao_reconhecido"` (não batem com nenhum país,
    ex.: `"England"`, que é um país constituinte do Reino Unido, não um
    membro da ISO 3166) são mantidos como estão, por não haver uma correção
    segura e automática.

    Args:
        df: Base já sem ausentes/duplicatas (ver `remover_ausentes_e_duplicatas`).

    Returns:
        Cópia do DataFrame com `native-country` corrigido.
    """
    referencia = ReferenciaPaises()
    valores_unicos = df["native-country"].unique()
    resultado = auditar(valores_unicos, referencia)
    correcoes = sugerir_correcoes(valores_unicos, referencia)

    df = df.copy()
    df["native-country"] = df["native-country"].replace(correcoes)

    print("\nCorreções de ortografia aplicadas em native-country "
          "(detectadas automaticamente contra a lista de países do pycountry):")
    if correcoes:
        for errado, certo in correcoes.items():
            qtd = (df["native-country"] == certo).sum()
            print(f"  - '{errado}' -> '{certo}' ({qtd} registros)")
    else:
        print("  (nenhuma correção automática foi necessária)")

    mantidos = {v: r for v, r in resultado.items() if r["status"] in ("ambiguo", "nao_reconhecido")}
    if mantidos:
        print("\nValores mantidos sem alteração (sem correção automática segura):")
        for v, r in sorted(mantidos.items()):
            qtd = (df["native-country"] == v).sum()
            extra = f", candidatos possíveis: {r['sugestoes']}" if r.get("sugestoes") else ""
            print(f"  - '{v}' ({qtd} registros) -- {r['status']}{extra}")

    return df


def main() -> None:
    """Executa o fluxo de limpeza completo e salva o resultado em
    `ARQUIVO_LIMPO` (ver `comum.py`).
    """
    df = carregar_bruto()
    print(f"Base original (treino + teste do UCI): {len(df)} linhas, {df.shape[1]} colunas")
    relatar_ausentes(df)

    df_limpo = remover_ausentes_e_duplicatas(df)
    df_limpo = corrigir_nomes_de_pais(df_limpo)

    print("\nDistribuição do alvo (income) após a limpeza:")
    print(df_limpo["income"].value_counts())
    print(df_limpo["income"].value_counts(normalize=True).round(3))

    caminho_saida = PASTA_DADOS / "adult_limpo.csv"
    df_limpo.to_csv(caminho_saida, index=False)
    print(f"\nBase limpa salva em: {caminho_saida}")
    print("Confirme o resultado executando: python 03_verificacao_profunda.py")


if __name__ == "__main__":
    main()
