"""
Passo 3 — Verificação profunda da base Adult DEPOIS da limpeza.

Faz duas coisas:

A) Repete, sobre ../dados/adult_limpo.csv, as mesmas checagens do passo 1
   (01_verificacao_qualidade.py), para confirmar que a limpeza (passo 2)
   realmente corrigiu os problemas encontrados. Aqui, diferente do passo 1,
   é esperado que TODAS as checagens passem.

B) Vai além das checagens básicas, com uma auditoria mais aprofundada,
   olhando para relações entre colunas e não só para cada coluna isolada:
   1. Pessoas com `capital-gain` e `capital-loss` positivos ao mesmo tempo
      (logicamente estranho: o ganho e a perda de capital são o resultado
      do mesmo ano fiscal).
   2. Combinação implausível entre `age` e `education-num` (uma pessoa não
      pode ter mais anos de estudo do que seria fisicamente possível para
      a idade dela).
   3. `relationship` inconsistente com `sex` (ex.: "Husband" com sexo
      diferente de "Male"), reportado como informação, não como erro, pois
      pode refletir como o censo de 1994 registrava a informação.
   4. Valores atípicos (outliers) nas colunas numéricas, pela regra do
      intervalo interquartil (IQR), reportados como informação: outlier não
      é, por si só, um erro de dados (ex.: `capital-gain` tem um valor
      "teto" de 99999 usado de propósito pelo Census Bureau, para limitar a
      divulgação de rendas muito altas).

Os itens de B) são informativos: eles não fazem o script "falhar", porque
não são, necessariamente, erros de dados — são pontos para discutir no
relatório.

Execução:
    python 03_verificacao_profunda.py

Código de saída: 0 se a parte A) passar por completo, 1 se alguma checagem
básica ainda falhar (nesse caso, a limpeza do passo 2 não funcionou).
"""

import pandas as pd

from comum import ARQUIVO_LIMPO, VerificadorQualidade


def checar_ganho_e_perda_simultaneos(df: pd.DataFrame) -> None:
    """Reporta quantos registros têm `capital-gain` e `capital-loss`
    positivos ao mesmo tempo -- logicamente estranho, já que as duas colunas
    descrevem o resultado do mesmo ano fiscal.

    Args:
        df: Base limpa (ver ../dados/adult_limpo.csv).
    """
    ambos_positivos = df[(df["capital-gain"] > 0) & (df["capital-loss"] > 0)]
    print(f"[INFO] Pessoas com capital-gain e capital-loss positivos ao mesmo tempo: "
          f"{len(ambos_positivos)} ({100 * len(ambos_positivos) / len(df):.2f}%)")


def checar_idade_vs_escolaridade(df: pd.DataFrame) -> None:
    """Reporta combinações implausíveis entre `age` e `education-num`.

    Usa uma heurística, não uma regra oficial do censo: para concluir um
    nível de ensino com código `education-num`, uma pessoa precisaria de
    pelo menos `education-num + 5` anos de idade (considerando o início da
    escolarização por volta dos 5-6 anos, sem nenhuma repetência).

    Args:
        df: Base limpa (ver ../dados/adult_limpo.csv).
    """
    limite_minimo = df["education-num"] + 5
    implausiveis = df[df["age"] < limite_minimo]
    print(f"[INFO] Combinações implausíveis de age x education-num "
          f"(idade menor que education-num + 5): {len(implausiveis)} "
          f"({100 * len(implausiveis) / len(df):.2f}%)")
    if len(implausiveis):
        print("       Exemplos:")
        print(implausiveis[["age", "education", "education-num"]].head(5).to_string(index=False))


def checar_relacao_vs_sexo(df: pd.DataFrame) -> None:
    """Reporta registros em que `relationship` diverge do `sex` esperado.

    Reportado como informação, não como erro, pois pode refletir como o
    censo de 1994 registrava a informação, e não um erro de digitação.

    Args:
        df: Base limpa (ver ../dados/adult_limpo.csv).
    """
    sexo_esperado_por_relacao = {"Husband": "Male", "Wife": "Female"}
    for relacao, sexo_esperado in sexo_esperado_por_relacao.items():
        subconjunto = df[df["relationship"] == relacao]
        divergentes = subconjunto[subconjunto["sex"] != sexo_esperado]
        print(f"[INFO] relationship='{relacao}' com sex != '{sexo_esperado}': "
              f"{len(divergentes)} de {len(subconjunto)} registros")


def checar_outliers_iqr(df: pd.DataFrame) -> None:
    """Reporta outliers nas colunas numéricas pela regra do intervalo
    interquartil (IQR): valores fora de `[Q1 - 1.5*IQR, Q3 + 1.5*IQR]`.

    Um outlier não é, por si só, um erro de dados -- por exemplo,
    `capital-gain` tem um valor "teto" de 99999 usado de propósito pelo
    Census Bureau para limitar a divulgação de rendas muito altas, o que
    aparece aqui como outlier sem ser um erro.

    Args:
        df: Base limpa (ver ../dados/adult_limpo.csv).
    """
    colunas_numericas = ["age", "hours-per-week", "capital-gain", "capital-loss"]
    for coluna in colunas_numericas:
        q1, q3 = df[coluna].quantile([0.25, 0.75])
        iqr = q3 - q1
        limite_inferior, limite_superior = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        outliers = df[(df[coluna] < limite_inferior) | (df[coluna] > limite_superior)]
        print(f"[INFO] Outliers em '{coluna}' (regra do IQR, fora de "
              f"[{limite_inferior:.1f}, {limite_superior:.1f}]): "
              f"{len(outliers)} ({100 * len(outliers) / len(df):.2f}%)")


def main() -> int:
    if not ARQUIVO_LIMPO.exists():
        print(f"Arquivo não encontrado: {ARQUIVO_LIMPO}")
        print("Execute antes: python 02_limpeza_dados.py")
        return 1

    df = pd.read_csv(ARQUIVO_LIMPO)
    print(f"Verificando {ARQUIVO_LIMPO.name} ({len(df)} linhas, {df.shape[1]} colunas)")

    print("\n--- A) Confirmação das checagens básicas (iguais ao passo 1) ---")
    verificador = VerificadorQualidade(df)
    passou = verificador.rodar_tudo()

    print("\n--- B) Auditoria mais profunda (checagens entre colunas) ---")
    checar_ganho_e_perda_simultaneos(df)
    checar_idade_vs_escolaridade(df)
    checar_relacao_vs_sexo(df)
    checar_outliers_iqr(df)

    print()
    if passou:
        print("Resultado da parte A): todas as checagens básicas passaram "
              "-- a limpeza do passo 2 funcionou.")
    else:
        print(f"Resultado da parte A): {len(verificador.falhas)} checagem(ns) "
              "básica(s) ainda falharam:")
        for f in verificador.falhas:
            print(f"  - {f}")

    return 0 if passou else 1


if __name__ == "__main__":
    raise SystemExit(main())
