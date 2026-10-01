"""
Passo 1 — Diagnóstico de qualidade da base Adult ANTES da limpeza.

Lê os arquivos originais do UCI (adult.data + adult.test), sem nenhuma
alteração, e roda um conjunto de checagens para levantar os problemas da
base: valores ausentes, duplicatas, erros de ortografia/pontuação, valores
fora de faixa etc. Este script serve para justificar, com números, cada
decisão tomada no passo 2 (02_limpeza_dados.py) — ele NÃO corrige nada.

É esperado que várias checagens falhem aqui: é exatamente isso que o passo 2
vai corrigir, e o que o passo 3 vai confirmar que foi corrigido.

Execução:
    python 01_verificacao_qualidade.py
"""

from comum import VerificadorQualidade, carregar_bruto


def main() -> None:
    """Carrega a base crua e roda `VerificadorQualidade.rodar_tudo()` sobre
    ela, imprimindo o diagnóstico completo no terminal.
    """
    df = carregar_bruto()
    print(f"Base original (treino + teste do UCI, sem limpeza): "
          f"{len(df)} linhas, {df.shape[1]} colunas\n")

    verificador = VerificadorQualidade(df)
    passou = verificador.rodar_tudo()

    print()
    if passou:
        print("Resultado: nenhum problema encontrado na base original "
              "(inesperado — confira se os arquivos em ../dados/ estão corretos).")
    else:
        print(f"Resultado: {len(verificador.falhas)} problema(s) encontrado(s) na base original:")
        for f in verificador.falhas:
            print(f"  - {f}")
        print("\nEsses problemas são tratados no passo 2 (02_limpeza_dados.py).")


if __name__ == "__main__":
    main()
