"""Código compartilhado pelos três scripts de limpeza (01, 02 e 03).

Mantém em um só lugar as constantes e a lógica de verificação usadas por
mais de um script, para que não fiquem duplicadas.

Contém:
    `carregar_bruto`: lê os arquivos originais do UCI (adult.data +
        adult.test).
    `FAIXAS_NUMERICAS` e `PADRAO_CARACTERES_VALIDOS`: parâmetros das
        checagens de qualidade.
    `VerificadorQualidade`: roda as checagens de qualidade, usada tanto no
        diagnóstico inicial (01) quanto na auditoria pós-limpeza (03). A
        checagem de ortografia de países usa `verificador_paises.py`, que
        compara os valores da base contra a lista real de países do
        pycountry, em vez de uma lista de erros conhecidos de antemão.
"""

import re
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from verificador_paises import ReferenciaPaises, auditar

PASTA_DADOS = Path(__file__).resolve().parent.parent / "dados"
ARQUIVO_LIMPO = PASTA_DADOS / "adult_limpo.csv"

COLUNAS = [
    "age", "workclass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race", "sex",
    "capital-gain", "capital-loss", "hours-per-week", "native-country",
    "income",
]

# Faixas plausíveis para as colunas numéricas (limites conhecidos da base
# Adult/Census Income; valores fora disso indicariam erro de digitação).
FAIXAS_NUMERICAS: Dict[str, Tuple[int, int]] = {
    "age": (0, 120),
    "hours-per-week": (1, 99),
    "capital-gain": (0, 99_999),
    "capital-loss": (0, 99_999),
    "education-num": (1, 16),
}

# Caracteres aceitos nas colunas de texto: letras, dígitos, espaço, hífen e
# os símbolos usados de propósito em algumas categorias originais do UCI
# (ex.: parênteses em "Outlying-US(Guam-USVI-etc)", "&" em "Trinadad&Tobago",
# "<" e "=" em valores como "<=50K").
PADRAO_CARACTERES_VALIDOS = re.compile(r"^[A-Za-z0-9\s\-()<>=&.]+$")


def carregar_bruto() -> pd.DataFrame:
    """Lê os dois arquivos originais do UCI, sem nenhuma limpeza, e os junta.

    Os arquivos `adult.data` (treino) e `adult.test` (teste), baixados de
    https://archive.ics.uci.edu/dataset/2/adult, são mantidos em ../dados/
    sem nenhuma alteração.

    Returns:
        DataFrame com os registros de treino e teste concatenados, com a
        coluna `income` já padronizada (o arquivo de teste traz os rótulos
        com um ponto final, ex.: ">50K.", removido aqui).
    """
    treino = pd.read_csv(
        PASTA_DADOS / "adult.data",
        names=COLUNAS,
        skipinitialspace=True,
        na_values="?",
    )
    teste = pd.read_csv(
        PASTA_DADOS / "adult.test",
        names=COLUNAS,
        skipinitialspace=True,
        na_values="?",
        skiprows=1,  # a primeira linha do adult.test é um comentário, não um dado
    )
    # Padronizado aqui, mesmo na checagem "bruta", porque sem isso a junção
    # treino+teste teria 4 categorias de alvo em vez de 2, o que atrapalharia
    # qualquer checagem feita sobre a coluna `income`.
    teste["income"] = teste["income"].str.rstrip(".")

    return pd.concat([treino, teste], ignore_index=True)


class VerificadorQualidade:
    """Roda um conjunto de checagens de qualidade sobre um DataFrame.

    Cada método `checar_*` registra, via `registrar`, se a checagem passou
    ou falhou; `rodar_tudo` executa todas em sequência.

    Attributes:
        df: DataFrame a verificar.
        falhas: Descrição de cada checagem que falhou, preenchida conforme
            os métodos `checar_*` são chamados.
    """

    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df
        self.falhas: List[str] = []

    def registrar(self, descricao: str, ok: bool, detalhe: str = "") -> None:
        """Imprime o resultado de uma checagem e guarda as que falharam.

        Args:
            descricao: Nome da checagem (ex.: "Nenhum valor ausente").
            ok: `True` se a checagem passou, `False` se falhou.
            detalhe: Texto complementar impresso após a descrição (ex.:
                contagem ou lista dos valores problemáticos).
        """
        status = "OK" if ok else "FALHOU"
        linha = f"[{status}] {descricao}"
        if detalhe:
            linha += f" -- {detalhe}"
        print(linha)
        if not ok:
            self.falhas.append(descricao)

    def checar_ausentes(self) -> None:
        """Falha se houver qualquer valor ausente (NaN) no DataFrame."""
        total = int(self.df.isna().sum().sum())
        self.registrar("Nenhum valor ausente", total == 0, f"{total} valores ausentes encontrados")

    def checar_duplicatas(self) -> None:
        """Falha se houver linhas inteiramente duplicadas."""
        total = int(self.df.duplicated().sum())
        self.registrar("Nenhuma linha duplicada", total == 0, f"{total} duplicatas encontradas")

    def checar_espacos_residuais(self) -> None:
        """Falha se alguma coluna de texto tiver espaço sobrando no início
        ou no fim de algum valor (ex.: `"Private "` em vez de `"Private"`).
        """
        colunas_texto = self.df.select_dtypes(include=["object", "str"]).columns
        problemas = []
        for coluna in colunas_texto:
            serie = self.df[coluna].dropna()
            afetados = serie[serie != serie.str.strip()]
            if len(afetados):
                problemas.append(f"{coluna} ({len(afetados)} valores)")
        self.registrar(
            "Nenhum espaço residual no início/fim do texto",
            len(problemas) == 0,
            "; ".join(problemas),
        )

    def checar_duplicatas_por_capitalizacao(self) -> None:
        """Falha se uma coluna de texto tiver a mesma categoria escrita de
        duas formas que só diferem em maiúsculas/minúsculas (ex.:
        `"Private"` e `"private"` contadas como categorias diferentes).
        """
        colunas_texto = self.df.select_dtypes(include=["object", "str"]).columns
        problemas = []
        for coluna in colunas_texto:
            grupos: Dict[str, List[str]] = {}
            for valor in self.df[coluna].dropna().unique():
                grupos.setdefault(valor.lower(), []).append(valor)
            conflitos = {chave: valores for chave, valores in grupos.items() if len(valores) > 1}
            if conflitos:
                problemas.append(f"{coluna}: {conflitos}")
        self.registrar(
            "Nenhuma categoria duplicada só por maiúsculas/minúsculas",
            len(problemas) == 0,
            "; ".join(problemas),
        )

    def checar_caracteres_validos(self) -> None:
        """Falha se alguma coluna de texto tiver caractere fora do esperado
        (ver `PADRAO_CARACTERES_VALIDOS`), o que indicaria corrupção de
        codificação ou um símbolo não previsto.
        """
        colunas_texto = self.df.select_dtypes(include=["object", "str"]).columns
        problemas = []
        for coluna in colunas_texto:
            invalidos = [
                valor for valor in self.df[coluna].dropna().unique()
                if not PADRAO_CARACTERES_VALIDOS.match(valor)
            ]
            if invalidos:
                problemas.append(f"{coluna}: {invalidos}")
        self.registrar(
            "Nenhum caractere fora do esperado nas colunas de texto",
            len(problemas) == 0,
            "; ".join(problemas),
        )

    def checar_ortografia_paises(self) -> None:
        """Audita `native-country` contra a lista real de países do
        pycountry (ver `verificador_paises.py`), sem usar nenhuma lista de
        erros feita à mão.

        A checagem só falha quando há valores classificados como
        `"possivel_erro"` (grafia parecida o bastante com um único país
        para ser, com confiança, um erro de digitação). Valores
        `"ambiguo"` ou `"nao_reconhecido"` são só informativos, porque
        podem ser categorias legítimas do dataset que não são, de fato,
        nomes de país -- por exemplo, `"England"`, que é um país
        constituinte do Reino Unido, não um membro da ISO 3166.
        """
        if "native-country" not in self.df.columns:
            return

        referencia = ReferenciaPaises()
        resultado = auditar(self.df["native-country"].dropna().unique(), referencia)

        erros = {valor: r for valor, r in resultado.items() if r["status"] == "possivel_erro"}
        self.registrar(
            "Nenhuma grafia de país não reconhecida pelo pycountry (ISO 3166)",
            len(erros) == 0,
            "; ".join(f"'{valor}' -> possivelmente '{r['sugestoes'][0]}'" for valor, r in erros.items()),
        )

        revisar = {valor: r for valor, r in resultado.items() if r["status"] in ("ambiguo", "nao_reconhecido")}
        if revisar:
            print("  [INFO] Valores de native-country fora da lista ISO 3166, "
                  "mas não corrigidos automaticamente (ver relatório):")
            for valor, r in sorted(revisar.items()):
                candidatos = f" -- candidatos: {r['sugestoes']}" if r.get("sugestoes") else ""
                print(f"    - '{valor}': {r['status']}{candidatos}")

    def checar_consistencia_educacao(self) -> None:
        """Falha se algum valor de `education` (texto) aparecer associado a
        mais de um código em `education-num`, o que indicaria inconsistência
        entre as duas representações da escolaridade.
        """
        if not {"education", "education-num"}.issubset(self.df.columns):
            return
        mapa = (
            self.df.dropna(subset=["education", "education-num"])
            .groupby("education")["education-num"]
            .nunique()
        )
        inconsistentes = mapa[mapa > 1]
        self.registrar(
            "education e education-num são consistentes entre si",
            len(inconsistentes) == 0,
            f"níveis com mais de um código: {dict(inconsistentes)}",
        )

    def checar_faixas_numericas(self) -> None:
        """Falha se alguma coluna numérica tiver valor fora da faixa
        plausível definida em `FAIXAS_NUMERICAS` (um indício de erro de
        digitação, não de um valor genuinamente extremo).
        """
        for coluna, (minimo, maximo) in FAIXAS_NUMERICAS.items():
            if coluna not in self.df.columns:
                continue
            serie = self.df[coluna].dropna()
            fora_da_faixa = serie[(serie < minimo) | (serie > maximo)]
            self.registrar(
                f"{coluna} dentro da faixa esperada [{minimo}, {maximo}]",
                len(fora_da_faixa) == 0,
                f"{len(fora_da_faixa)} valores fora da faixa",
            )

    def rodar_tudo(self) -> bool:
        """Executa todas as checagens, na ordem.

        Returns:
            `True` se todas as checagens passaram, `False` se alguma falhou
            (nesse caso, `self.falhas` lista quais).
        """
        self.checar_ausentes()
        self.checar_duplicatas()
        self.checar_espacos_residuais()
        self.checar_duplicatas_por_capitalizacao()
        self.checar_caracteres_validos()
        self.checar_ortografia_paises()
        self.checar_consistencia_educacao()
        self.checar_faixas_numericas()
        return len(self.falhas) == 0
