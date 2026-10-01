"""Verificação de ortografia de `native-country` contra uma referência real.

Compara cada valor da coluna `native-country` com a lista de nomes de país
da biblioteca `pycountry` (nomes oficiais segundo a norma ISO 3166, mais
países históricos, como a antiga Iugoslávia). A classificação de cada valor
("reconhecido", "possível erro" etc.) é decidida por comparação de texto,
não por uma lista de erros conhecidos combinada de antemão -- o objetivo é
que o script encontre os erros por conta própria.

Uso:
    from verificador_paises import ReferenciaPaises, classificar, para_estilo_dataset

    referencia = ReferenciaPaises()
    resultado = classificar("Columbia", referencia)
    # {'status': 'possivel_erro', 'sugestoes': ['Colombia'], 'motivo': 'grafia parecida'}
"""

import difflib
import re
from typing import Dict, Iterable, List, Literal, Optional, Set, TypedDict

import pycountry

# Abaixo de 4 caracteres, uma "primeira palavra" em comum é fraca demais para
# indicar algo (ex.: países cujo nome começa com "A", "Al" etc.).
TAMANHO_MINIMO_PRIMEIRA_PALAVRA = 4

# Quão parecidos dois nomes precisam ser (0 a 1) para o script sugerir que um
# é uma grafia errada do outro. Calibrado para pegar erros reais (ex.:
# "Columbia" x "Colombia" = 0.88) sem sugerir correções erradas para nomes
# que só por acaso se parecem um pouco (ex.: "England" x "Greenland" = 0.75).
LIMIAR_SIMILARIDADE = 0.78

StatusPais = Literal["reconhecido", "possivel_erro", "ambiguo", "nao_reconhecido"]


class ClassificacaoPais(TypedDict, total=False):
    """Resultado de `classificar` para um único valor de `native-country`."""

    status: StatusPais
    sugestoes: List[str]
    motivo: str


def normalizar(texto: str) -> str:
    """Reduz um texto a letras minúsculas sem espaços nem pontuação.

    Isso faz com que "Trinidad and Tobago", "Trinidad-and-Tobago" e
    "TRINIDAD AND TOBAGO" sejam todos comparados da mesma forma.

    Args:
        texto: Texto original.

    Returns:
        Texto normalizado, só com `[a-z]`.
    """
    return re.sub(r"[^a-z]", "", texto.lower())


def palavras(texto: str) -> List[str]:
    """Separa um texto em palavras, em minúsculas.

    Args:
        texto: Texto original.

    Returns:
        Lista de palavras (sequências de letras) encontradas no texto.
    """
    return re.findall(r"[A-Za-z]+", texto.lower())


def para_estilo_dataset(nome: str) -> str:
    """Converte um nome de país para o estilo usado na base Adult.

    Args:
        nome: Nome de país com palavras separadas por espaço, ex.:
            "Trinidad and Tobago".

    Returns:
        Mesmo nome com as palavras separadas por hífen, ex.:
        "Trinidad-and-Tobago", no padrão das demais categorias do dataset.
    """
    return "-".join(nome.split())


class ReferenciaPaises:
    """Lista de nomes de país reconhecidos, construída a partir do pycountry
    (países atuais e históricos) -- não a partir de uma lista de erros feita
    à mão.

    Attributes:
        normalizado_para_id: Nome de país normalizado -> identificador do
            país (código ISO alpha-2/alpha-3, ou o próprio nome para países
            históricos sem código).
        normalizado_para_nome: Nome de país normalizado -> nome de exibição
            original (ex.: "colombia" -> "Colombia").
        primeira_palavra_para_ids: Primeira palavra de um nome composto ->
            conjunto de identificadores de país cujo nome começa com ela
            (ex.: "south" -> {South Africa, South Korea, South Sudan, ...}).
        id_para_nome_preferido: Identificador do país -> nome mais curto e
            mais usual para exibição (o `common_name` do pycountry, quando
            existe, senão o `name`).
    """

    def __init__(self) -> None:
        self.normalizado_para_id: Dict[str, str] = {}
        self.normalizado_para_nome: Dict[str, str] = {}
        self.primeira_palavra_para_ids: Dict[str, Set[str]] = {}
        self.id_para_nome_preferido: Dict[str, str] = {}

        for colecao in (pycountry.countries, pycountry.historic_countries):
            for pais in colecao:
                identificador = (
                    getattr(pais, "alpha_2", None)
                    or getattr(pais, "alpha_3", None)
                    or pais.name
                )
                if identificador not in self.id_para_nome_preferido:
                    preferido = getattr(pais, "common_name", None) or pais.name
                    self.id_para_nome_preferido[identificador] = preferido.split(",")[0].strip()

                nomes = []
                for campo in ("name", "common_name", "official_name"):
                    valor = getattr(pais, campo, None)
                    if valor:
                        nomes.append(valor)
                        # Nomes ISO no formato "Taiwan, Province of China" ou
                        # "Korea, Republic of" também valem pela parte antes
                        # da vírgula, que costuma ser o nome mais usual.
                        if "," in valor:
                            nomes.append(valor.split(",")[0])

                for nome in nomes:
                    chave = normalizar(nome)
                    if not chave:
                        continue
                    self.normalizado_para_nome.setdefault(chave, nome)
                    self.normalizado_para_id.setdefault(chave, identificador)

                    palavras_do_nome = palavras(nome)
                    if len(palavras_do_nome) > 1:
                        self.primeira_palavra_para_ids.setdefault(palavras_do_nome[0], set()).add(identificador)

    def nomes_normalizados(self) -> Iterable[str]:
        """Retorna todos os nomes de país conhecidos, já normalizados."""
        return self.normalizado_para_nome.keys()


def classificar(
    valor: str, referencia: ReferenciaPaises, limiar: float = LIMIAR_SIMILARIDADE
) -> ClassificacaoPais:
    """Classifica um único valor de `native-country` contra a referência.

    A classificação segue três etapas, na ordem: (1) checa se o valor já é
    um nome de país reconhecido; (2) checa se o valor é a primeira palavra
    de um nome composto (nome incompleto, ex.: "Hong" de "Hong Kong"); (3)
    usa similaridade de texto (`difflib`) para achar grafias parecidas.

    Args:
        valor: Valor de `native-country` a classificar (ex.: "Columbia").
        referencia: Referência de países construída por `ReferenciaPaises`.
        limiar: Similaridade mínima (0 a 1) para considerar duas grafias
            parecidas o bastante para sugerir uma correção.

    Returns:
        Um `ClassificacaoPais` com um dos status:

        * `"reconhecido"`: o valor já é um nome de país válido.
        * `"possivel_erro"`: o valor parece uma grafia errada, ou um nome
          incompleto, de um único país identificável -- `sugestoes` traz o
          nome correto.
        * `"ambiguo"`: o valor pode corresponder a mais de um país
          diferente -- não é seguro corrigir automaticamente.
        * `"nao_reconhecido"`: o valor não bate, nem de forma aproximada,
          com nenhum país da referência (pode ser uma categoria legítima,
          mas não padronizada, como uma região em vez de um país).
    """
    chave = normalizar(valor)
    if chave in referencia.normalizado_para_nome:
        return {"status": "reconhecido"}

    palavras_do_valor = palavras(valor)
    primeira_palavra = palavras_do_valor[0] if palavras_do_valor else chave
    if (
        len(primeira_palavra) >= TAMANHO_MINIMO_PRIMEIRA_PALAVRA
        and primeira_palavra in referencia.primeira_palavra_para_ids
    ):
        ids_candidatos = referencia.primeira_palavra_para_ids[primeira_palavra]
        if len(ids_candidatos) == 1:
            (identificador,) = ids_candidatos
            sugestao = referencia.id_para_nome_preferido[identificador]
            return {"status": "possivel_erro", "sugestoes": [sugestao], "motivo": "nome incompleto"}
        sugestoes = sorted({referencia.id_para_nome_preferido[i] for i in ids_candidatos})
        return {"status": "ambiguo", "sugestoes": sugestoes, "motivo": "várias correspondências"}

    candidatos = difflib.get_close_matches(chave, referencia.nomes_normalizados(), n=3, cutoff=limiar)
    if candidatos:
        ids_candidatos = {referencia.normalizado_para_id[c] for c in candidatos}
        sugestoes = sorted({referencia.id_para_nome_preferido[i] for i in ids_candidatos})
        if len(sugestoes) == 1:
            return {"status": "possivel_erro", "sugestoes": sugestoes, "motivo": "grafia parecida"}
        return {"status": "ambiguo", "sugestoes": sugestoes, "motivo": "grafia parecida com mais de um país"}

    return {"status": "nao_reconhecido"}


def auditar(
    valores: Iterable[str], referencia: Optional[ReferenciaPaises] = None
) -> Dict[str, ClassificacaoPais]:
    """Classifica uma coleção de valores únicos de `native-country`.

    Args:
        valores: Valores a classificar (duplicatas são ignoradas).
        referencia: Referência de países já construída. Quando omitida, uma
            nova é criada (mais lento se chamado muitas vezes em sequência).

    Returns:
        Mapa `{valor_original: resultado_de_classificar(valor_original)}`.
    """
    referencia = referencia or ReferenciaPaises()
    return {valor: classificar(valor, referencia) for valor in sorted(set(valores))}


def sugerir_correcoes(
    valores: Iterable[str], referencia: Optional[ReferenciaPaises] = None
) -> Dict[str, str]:
    """Monta o mapa de correções automáticas para os erros de grafia certos.

    Args:
        valores: Valores de `native-country` a auditar.
        referencia: Referência de países já construída (ver `auditar`).

    Returns:
        Mapa `{valor_errado: valor_corrigido}`, já no estilo hífen do
        dataset, contendo só os valores classificados como `"possivel_erro"`
        (um único país candidato). Valores `"ambiguo"` ou `"nao_reconhecido"`
        não entram aqui, porque não há uma correção segura e automática a
        aplicar.
    """
    resultado = auditar(valores, referencia)
    correcoes = {}
    for valor, classificacao in resultado.items():
        if classificacao["status"] == "possivel_erro":
            (sugestao,) = classificacao["sugestoes"]
            correcoes[valor] = para_estilo_dataset(sugestao)
    return correcoes
