"""Especificações flexíveis de séries por período.

Toda grandeza anual de um projeto (receita, custo, quantidade, preço, percentual
...) é descrita por uma *especificação* (``Spec``) que este módulo resolve em
um vetor NumPy com um valor por período (1..n). São aceitas:

* um número → valor constante em todos os períodos;
* uma sequência de comprimento ``n`` → valores explícitos, período 1..n;
* um dicionário ``{período: valor}`` → valores esparsos (demais períodos = 0);
* um ``numpy.ndarray`` 2-D de forma ``(S, n)`` → uma trajetória por amostra
  (simulação de Monte Carlo);
* um ``numpy.ndarray`` de forma ``(S, 1)`` → escalar por amostra, replicado;
* uma função ``f(ctx)`` → recebe o contexto de cálculo (``ctx.t``, ``ctx.n``,
  ``ctx.receita`` ...) e devolve outra especificação;
* os objetos auxiliares deste módulo: :class:`Crescimento`, :class:`Variacoes`,
  :class:`Percentual`, :class:`Produto`, :class:`Soma` e :class:`Degraus`.

A convenção de formas é o que permite vetorizar simulações: parâmetros
escalares incertos entram como colunas ``(S, 1)`` e o *broadcasting* do NumPy
propaga a dimensão de amostras por todo o cálculo do fluxo de caixa.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

Spec = Any


class Contexto(dict):
    """Dicionário com acesso por atributo, usado na resolução de séries.

    Contém sempre ``n`` (número de períodos) e ``t`` (vetor 1..n). Durante o
    cálculo de um projeto recebe também as linhas já calculadas (``receita``,
    ``custo`` e cada item nomeado), o que permite escrever custos como
    percentual da receita ou funções de outras linhas.
    """

    def __getattr__(self, chave):
        try:
            return self[chave]
        except KeyError as exc:  # pragma: no cover - mensagem de erro
            raise AttributeError(chave) from exc

    def __setattr__(self, chave, valor):
        self[chave] = valor


def novo_contexto(n: int, **extras) -> Contexto:
    """Cria um contexto vazio para ``n`` períodos."""
    ctx = Contexto(n=n, t=np.arange(1, n + 1, dtype=float))
    ctx.update(extras)
    return ctx


# --------------------------------------------------------------------------- #
# Resolução
# --------------------------------------------------------------------------- #
def escalar(valor: Spec) -> float | np.ndarray:
    """Normaliza um parâmetro escalar: ``float`` ou coluna ``(S, 1)``.

    Aceita número, array 0-D, array de um elemento ou coluna ``(S, 1)``
    (um valor por amostra). Qualquer outra forma é rejeitada, pois não
    representa um escalar.
    """
    arr = np.asarray(valor, dtype=float)
    if arr.ndim == 0:
        return float(arr)
    if arr.ndim == 1 and arr.shape[0] == 1:
        return float(arr[0])
    if arr.ndim == 2 and arr.shape[1] == 1:
        return arr
    raise ValueError(
        f"esperado parâmetro escalar (número ou coluna (S, 1)); recebido forma {arr.shape}"
    )


def _ajustar(arr: np.ndarray, comprimento: int) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)
    if arr.ndim == 0:
        return np.full(comprimento, float(arr))
    if arr.ndim == 1:
        if arr.shape[0] == comprimento:
            return arr.astype(float, copy=True)
        if arr.shape[0] == 1:
            return np.full(comprimento, float(arr[0]))
        raise ValueError(
            f"série com {arr.shape[0]} valores; esperados {comprimento} "
            "(um por período)"
        )
    if arr.ndim == 2:
        if arr.shape[1] == comprimento:
            return arr.astype(float, copy=True)
        if arr.shape[1] == 1:
            return np.broadcast_to(arr, (arr.shape[0], comprimento)).astype(float)
        raise ValueError(
            f"matriz (S, {arr.shape[1]}); esperada (S, {comprimento}) ou (S, 1)"
        )
    raise ValueError(f"forma não suportada para série: {arr.shape}")


def _esparsa(mapa: Mapping, comprimento: int, inicio: int) -> np.ndarray:
    valores = {int(k): np.asarray(v, dtype=float) for k, v in mapa.items()}
    amostras = [v.shape[0] for v in valores.values() if v.ndim == 2]
    for k in valores:
        if not inicio <= k < inicio + comprimento:
            raise ValueError(
                f"período {k} fora do intervalo {inicio}..{inicio + comprimento - 1}"
            )
    if amostras:
        s = max(amostras)
        saida = np.zeros((s, comprimento))
        for k, v in valores.items():
            saida[:, k - inicio] = v.reshape(-1) if v.ndim else float(v)
    else:
        saida = np.zeros(comprimento)
        for k, v in valores.items():
            saida[k - inicio] = float(v)
    return saida


def resolver(spec: Spec, n: int, ctx: Contexto | None = None, inicio: int = 1) -> np.ndarray:
    """Resolve uma especificação em vetor de valores por período.

    Parameters
    ----------
    spec
        Especificação (ver docstring do módulo).
    n
        Número de períodos do projeto.
    ctx
        Contexto com linhas já calculadas (opcional).
    inicio
        Primeiro período coberto: ``1`` (séries operacionais, comprimento
        ``n``) ou ``0`` (fluxos que incluem o instante zero, comprimento
        ``n + 1``).

    Returns
    -------
    numpy.ndarray
        Forma ``(L,)`` ou ``(S, L)`` com ``L = n + 1 - inicio``.
    """
    comprimento = n + 1 - inicio
    if ctx is None:
        ctx = novo_contexto(n)
    if isinstance(spec, SerieBase):
        return _ajustar(spec.resolver(n, ctx), comprimento)
    if callable(spec):
        return resolver(spec(ctx), n, ctx, inicio)
    if isinstance(spec, Mapping):
        return _esparsa(spec, comprimento, inicio)
    return _ajustar(np.asarray(spec, dtype=float), comprimento)


# --------------------------------------------------------------------------- #
# Especificações auxiliares
# --------------------------------------------------------------------------- #
class SerieBase:
    """Classe-base das especificações compostas."""

    def resolver(self, n: int, ctx: Contexto) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    # Operadores para compor especificações de forma natural.
    def __add__(self, outro):
        return Soma(self, outro)

    __radd__ = __add__

    def __mul__(self, outro):
        return Produto(self, outro)

    __rmul__ = __mul__

    def __neg__(self):
        return Produto(self, -1.0)


@dataclass
class Crescimento(SerieBase):
    """Valor ``base`` no período 1 crescendo à ``taxa`` por período.

    ``taxa`` pode ser um número, uma série (taxa de cada período, a partir do
    2º) ou uma coluna ``(S, 1)``. ``periodos`` limita o número de períodos com
    crescimento (ex.: ``periodos=5`` cresce do período 2 ao 6 e estabiliza).

    Exemplo: ``Crescimento(60_000, 0.10, periodos=5)`` reproduz "60.000
    unidades com crescimento anual de 10% nos próximos 5 anos".
    """

    base: Spec
    taxa: Spec = 0.0
    periodos: int | None = None

    def resolver(self, n, ctx):
        base = escalar(self.base)
        g = resolver(self.taxa, n, ctx).copy()
        g[..., 0] = 0.0
        if self.periodos is not None:
            g[..., self.periodos + 1 :] = 0.0
        return base * np.cumprod(1.0 + g, axis=-1)


@dataclass
class Variacoes(SerieBase):
    """Valor ``base`` no período 1 com variações percentuais encadeadas.

    ``variacoes`` é um dicionário ``{período: variação}``. Exemplo (preço do
    Projeto 4 do caso Gama): ``Variacoes(20, {4: -0.05, 6: -0.04})`` dá
    20, 20, 20, 19, 19, 18,24.
    """

    base: Spec
    variacoes: Mapping[int, Spec] = field(default_factory=dict)

    def resolver(self, n, ctx):
        base = escalar(self.base)
        g = resolver(dict(self.variacoes), n, ctx)
        g[..., 0] = 0.0
        return base * np.cumprod(1.0 + g, axis=-1)


@dataclass
class Percentual(SerieBase):
    """Percentual de outra linha do contexto (por padrão a receita total).

    ``pct`` pode variar por período: ``Percentual([0.67, 0.63, ...])``.
    ``de`` nomeia a linha de referência: ``"receita"``, ``"custo"`` ou o nome
    de qualquer item já calculado.
    """

    pct: Spec
    de: str = "receita"

    def resolver(self, n, ctx):
        if self.de not in ctx:
            raise KeyError(
                f"linha '{self.de}' ainda não calculada no contexto; "
                f"disponíveis: {sorted(k for k in ctx if k not in ('n', 't'))}"
            )
        return resolver(self.pct, n, ctx) * ctx[self.de]


class Produto(SerieBase):
    """Produto de especificações, período a período (ex.: quantidade × preço)."""

    def __init__(self, *fatores: Spec):
        self.fatores = fatores

    def resolver(self, n, ctx):
        saida = np.ones(n)
        for f in self.fatores:
            saida = saida * resolver(f, n, ctx)
        return saida

    def __repr__(self):
        return f"Produto{self.fatores!r}"


class Soma(SerieBase):
    """Soma de especificações, período a período."""

    def __init__(self, *parcelas: Spec):
        self.parcelas = parcelas

    def resolver(self, n, ctx):
        saida = np.zeros(n)
        for p in self.parcelas:
            saida = saida + resolver(p, n, ctx)
        return saida

    def __repr__(self):
        return f"Soma{self.parcelas!r}"


@dataclass
class Degraus(SerieBase):
    """Série em degraus: ``{período_inicial: valor}`` mantém o valor até mudar.

    Exemplo: ``Degraus({1: 100, 4: 120})`` → 100, 100, 100, 120, 120, ...
    """

    degraus: Mapping[int, Spec]

    def resolver(self, n, ctx):
        chaves = sorted(int(k) for k in self.degraus)
        if not chaves or chaves[0] != 1:
            raise ValueError("Degraus precisa definir o valor do período 1")
        valores = [escalar(self.degraus[k]) for k in chaves]
        saida = None
        for i, k in enumerate(chaves):
            fim = chaves[i + 1] if i + 1 < len(chaves) else n + 1
            mascara = np.zeros(n)
            mascara[k - 1 : fim - 1] = 1.0
            termo = valores[i] * mascara
            saida = termo if saida is None else saida + termo
        return saida


def unidades_x_preco(unidades: Spec, preco: Spec) -> Produto:
    """Atalho legível para receita = unidades × preço."""
    return Produto(unidades, preco)
