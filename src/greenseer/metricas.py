"""Indicadores de avaliação de investimentos.

Todas as funções aceitam um :class:`~greenseer.cashflow.FluxoDeCaixa`, um
projeto (objeto com método ``fluxo()``) ou um array ``(n+1,)`` / ``(S, n+1)``
e devolvem um escalar (fluxo único) ou um vetor ``(S,)`` (fluxos amostrados).

Indicadores disponíveis (nomenclatura do curso de Análise de Investimentos):

* ``vp``   — valor atual dos fluxos (VAFLCx);
* ``vpl``  — valor presente líquido;
* ``tir``  — taxa interna de retorno;
* ``tirm`` — TIR modificada (reinvestimento a uma taxa explícita);
* ``payback`` — simples ou ajustado (descontado), com interpolação;
* ``ibc``  — índice benefício/custo (índice de lucratividade);
* ``rau``  — resultado anual uniforme (equivalente anual do VPL);
* ``roia`` — retorno adicional sobre o investimento;
* ``taxa_fisher`` — taxa de interseção (TIR do fluxo marginal).
"""

from __future__ import annotations

import warnings
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from .cashflow import FluxoDeCaixa, alinhar, como_fluxo, como_fluxos

LIMITES_TIR = (-0.9, 10.0)


# --------------------------------------------------------------------------- #
# Auxiliares
# --------------------------------------------------------------------------- #
def _v(fluxo) -> np.ndarray:
    if isinstance(fluxo, FluxoDeCaixa):
        return fluxo.valores
    if hasattr(fluxo, "fluxo") and callable(fluxo.fluxo):
        return fluxo.fluxo().valores
    return np.asarray(fluxo, dtype=float)


def _saida(x: np.ndarray, era_1d: bool):
    x = np.asarray(x, dtype=float)
    if era_1d:
        return float(x.reshape(-1)[0]) if x.size == 1 else x
    return x


def _horner(v: np.ndarray, taxas) -> np.ndarray:
    """VPL por esquema de Horner em x = 1/(1+i), estável para fluxos longos.

    ``v``: (n+1,) ou (S, n+1); ``taxas``: escalar, (R,) ou (S,).
    Retorna (R,) para v 1-D com várias taxas; (S,) para v 2-D com taxa escalar
    ou (S,) taxas.
    """
    v = np.asarray(v, dtype=float)
    taxas = np.asarray(taxas, dtype=float)
    x = 1.0 / (1.0 + taxas)
    if v.ndim == 1:
        acc = np.zeros_like(x, dtype=float)
        for coef in v[::-1]:
            acc = acc * x + coef
        return acc
    # v 2-D
    x = np.broadcast_to(x, (v.shape[0],)) if x.ndim == 0 else x
    acc = np.zeros(v.shape[0])
    for k in range(v.shape[1] - 1, -1, -1):
        acc = acc * x + v[:, k]
    return acc


# --------------------------------------------------------------------------- #
# Fatores de equivalência
# --------------------------------------------------------------------------- #
def fator_valor_atual(taxa, n: int):
    """Fator (P/A, i, n): valor atual de uma série uniforme unitária."""
    taxa = np.asarray(taxa, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        f = np.where(taxa == 0, float(n), (1 - (1 + taxa) ** (-n)) / taxa)
    return float(f) if f.ndim == 0 else f


def fator_recuperacao_capital(taxa, n: int):
    """Fator (A/P, i, n): série uniforme equivalente a um valor presente unitário."""
    return 1.0 / fator_valor_atual(taxa, n)


def fator_acumulacao(taxa, n: int):
    """Fator (F/A, i, n): valor futuro de uma série uniforme unitária."""
    taxa = np.asarray(taxa, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        f = np.where(taxa == 0, float(n), ((1 + taxa) ** n - 1) / taxa)
    return float(f) if f.ndim == 0 else f


def fator_desconto(taxa, t):
    """Fator (P/F, i, t) = (1+i)^-t."""
    return (1.0 + np.asarray(taxa, dtype=float)) ** (-np.asarray(t, dtype=float))


# --------------------------------------------------------------------------- #
# Valor presente
# --------------------------------------------------------------------------- #
def vp(fluxo, taxa, a_partir_de: int = 1):
    """Valor atual dos fluxos a partir do período ``a_partir_de`` (VAFLCx)."""
    v = _v(fluxo)
    era_1d = v.ndim == 1
    v2 = np.atleast_2d(v).copy()
    v2[:, :a_partir_de] = 0.0
    return _saida(_horner(v2, taxa), era_1d)


def vpl(fluxo, taxa):
    """Valor presente líquido (inclui o instante zero)."""
    v = _v(fluxo)
    era_1d = v.ndim == 1
    return _saida(_horner(np.atleast_2d(v), taxa), era_1d)


def vpl_perfil(fluxo, taxas):
    """VPL para uma grade de taxas: (R,) para fluxo 1-D, (S, R) para 2-D."""
    v = _v(fluxo)
    taxas = np.asarray(taxas, dtype=float)
    if v.ndim == 1:
        return _horner(v, taxas)
    return np.stack([_horner(v, r) for r in taxas], axis=1)


def valor_futuro(fluxo, taxa):
    """Valor dos fluxos capitalizados até o último período."""
    v = _v(fluxo)
    n = v.shape[-1] - 1
    return vpl(v, taxa) * (1 + taxa) ** n


# --------------------------------------------------------------------------- #
# Taxa interna de retorno
# --------------------------------------------------------------------------- #
def _tir_1d(v: np.ndarray, todas: bool, limites=LIMITES_TIR):
    lo, hi = limites
    grade = np.concatenate(
        [np.linspace(lo, -0.01, 60), np.linspace(0.0, 1.0, 401), np.linspace(1.02, hi, 150)]
    )
    f = _horner(v, grade)
    raizes: list[float] = []
    for i in range(len(grade) - 1):
        fi, fj = f[i], f[i + 1]
        if fi == 0.0:
            raizes.append(float(grade[i]))
        elif fi * fj < 0:
            raizes.append(
                float(brentq(lambda r: _horner(v, r), grade[i], grade[i + 1], xtol=1e-12, maxiter=200))
            )
    if f[-1] == 0.0:
        raizes.append(float(grade[-1]))
    raizes = sorted(set(round(r, 12) for r in raizes))
    if todas:
        return np.array(raizes)
    if not raizes:
        return float("nan")
    if len(raizes) > 1:
        warnings.warn(
            f"fluxo com múltiplas TIR {['%.2f%%' % (100 * r) for r in raizes]}; "
            "devolvendo a de menor valor absoluto — prefira VPL/TIRM.",
            stacklevel=3,
        )
    return min(raizes, key=abs)


def _tir_2d(v: np.ndarray, limites=LIMITES_TIR) -> np.ndarray:
    s = v.shape[0]
    lo = np.full(s, limites[0])
    hi = np.full(s, limites[1])
    flo = _horner(v, lo)
    fhi = _horner(v, hi)
    ok = np.sign(flo) != np.sign(fhi)
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        fmid = _horner(v, mid)
        mesmo = np.sign(fmid) == np.sign(flo)
        lo = np.where(mesmo, mid, lo)
        flo = np.where(mesmo, fmid, flo)
        hi = np.where(mesmo, hi, mid)
    saida = 0.5 * (lo + hi)
    saida[~ok] = np.nan
    # amostras sem mudança de sinal nos extremos podem ter 0 ou 2+ raízes
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i in np.where(~ok)[0]:
            saida[i] = _tir_1d(v[i], False, limites)
    return saida


def tir(fluxo, todas: bool = False, limites=LIMITES_TIR):
    """Taxa interna de retorno.

    Para fluxos 1-D varre o intervalo ``limites`` em busca de mudanças de
    sinal do VPL e refina cada raiz (Brent). Com ``todas=True`` devolve todas
    as raízes reais (fluxos não convencionais podem ter mais de uma TIR). Para
    fluxos amostrados usa bissecção vetorizada.
    """
    v = _v(fluxo)
    if v.ndim == 1:
        return _tir_1d(v, todas, limites)
    if todas:
        raise ValueError("todas=True só é suportado para fluxo 1-D")
    return _tir_2d(v, limites)


def tirm(fluxo, taxa_reinvestimento, taxa_financiamento=None):
    """TIR modificada.

    Entradas capitalizadas até ``n`` à ``taxa_reinvestimento``; saídas
    descontadas a ``taxa_financiamento`` (padrão: a mesma taxa). É a resposta
    à crítica "Cuidado com a TIR": a TIR tradicional supõe reinvestimento à
    própria TIR, a TIRM usa uma taxa realista (normalmente o custo de capital).
    """
    v = _v(fluxo)
    era_1d = v.ndim == 1
    v2 = np.atleast_2d(v)
    n = v2.shape[1] - 1
    t = np.arange(n + 1)
    tf = taxa_reinvestimento if taxa_financiamento is None else taxa_financiamento
    tr = np.asarray(taxa_reinvestimento, dtype=float).reshape(-1, 1) if np.ndim(taxa_reinvestimento) else taxa_reinvestimento
    tf = np.asarray(tf, dtype=float).reshape(-1, 1) if np.ndim(tf) else tf
    pos = np.where(v2 > 0, v2, 0.0)
    neg = np.where(v2 < 0, v2, 0.0)
    vf = np.sum(pos * (1 + tr) ** (n - t), axis=1)
    vpn = -np.sum(neg * (1 + tf) ** (-t), axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(vpn > 0, (vf / vpn) ** (1.0 / n) - 1.0, np.nan)
    return _saida(r, era_1d)


# --------------------------------------------------------------------------- #
# Payback, IBC, RAU, ROIA
# --------------------------------------------------------------------------- #
def payback(fluxo, taxa=None):
    """Tempo de recuperação do capital, com interpolação linear no período.

    ``taxa=None`` → payback simples; com taxa → payback ajustado (descontado).
    Devolve ``nan`` se o capital não é recuperado dentro do horizonte.
    """
    v = _v(fluxo)
    era_1d = v.ndim == 1
    v2 = np.atleast_2d(v).astype(float)
    if taxa is not None:
        t = np.arange(v2.shape[1])
        tx = np.asarray(taxa, dtype=float)
        fat = (1 + tx.reshape(-1, 1)) ** (-t) if tx.ndim else (1 + tx) ** (-t)
        v2 = v2 * fat
    cum = np.cumsum(v2, axis=1)
    cruz = (cum[:, :-1] < 0) & (cum[:, 1:] >= 0)
    tem = cruz.any(axis=1)
    idx = np.argmax(cruz, axis=1) + 1  # período em que recupera
    cum_ant = np.take_along_axis(cum, (idx - 1)[:, None], axis=1)[:, 0]
    fluxo_t = np.take_along_axis(v2, idx[:, None], axis=1)[:, 0]
    with np.errstate(divide="ignore", invalid="ignore"):
        frac = np.where(fluxo_t != 0, -cum_ant / fluxo_t, 0.0)
    pb = (idx - 1) + frac
    pb = np.where(tem, pb, np.nan)
    pb = np.where(cum[:, 0] >= 0, 0.0, pb)
    return _saida(pb, era_1d)


def ibc(fluxo, taxa):
    """Índice benefício/custo: VA das entradas ÷ VA das saídas (aceita se > 1)."""
    v = _v(fluxo)
    era_1d = v.ndim == 1
    v2 = np.atleast_2d(v)
    pos = np.where(v2 > 0, v2, 0.0)
    neg = np.where(v2 < 0, -v2, 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = _horner(pos, taxa) / _horner(neg, taxa)
    return _saida(r, era_1d)


def rau(fluxo, taxa, n: int | None = None):
    """Resultado Anual Uniforme: VPL distribuído em série uniforme na vida do projeto.

    RAU = VPL × (A/P, i, n). É o indicador adequado para comparar projetos de
    vidas diferentes sob a hipótese de repetição (equivale ao VPL do
    empilhamento até o MMC das vidas, anualizado).
    """
    v = _v(fluxo)
    n = v.shape[-1] - 1 if n is None else n
    return vpl(v, taxa) * fator_recuperacao_capital(taxa, n)


def rau_componentes(fluxo, taxa) -> dict:
    """Decompõe o RAU: equivalente anual do fluxo (−) equivalente anual do investimento."""
    v = _v(fluxo)
    n = v.shape[-1] - 1
    a_p = fator_recuperacao_capital(taxa, n)
    eau_fluxo = vp(v, taxa, 1) * a_p
    eau_inv = (-np.atleast_2d(v)[:, 0] if v.ndim == 2 else -v[0]) * a_p
    return {"eau_fluxo": eau_fluxo, "eau_investimento": eau_inv, "rau": eau_fluxo - eau_inv}


def roia(fluxo, taxa):
    """Retorno adicional sobre o investimento: (IBC)^(1/n) − 1, por período."""
    v = _v(fluxo)
    n = v.shape[-1] - 1
    return ibc(v, taxa) ** (1.0 / n) - 1.0


def taxa_fisher(fluxo_a, fluxo_b, todas: bool = False):
    """Taxa de Fisher: TIR do fluxo marginal (B − A); VPL_A = VPL_B nessa taxa."""
    a, b = alinhar(_v(fluxo_a), _v(fluxo_b))
    return tir(b - a, todas=todas)


# --------------------------------------------------------------------------- #
# Resumo
# --------------------------------------------------------------------------- #
COLUNAS_INDICADORES = {
    "investimento": "Investimento",
    "vida": "Vida (anos)",
    "vpl": "VPL",
    "tir": "TIR",
    "tirm": "TIRM",
    "rau": "RAU",
    "ibc": "IBC",
    "roia": "ROIA",
    "payback": "Payback simples",
    "payback_descontado": "Payback ajustado",
}


def indicadores(fluxo, taxa, taxa_reinvestimento=None) -> dict:
    """Dicionário com todos os indicadores de um fluxo."""
    v = _v(fluxo)
    tr = taxa if taxa_reinvestimento is None else taxa_reinvestimento
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return {
            "investimento": _saida(np.atleast_2d(v)[:, 0] * -1, v.ndim == 1),
            "vida": v.shape[-1] - 1,
            "vpl": vpl(v, taxa),
            "tir": tir(v),
            "tirm": tirm(v, tr),
            "rau": rau(v, taxa),
            "ibc": ibc(v, taxa),
            "roia": roia(v, taxa),
            "payback": payback(v),
            "payback_descontado": payback(v, taxa),
        }


def tabela_indicadores(fluxos, taxa, taxa_reinvestimento=None, rotulos: bool = False) -> pd.DataFrame:
    """Tabela de indicadores (linhas = fluxos/projetos)."""
    linhas = {}
    for f in como_fluxos(fluxos):
        ind = indicadores(f.media() if f.vetorizado else f, taxa, taxa_reinvestimento)
        linhas[f.nome] = ind
    df = pd.DataFrame(linhas).T
    df.index.name = "Projeto"
    df = df.astype({c: float for c in df.columns if c != "vida"})
    df["vida"] = df["vida"].astype(int)
    if rotulos:
        df = df.rename(columns=COLUNAS_INDICADORES)
    return df
