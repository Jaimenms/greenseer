"""Análise de cenários, sensibilidade e taxa ajustada ao risco.

Segue as notas de Análise de Risco: cenários *pessimista / mais provável /
otimista* com probabilidades, valor esperado do VPL, desvio padrão,
coeficiente de variação (CV = DP / E) e taxa de desconto ajustada ao risco em
função do CV.

Um *modelo* é uma função ``modelo(parametros: dict) -> Projeto | FluxoDeCaixa``.
Cenários e simulações apenas alteram o dicionário de parâmetros e reavaliam o
modelo — a mesma função serve para o caso-base, para cenários discretos e para
Monte Carlo (com parâmetros vetorizados).
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from .cashflow import FluxoDeCaixa, como_fluxo
from .metricas import COLUNAS_INDICADORES, indicadores

Modelo = Callable[[Mapping[str, Any]], Any]
METRICAS = ["investimento", "vpl", "tir", "tirm", "rau", "ibc", "roia", "payback", "payback_descontado"]


@dataclass
class Cenario:
    """Conjunto nomeado de parâmetros que sobrescreve o caso-base."""

    nome: str
    parametros: dict = field(default_factory=dict)
    probabilidade: float | None = None
    descricao: str = ""


def _normalizar_modelos(modelos) -> dict[str, Modelo]:
    if isinstance(modelos, Mapping):
        return dict(modelos)
    return {"Projeto": modelos}


def avaliar(modelo: Modelo, parametros: Mapping, taxa, taxa_reinvestimento=None) -> dict:
    """Avalia um modelo com um dicionário de parâmetros e devolve os indicadores."""
    fluxo = como_fluxo(modelo(parametros))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return indicadores(fluxo, taxa, taxa_reinvestimento)


# --------------------------------------------------------------------------- #
# Estatísticas com probabilidades
# --------------------------------------------------------------------------- #
def valor_esperado(valores, probabilidades=None) -> float:
    v = np.asarray(valores, dtype=float)
    p = np.full(len(v), 1.0 / len(v)) if probabilidades is None else np.asarray(probabilidades, dtype=float)
    return float(np.sum(v * p) / np.sum(p))


def desvio_padrao(valores, probabilidades=None) -> float:
    v = np.asarray(valores, dtype=float)
    p = np.full(len(v), 1.0 / len(v)) if probabilidades is None else np.asarray(probabilidades, dtype=float)
    p = p / p.sum()
    e = np.sum(v * p)
    return float(np.sqrt(np.sum(p * (v - e) ** 2)))


def coeficiente_de_variacao(valores, probabilidades=None) -> float:
    e = valor_esperado(valores, probabilidades)
    return float("nan") if e == 0 else desvio_padrao(valores, probabilidades) / abs(e)


def taxa_ajustada_ao_risco(cv, curva: Sequence[tuple[float, float]]):
    """Taxa de desconto ajustada ao risco por interpolação linear numa curva CV → taxa.

    ``curva`` = lista de pares ``(cv, taxa)`` como no gráfico das notas (ex.:
    ``[(0.0, 0.06), (0.3, 0.07), (0.8, 0.10), (2.0, 0.22)]``). Fora do
    intervalo a taxa é mantida constante.
    """
    pontos = sorted(curva)
    xs = [p[0] for p in pontos]
    ys = [p[1] for p in pontos]
    r = np.interp(np.asarray(cv, dtype=float), xs, ys)
    return float(r) if r.ndim == 0 else r


# --------------------------------------------------------------------------- #
# Cenários
# --------------------------------------------------------------------------- #
class AnaliseDeCenarios:
    """Avalia um ou vários modelos sob um conjunto de cenários.

    Parameters
    ----------
    modelos
        ``{nome: modelo}`` ou um único modelo.
    base
        Parâmetros do caso-base (compartilhados por todos os modelos).
    cenarios
        Lista de :class:`Cenario`; cada um sobrescreve parte dos parâmetros.
    taxa
        Taxa mínima de atratividade usada em VPL/RAU/IBC.
    """

    def __init__(self, modelos, base: Mapping, cenarios: Sequence[Cenario], taxa: float, taxa_reinvestimento: float | None = None):
        self.modelos = _normalizar_modelos(modelos)
        self.base = dict(base)
        self.cenarios = list(cenarios)
        self.taxa = taxa
        self.taxa_reinvestimento = taxa_reinvestimento
        self._resultado: pd.DataFrame | None = None
        self._fluxos: dict[tuple[str, str], FluxoDeCaixa] = {}

    @property
    def probabilidades(self) -> np.ndarray | None:
        p = [c.probabilidade for c in self.cenarios]
        if any(x is None for x in p):
            return None
        p = np.asarray(p, dtype=float)
        if not np.isclose(p.sum(), 1.0):
            raise ValueError(f"probabilidades dos cenários somam {p.sum():.3f} (esperado 1)")
        return p

    def parametros(self, cenario: Cenario) -> dict:
        return {**self.base, **cenario.parametros}

    def executar(self) -> pd.DataFrame:
        """Tabela longa: uma linha por (projeto, cenário) com todos os indicadores."""
        linhas = []
        for nome, modelo in self.modelos.items():
            for c in self.cenarios:
                fluxo = como_fluxo(modelo(self.parametros(c)))
                self._fluxos[(nome, c.nome)] = fluxo
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    ind = indicadores(fluxo, self.taxa, self.taxa_reinvestimento)
                linhas.append({"Projeto": nome, "Cenário": c.nome, "Probabilidade": c.probabilidade, **{k: ind[k] for k in METRICAS}})
        self._resultado = pd.DataFrame(linhas)
        return self._resultado

    @property
    def resultado(self) -> pd.DataFrame:
        if self._resultado is None:
            self.executar()
        return self._resultado

    def fluxos(self) -> dict[tuple[str, str], FluxoDeCaixa]:
        _ = self.resultado
        return dict(self._fluxos)

    def tabela(self, metrica: str = "vpl") -> pd.DataFrame:
        """Pivô projeto × cenário para uma métrica."""
        df = self.resultado.pivot(index="Projeto", columns="Cenário", values=metrica)
        ordem = [c.nome for c in self.cenarios]
        return df[ordem].reindex(list(self.modelos))

    def resumo(self, metrica: str = "vpl") -> pd.DataFrame:
        """Valor esperado, desvio padrão, CV, mínimo, máximo e P(métrica < 0) por projeto."""
        p = self.probabilidades
        df = self.tabela(metrica)
        linhas = {}
        for proj, linha in df.iterrows():
            v = linha.values.astype(float)
            e = valor_esperado(v, p)
            dp = desvio_padrao(v, p)
            pesos = np.full(len(v), 1.0 / len(v)) if p is None else p
            linhas[proj] = {
                "Valor esperado": e,
                "Desvio padrão": dp,
                "CV": dp / abs(e) if e else np.nan,
                "Mínimo": float(np.nanmin(v)),
                "Máximo": float(np.nanmax(v)),
                "P(< 0)": float(np.sum(pesos[v < 0])),
            }
        out = pd.DataFrame(linhas).T
        out.index.name = "Projeto"
        return out

    def taxas_ajustadas(self, curva: Sequence[tuple[float, float]], metrica: str = "vpl") -> pd.Series:
        """Taxa ajustada ao risco de cada projeto a partir do CV da métrica."""
        cv = self.resumo(metrica)["CV"]
        return pd.Series(taxa_ajustada_ao_risco(cv.values, curva), index=cv.index, name="Taxa ajustada")

    def reavaliar(self, taxas: Mapping[str, float], cenario: str | None = None) -> pd.DataFrame:
        """Recalcula VPL/RAU/IBC com uma taxa específica por projeto (ajustada ao risco)."""
        nome_c = cenario or self.cenarios[0].nome
        linhas = {}
        for proj in self.modelos:
            cen = next(c for c in self.cenarios if c.nome == nome_c)
            f = self._fluxos.get((proj, nome_c)) or como_fluxo(self.modelos[proj](self.parametros(cen)))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ind = indicadores(f, float(taxas[proj]), self.taxa_reinvestimento)
            linhas[proj] = {"Taxa": float(taxas[proj]), "VPL": ind["vpl"], "RAU": ind["rau"], "IBC": ind["ibc"], "TIR": ind["tir"]}
        out = pd.DataFrame(linhas).T
        out.index.name = "Projeto"
        return out


# --------------------------------------------------------------------------- #
# Sensibilidade (tornado) e ponto de equilíbrio
# --------------------------------------------------------------------------- #
def sensibilidade(
    modelo: Modelo,
    base: Mapping,
    variacoes: Mapping[str, Any],
    taxa: float,
    metrica: str = "vpl",
    taxa_reinvestimento: float | None = None,
    rotulos: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    """Análise de sensibilidade um-parâmetro-por-vez (gráfico tornado).

    ``variacoes``: ``{parametro: (valor_baixo, valor_alto)}`` ou
    ``{parametro: fracao}`` (→ base × (1 ∓ fracao)). Devolve tabela ordenada
    pela amplitude do impacto na métrica.
    """
    rotulos = rotulos or {}
    base = dict(base)
    ref = avaliar(modelo, base, taxa, taxa_reinvestimento)[metrica]
    linhas = []
    for par, var in variacoes.items():
        if isinstance(var, (int, float)):
            b0 = float(base[par])
            baixo, alto = b0 * (1 - var), b0 * (1 + var)
        else:
            baixo, alto = var
        m_baixo = avaliar(modelo, {**base, par: baixo}, taxa, taxa_reinvestimento)[metrica]
        m_alto = avaliar(modelo, {**base, par: alto}, taxa, taxa_reinvestimento)[metrica]
        linhas.append(
            {
                "Parâmetro": rotulos.get(par, par),
                "chave": par,
                "Valor baixo": baixo,
                "Valor alto": alto,
                "Métrica (baixo)": m_baixo,
                "Métrica (alto)": m_alto,
                "Base": ref,
                "Amplitude": abs(m_alto - m_baixo),
            }
        )
    df = pd.DataFrame(linhas).sort_values("Amplitude", ascending=False).reset_index(drop=True)
    df.attrs["metrica"] = metrica
    return df


def ponto_de_equilibrio(
    modelo: Modelo,
    base: Mapping,
    parametro: str,
    taxa: float,
    metrica: str = "vpl",
    alvo: float = 0.0,
    intervalo: tuple[float, float] | None = None,
) -> float:
    """Valor do parâmetro que leva a métrica ao ``alvo`` (ex.: VPL = 0)."""
    base = dict(base)

    def f(x):
        return avaliar(modelo, {**base, parametro: x}, taxa)[metrica] - alvo

    if intervalo is None:
        b0 = float(base[parametro])
        if b0 == 0:
            raise ValueError("informe 'intervalo' quando o valor-base do parâmetro é zero")
        fatores = [0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0, 1.1, 1.25, 1.5, 2.0, 3.0, 5.0, 10.0]
        pontos = [b0 * k for k in fatores]
        vals = [f(x) for x in pontos]
        for i in range(len(pontos) - 1):
            if np.sign(vals[i]) != np.sign(vals[i + 1]):
                return float(brentq(f, pontos[i], pontos[i + 1], xtol=1e-10))
        raise ValueError("não houve mudança de sinal da métrica no intervalo explorado; informe 'intervalo'")
    return float(brentq(f, intervalo[0], intervalo[1], xtol=1e-10))
