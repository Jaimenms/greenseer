"""Formatação de tabelas para relatórios e apresentações (pandas Styler)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from .cashflow import como_fluxos
from .metricas import COLUNAS_INDICADORES


def fmt_moeda(x, simbolo: str = "$", casas: int = 0) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    s = f"{abs(x):,.{casas}f}".replace(",", "X").replace(".", ",").replace("X", ".")
    return f"({simbolo} {s})" if x < 0 else f"{simbolo} {s}"


def fmt_pct(x, casas: int = 2) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{100 * x:.{casas}f}%".replace(".", ",")


def fmt_num(x, casas: int = 2) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{x:,.{casas}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def fmt_anos(x, casas: int = 2) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "não recupera"
    return f"{x:.{casas}f} anos".replace(".", ",")


_FORMATADORES = {
    "moeda": fmt_moeda,
    "pct": fmt_pct,
    "num": fmt_num,
    "anos": fmt_anos,
    "int": lambda x: f"{int(x)}",
}

TIPOS_PADRAO = {
    "investimento": "moeda", "vpl": "moeda", "rau": "moeda", "VPL": "moeda", "RAU": "moeda", "Investimento": "moeda",
    "VPL marginal": "moeda", "RAU marginal": "moeda", "Investimento marginal": "moeda", "RAU (soma)": "moeda",
    "Valor esperado": "moeda", "Desvio padrão": "moeda", "Média": "moeda", "Mediana": "moeda", "Mínimo": "moeda", "Máximo": "moeda",
    "P5": "moeda", "P95": "moeda", "CVaR 5%": "moeda", "P10": "moeda", "P90": "moeda", "Base": "moeda",
    "tir": "pct", "tirm": "pct", "roia": "pct", "TIR": "pct", "TIRM": "pct", "TIR marginal": "pct", "TIRM marginal": "pct",
    "P(< 0)": "pct", "P(melhor)": "pct", "Probabilidade": "pct", "Taxa": "pct", "Taxa ajustada": "pct", "ROIA": "pct",
    "ibc": "num", "IBC": "num", "CV": "num",
    "payback": "anos", "payback_descontado": "anos", "Payback simples": "anos", "Payback ajustado": "anos",
    "vida": "int", "Vida (anos)": "int", "Horizonte": "int", "Nº projetos": "int", "Nº de TIRs": "int", "n válidas": "int",
}

MAIOR_MELHOR = {"vpl", "tir", "tirm", "rau", "ibc", "roia", "VPL", "TIR", "TIRM", "RAU", "IBC", "ROIA", "Valor esperado", "Média", "RAU (soma)", "P(melhor)", "Mediana", "P5", "P95", "VPL marginal", "RAU marginal"}
MENOR_MELHOR = {"payback", "payback_descontado", "Payback simples", "Payback ajustado", "CV", "Desvio padrão", "P(< 0)"}


def estilizar(
    df: pd.DataFrame,
    tipos: Mapping[str, str] | None = None,
    destacar: Sequence[str] | None = None,
    titulo: str | None = None,
    rotulos: bool = True,
    precisao_moeda: int = 0,
):
    """Aplica formatação pt-BR e destaque (negrito) do melhor valor em cada coluna.

    ``tipos``: ``{coluna: 'moeda'|'pct'|'num'|'anos'|'int'}`` (complementa o padrão).
    ``destacar``: colunas em que o melhor valor é destacado (padrão: todas as conhecidas).
    """
    d = df.copy()
    tipos = {**TIPOS_PADRAO, **(tipos or {})}
    formatos = {}
    for c in d.columns:
        t = tipos.get(c)
        if t == "moeda":
            formatos[c] = lambda x, k=precisao_moeda: fmt_moeda(x, casas=k)
        elif t in _FORMATADORES:
            formatos[c] = _FORMATADORES[t]
        elif pd.api.types.is_float_dtype(d[c]):
            formatos[c] = fmt_num
    alvo = list(destacar) if destacar is not None else [c for c in d.columns if c in MAIOR_MELHOR or c in MENOR_MELHOR]
    if rotulos:
        d = d.rename(columns=COLUNAS_INDICADORES)
        formatos = {COLUNAS_INDICADORES.get(k, k): v for k, v in formatos.items()}
        alvo = [COLUNAS_INDICADORES.get(c, c) for c in alvo]
    sty = d.style.format(formatos, na_rep="—")
    for c in alvo:
        if c not in d.columns or not pd.api.types.is_numeric_dtype(d[c]):
            continue
        col = d[c].astype(float)
        if col.isna().all():
            continue
        chave = col.idxmax() if (c in MAIOR_MELHOR or COLUNAS_INDICADORES.get(c, c) in MAIOR_MELHOR) else col.idxmin()
        sty = sty.map(lambda v, k=chave: "font-weight: 700; background-color: #e8f1fc", subset=pd.IndexSlice[[chave], [c]])
    sty = sty.set_table_styles(
        [
            {"selector": "th", "props": "background-color: #f3f2ee; color: #0b0b0b; font-weight: 600; text-align: right; padding: 6px 10px; border-bottom: 1px solid #c3c2b7;"},
            {"selector": "td", "props": "text-align: right; padding: 5px 10px; border-bottom: 1px solid #e1e0d9; font-variant-numeric: tabular-nums;"},
            {"selector": "th.row_heading", "props": "text-align: left;"},
            {"selector": "caption", "props": "caption-side: top; text-align: left; font-weight: 600; font-size: 1.05em; color: #0b0b0b; padding-bottom: 6px;"},
        ]
    )
    if titulo:
        sty = sty.set_caption(titulo)
    return sty


def tabela_fluxos(fluxos, transpor: bool = False) -> pd.DataFrame:
    """DataFrame anos × projetos com os fluxos (zeros além da vida de cada um)."""
    fs = [f.media() for f in como_fluxos(fluxos)]
    H = max(f.n for f in fs)
    df = pd.DataFrame({f.nome: f.estender(H).valores for f in fs}, index=pd.Index(range(H + 1), name="ano"))
    return df.T if transpor else df


def estilizar_demonstrativo(df: pd.DataFrame, titulo: str | None = None):
    """Formata a tabela DRE → fluxo de caixa: moeda, totais em negrito, saídas entre parênteses."""
    sty = df.style.format(lambda x: fmt_moeda(x) if x != 0 else "–")
    totais = [i for i in df.index if str(i).startswith("(=)")]
    memo = [i for i in df.index if str(i).startswith("Memo")]
    sty = sty.set_properties(subset=pd.IndexSlice[totais, :], **{"font-weight": "700", "border-top": "1px solid #52514e"})
    sty = sty.set_properties(subset=pd.IndexSlice[memo, :], **{"color": "#898781", "font-style": "italic"})
    sty = sty.set_table_styles(
        [
            {"selector": "th", "props": "background-color: #f3f2ee; color: #0b0b0b; font-weight: 600; text-align: right; padding: 5px 10px;"},
            {"selector": "td", "props": "text-align: right; padding: 4px 10px; border-bottom: 1px solid #efeeea; font-variant-numeric: tabular-nums; white-space: pre;"},
            {"selector": "th.row_heading", "props": "text-align: left; white-space: pre;"},
            {"selector": "caption", "props": "caption-side: top; text-align: left; font-weight: 600; font-size: 1.05em; padding-bottom: 6px;"},
        ]
    )
    if titulo:
        sty = sty.set_caption(titulo)
    return sty


# --------------------------------------------------------------------------- #
# Gradientes de cor sem matplotlib
# --------------------------------------------------------------------------- #
def _hex_para_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _interpolar(c1: str, c2: str, t: float) -> str:
    a, b = _hex_para_rgb(c1), _hex_para_rgb(c2)
    t = min(max(float(t), 0.0), 1.0)
    return "#%02x%02x%02x" % tuple(int(round(a[i] + (b[i] - a[i]) * t)) for i in range(3))


def _texto_para(fundo: str) -> str:
    r, g, b = _hex_para_rgb(fundo)
    return "#0b0b0b" if (0.299 * r + 0.587 * g + 0.114 * b) > 140 else "#f0efec"


def estilizar_gradiente(df: pd.DataFrame, vmin: float | None = None, vmax: float | None = None, inverter: bool = False, formato: str = "{:,.2f}", titulo: str | None = None, cores: tuple[str, str] = ("#cde2fb", "#184f95")):
    """Fundo em escala sequencial (azul claro → escuro) proporcional ao valor; sem dependência de matplotlib."""
    vals = df.astype(float)
    lo = float(vals.min().min()) if vmin is None else vmin
    hi = float(vals.max().max()) if vmax is None else vmax

    def cor(v):
        if pd.isna(v):
            return ""
        t = 0.5 if hi == lo else (float(v) - lo) / (hi - lo)
        t = 1 - t if inverter else t
        fundo = _interpolar(cores[0], cores[1], t)
        return f"background-color: {fundo}; color: {_texto_para(fundo)}"

    sty = df.style.format(formato).map(cor)
    sty = sty.set_table_styles([{"selector": "th", "props": "background-color: #f3f2ee; font-weight: 600; text-align: right; padding: 5px 10px;"}, {"selector": "td", "props": "text-align: right; padding: 4px 10px;"}, {"selector": "caption", "props": "caption-side: top; text-align: left; font-weight: 600; padding-bottom: 6px;"}])
    if titulo:
        sty = sty.set_caption(titulo)
    return sty


def estilizar_divergente(df: pd.DataFrame, amplitude: float | None = None, formato: str = "{:+.2f}", titulo: str | None = None):
    """Fundo divergente (vermelho ← cinza → azul) em torno de zero, para correlações ou diferenças."""
    vals = df.astype(float)
    amp = float(vals.abs().max().max()) if amplitude is None else amplitude
    amp = amp or 1.0

    def cor(v):
        if pd.isna(v):
            return ""
        t = min(max(float(v) / amp, -1.0), 1.0)
        fundo = _interpolar("#f0efec", "#2a78d6", t) if t >= 0 else _interpolar("#f0efec", "#e34948", -t)
        return f"background-color: {fundo}; color: {_texto_para(fundo)}"

    sty = df.style.format(formato).map(cor)
    sty = sty.set_table_styles([{"selector": "th", "props": "background-color: #f3f2ee; font-weight: 600; text-align: right; padding: 5px 10px;"}, {"selector": "td", "props": "text-align: right; padding: 4px 10px;"}, {"selector": "caption", "props": "caption-side: top; text-align: left; font-weight: 600; padding-bottom: 6px;"}])
    if titulo:
        sty = sty.set_caption(titulo)
    return sty
