"""Gráficos para apresentação ao board (Plotly).

Convenções visuais: marcas finas, grade em linha fina, cada projeto com uma
cor fixa (a mesma em todos os gráficos), entradas em azul e saídas em
vermelho, rótulos diretos apenas onde importam. Todas as funções devolvem um
``plotly.graph_objects.Figure`` pronto para ``fig.show()`` ou
:func:`exportar`.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

from .cashflow import FluxoDeCaixa, como_fluxos, mmc
from .comparacao import equalizar
from .metricas import payback as _payback
from .metricas import taxa_fisher, tir, vpl, vpl_perfil

# --------------------------------------------------------------------------- #
# Paleta e tema
# --------------------------------------------------------------------------- #
PALETA = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
COR_ENTRADA = "#2a78d6"
COR_SAIDA = "#e34948"
COR_NEUTRA = "#c3c2b7"
COR_DESTAQUE = "#184f95"
COR_BOM = "#0ca30c"
COR_RUIM = "#d03b3b"
INK = "#0b0b0b"
INK_2 = "#52514e"
INK_MUDA = "#898781"
GRADE = "#e1e0d9"
EIXO = "#c3c2b7"
SUPERFICIE = "#fcfcfb"
FONTE = 'system-ui, -apple-system, "Segoe UI", Helvetica, Arial, sans-serif'

_CORES: dict[str, str] = {}


def cor_de(nome: str) -> str:
    """Cor fixa de uma entidade (projeto/carteira) — a mesma em todos os gráficos."""
    if nome not in _CORES:
        _CORES[nome] = PALETA[len(_CORES) % len(PALETA)]
    return _CORES[nome]


def definir_cores(mapa: Mapping[str, str]) -> None:
    """Fixa explicitamente as cores (ex.: cores da marca da empresa)."""
    _CORES.update(mapa)


def tema() -> go.layout.Template:
    t = go.layout.Template()
    t.layout = go.Layout(
        font=dict(family=FONTE, size=13, color=INK),
        paper_bgcolor=SUPERFICIE,
        plot_bgcolor=SUPERFICIE,
        colorway=PALETA,
        title=dict(x=0.0, xanchor="left", font=dict(size=18, color=INK), pad=dict(b=8)),
        xaxis=dict(showgrid=False, zeroline=False, linecolor=EIXO, ticks="outside", tickcolor=EIXO, tickfont=dict(color=INK_2), title_font=dict(color=INK_2)),
        yaxis=dict(showgrid=True, gridcolor=GRADE, gridwidth=1, zeroline=True, zerolinecolor=EIXO, zerolinewidth=1, showline=False, tickfont=dict(color=INK_2), title_font=dict(color=INK_2)),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0, bgcolor="rgba(0,0,0,0)", font=dict(color=INK_2)),
        margin=dict(l=70, r=30, t=80, b=60),
        hoverlabel=dict(bgcolor="white", bordercolor=GRADE, font=dict(family=FONTE, size=12, color=INK)),
        separators=",.",
        bargap=0.4,
        bargroupgap=0.08,
    )
    return t


pio.templates["greenseer"] = tema()
pio.templates.default = "greenseer"


def configurar_notebook(estatico: bool | None = None) -> str:
    """Configura o renderizador Plotly para notebooks.

    Saída interativa (mimetype + HTML via CDN) e, se ``estatico`` (padrão:
    detecta kaleido/Chrome), também uma imagem PNG — assim o notebook mostra
    os gráficos mesmo onde não há JavaScript (GitHub, PDF).
    """
    base = "plotly_mimetype+notebook_connected"
    if estatico is None:
        try:
            go.Figure().to_image(format="png", width=10, height=10)
            estatico = True
        except Exception:
            estatico = False
    pio.renderers.default = base + "+png" if estatico else base
    pio.renderers["png"].width = 1000
    pio.renderers["png"].height = 450
    pio.renderers["png"].scale = 2
    return pio.renderers.default


def _moeda(x: float, simbolo: str = "$") -> str:
    s = f"{abs(x):,.0f}".replace(",", "X").replace(".", ",").replace("X", ".")
    return f"−{simbolo} {s}" if x < 0 else f"{simbolo} {s}"


def _pct(x: float, casas: int = 1) -> str:
    return f"{100 * x:.{casas}f}%".replace(".", ",")


# --------------------------------------------------------------------------- #
# Fluxos de caixa
# --------------------------------------------------------------------------- #
COR_COMPONENTES = {
    "investimento": COR_SAIDA,       # vermelho
    "capital_de_giro": "#e87ba4",    # magenta
    "valor_residual": "#6250d6",     # violeta
    "financiamento": "#008300",      # verde
    "outros": "#898781",             # cinza
}


def componentes_projeto(projeto, tipo: str = "global") -> list[dict]:
    """Decompõe o fluxo de um :class:`~greenseer.projeto.Projeto` em componentes por ano.

    Devolve uma lista ordenada de ``{"nome", "valores" (0..n), "cor"}``; os
    componentes positivos são empilhados para cima e os negativos para baixo,
    na ordem da lista. O valor residual já vem líquido do IR sobre o ganho de
    capital (como calculado pelo projeto).
    """
    demo = projeto.calcular() if hasattr(projeto, "calcular") else projeto
    L = {k: (np.asarray(v).mean(axis=0) if np.ndim(v) == 2 else np.asarray(v, dtype=float)) for k, v in demo.linhas.items()}
    nome = demo.nome
    comps = []
    if tipo == "proprio":
        if "fccp" not in L:
            raise ValueError("projeto sem financiamento: não há fluxo do capital próprio")
        operacional = L["lucro_liquido"] + L["depreciacao"]
        comps.append({"nome": "Operacional (lucro líquido + depreciação)", "valores": operacional, "cor": cor_de(nome)})
        comps.append({"nome": "Entrada do financiamento", "valores": L["entrada_financiamento"], "cor": COR_COMPONENTES["financiamento"]})
        comps.append({"nome": "Amortização do financiamento", "valores": -L["amortizacao"], "cor": COR_COMPONENTES["financiamento"]})
    else:
        comps.append({"nome": "Operacional (NOPAT + depreciação)", "valores": L["fco"], "cor": cor_de(nome)})
    comps.append({"nome": "Investimento", "valores": L["investimento"], "cor": COR_COMPONENTES["investimento"]})
    comps.append({"nome": "Capital de giro", "valores": L["capital_de_giro"], "cor": COR_COMPONENTES["capital_de_giro"]})
    comps.append({"nome": "Valor residual (líquido de IR)", "valores": L["valor_residual"], "cor": COR_COMPONENTES["valor_residual"]})
    comps.append({"nome": "Outros fluxos", "valores": L["outros"], "cor": COR_COMPONENTES["outros"]})
    return [c for c in comps if np.any(c["valores"] != 0)]


def _desenhar_setas(fig, t, componentes, row: int | None = None, col: int | None = None, legenda: bool = True, rotulo_projeto: str = "") -> tuple[float, float]:
    """Desenha setas empilhadas (positivas para cima, negativas para baixo) e devolve (y_min, y_max)."""
    kw = {} if row is None else {"row": row, "col": col}
    base_cima = np.zeros(len(t))
    base_baixo = np.zeros(len(t))
    y_min, y_max = 0.0, 0.0
    for comp in componentes:
        v = np.asarray(comp["valores"], dtype=float)
        pos = v > 0
        neg = v < 0
        for mask, base in ((pos, base_cima), (neg, base_baixo)):
            if not mask.any():
                continue
            xs, ys = [], []
            for tt in t[mask]:
                xs += [tt, tt, None]
                ys += [base[tt], base[tt] + v[tt], None]
            fig.add_scatter(x=xs, y=ys, mode="lines", line=dict(color=comp["cor"], width=2.5), hoverinfo="skip", showlegend=False, **kw)
            tips = base[mask] + v[mask]
            fig.add_scatter(
                x=t[mask], y=tips, mode="markers", name=comp["nome"], showlegend=False,
                marker=dict(symbol="triangle-up" if base is base_cima else "triangle-down", size=11, color=comp["cor"], line=dict(color=SUPERFICIE, width=1)),
                customdata=v[mask], hovertemplate=f"{rotulo_projeto}ano %{{x}}<br>{comp['nome']}: %{{customdata:,.0f}}<extra></extra>", **kw,
            )
            base[mask] += v[mask]
        y_max = max(y_max, float(base_cima.max()))
        y_min = min(y_min, float(base_baixo.min()))
        if legenda:
            fig.add_scatter(x=[None], y=[None], mode="markers", name=comp["nome"], marker=dict(symbol="square", size=10, color=comp["cor"]), showlegend=True, hoverinfo="skip", **kw)
    return y_min, y_max


def grafico_fluxo(fluxo, titulo: str | None = None, taxa: float | None = None, acumulado: bool = False, altura: int = 420, estilo: str = "setas", tipo: str = "global") -> go.Figure:
    """Diagrama de fluxo de caixa de um projeto ou fluxo.

    ``estilo="setas"`` (padrão): setas para cima nas entradas e para baixo nas
    saídas. Para um :class:`~greenseer.projeto.Projeto`, cada ano é decomposto
    em componentes empilhados com cores próprias — operacional, investimento,
    capital de giro, **valor residual líquido de IR** (sobre o ganho em relação
    ao valor contábil, no último ano) e, no fluxo do capital próprio
    (``tipo="proprio"``), entrada e amortização do financiamento.
    ``estilo="barras"``: barras do fluxo líquido. ``acumulado=True`` sobrepõe o
    acumulado (descontado a ``taxa``) e marca o payback.
    """
    eh_projeto = hasattr(fluxo, "calcular")
    f = (fluxo.fluxo(tipo) if eh_projeto else como_fluxos([fluxo])[0]).media()
    v = f.valores
    fig = go.Figure()
    if estilo == "setas":
        comps = componentes_projeto(fluxo, tipo) if eh_projeto else [
            {"nome": "Entradas", "valores": np.where(v > 0, v, 0.0), "cor": cor_de(f.nome)},
            {"nome": "Saídas / investimentos", "valores": np.where(v < 0, v, 0.0), "cor": COR_SAIDA},
        ]
        y_min, y_max = _desenhar_setas(fig, f.t, comps)
        folga = 0.12 * (y_max - y_min or 1.0)
        fig.update_yaxes(range=[y_min - folga, y_max + folga], zeroline=True, zerolinecolor=INK, zerolinewidth=1.5, showgrid=False)
        fig.update_xaxes(showline=False, ticks="", range=[-0.7, f.n + 0.7])
    else:
        fig.add_bar(
            x=f.t, y=v, name=f.nome or "Fluxo",
            marker=dict(color=[COR_ENTRADA if x >= 0 else COR_SAIDA for x in v], line=dict(width=0)),
            hovertemplate="Ano %{x}<br>%{y:,.0f}<extra></extra>", width=0.6, showlegend=False,
        )
    if acumulado:
        fat = (1 + taxa) ** (-f.t) if taxa is not None else np.ones_like(f.t, dtype=float)
        cum = np.cumsum(v * fat)
        rot = "Acumulado descontado" if taxa is not None else "Acumulado"
        fig.add_scatter(x=f.t, y=cum, mode="lines+markers", name=rot, line=dict(color=INK_2, width=2, dash="dot"), marker=dict(size=7, color=INK_2), hovertemplate="Ano %{x}<br>" + rot + ": %{y:,.0f}<extra></extra>")
        pb = _payback(f, taxa)
        if not np.isnan(pb):
            fig.add_vline(x=pb, line=dict(color=INK_MUDA, width=1, dash="dot"))
            fig.add_annotation(x=pb, y=1, yref="paper", text=f"payback {pb:.1f} anos".replace(".", ","), showarrow=False, yshift=2, font=dict(size=11, color=INK_2), xanchor="left", xshift=4)
        if estilo == "setas":
            lo = min(fig.layout.yaxis.range[0], float(cum.min()) * 1.1)
            hi = max(fig.layout.yaxis.range[1], float(cum.max()) * 1.1)
            fig.update_yaxes(range=[lo, hi])
    subtitulo = "↑ entradas · ↓ saídas · setas empilhadas = componentes do mesmo ano" if estilo == "setas" else None
    fig.update_layout(
        title=dict(text=titulo or f"Fluxo de caixa — {f.nome}", subtitle=dict(text=subtitulo, font=dict(size=11, color=INK_2)) if subtitulo else None),
        xaxis_title="Ano", yaxis_title="Fluxo ($)", height=altura + (60 if estilo == "setas" else 0), showlegend=(estilo == "setas") or acumulado, margin=dict(t=90, b=120 if estilo == "setas" else 60),
        legend=dict(orientation="h", y=-0.32, yanchor="top", x=0, xanchor="left") if estilo == "setas" else None,
    )
    fig.update_xaxes(tickmode="array", tickvals=list(f.t))
    return fig


def grafico_fluxos(fluxos, titulo: str = "Fluxos de caixa dos projetos", altura: int = 420) -> go.Figure:
    """Barras agrupadas por ano, uma cor por projeto."""
    fs = [f.media() for f in como_fluxos(fluxos)]
    fig = go.Figure()
    for f in fs:
        fig.add_bar(x=f.t, y=f.valores, name=f.nome, marker=dict(color=cor_de(f.nome), line=dict(width=0)), hovertemplate=f"{f.nome}<br>Ano %{{x}}: %{{y:,.0f}}<extra></extra>")
    fig.update_layout(title=titulo, barmode="group", xaxis_title="Ano", yaxis_title="$", height=altura)
    return fig


def componentes_empilhadas(fluxo, ciclos: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Entradas (≥ 0) e saídas (≤ 0) por período de um fluxo empilhado em ``ciclos`` repetições.

    Diferente de :meth:`FluxoDeCaixa.replicar`, não soma os valores no ano em
    que um ciclo termina e o seguinte começa: a última entrada do ciclo
    anterior e o investimento do novo ciclo ficam separados, para que o
    diagrama mostre uma seta para cima e outra para baixo nesse ano.
    """
    f = como_fluxos([fluxo])[0].media()
    v = f.valores
    n = f.n
    H = n * ciclos
    entradas = np.zeros(H + 1)
    saidas = np.zeros(H + 1)
    for k in range(ciclos):
        seg = slice(k * n, k * n + n + 1)
        entradas[seg] += np.where(v > 0, v, 0.0)
        saidas[seg] += np.where(v < 0, v, 0.0)
    return entradas, saidas


def grafico_diagrama_fluxo(fluxos, metodo: str | None = None, horizonte: int | None = None, titulo: str | None = None, altura: int | None = None, subtitulo: str | None = None) -> go.Figure:
    """Diagrama de fluxo de caixa com setas: para cima = entradas, para baixo = investimentos/saídas.

    Com ``metodo="mmc"`` cada projeto é empilhado até o mínimo múltiplo comum
    das vidas (ou ``horizonte``); nos anos de reinvestimento aparecem as duas
    setas (a última entrada do ciclo que termina e o investimento do ciclo
    que começa). Um painel por projeto, eixo do tempo compartilhado.
    """
    fs = como_fluxos(fluxos)
    vidas = [f.n for f in fs]
    if metodo == "mmc":
        H = mmc(*vidas) if horizonte is None else int(horizonte)
        ciclos = {f.nome: H // f.n for f in fs}
    else:
        H = max(vidas) if horizonte is None else int(horizonte)
        ciclos = {f.nome: 1 for f in fs}
    fig = make_subplots(rows=len(fs), cols=1, shared_xaxes=True, vertical_spacing=0.16 / max(len(fs) - 1, 1) if len(fs) > 1 else 0, subplot_titles=[f.nome for f in fs])
    y_max = 0.0
    y_min = 0.0
    for i, f in enumerate(fs, start=1):
        entradas, saidas = componentes_empilhadas(f, ciclos[f.nome])
        t = np.arange(len(entradas))
        comps = [
            {"nome": f"{f.nome} · entradas", "valores": entradas, "cor": cor_de(f.nome)},
            {"nome": "Investimentos / saídas", "valores": saidas, "cor": COR_SAIDA},
        ]
        lo, hi = _desenhar_setas(fig, t, comps, i, 1, legenda=False, rotulo_projeto=f"{f.nome} · ")
        y_min, y_max = min(y_min, lo), max(y_max, hi)
        fig.add_scatter(x=[None], y=[None], mode="markers", name=f"{f.nome} · entradas", marker=dict(symbol="triangle-up", size=11, color=cor_de(f.nome)), showlegend=True, hoverinfo="skip", row=i, col=1)
        if i == 1:
            fig.add_scatter(x=[None], y=[None], mode="markers", name="Investimentos / saídas", marker=dict(symbol="triangle-down", size=11, color=COR_SAIDA), showlegend=True, hoverinfo="skip", row=i, col=1)
        if ciclos[f.nome] > 1:
            for k in range(1, ciclos[f.nome]):
                fig.add_annotation(x=k * f.n, y=0, text="↻", showarrow=False, xshift=11, yshift=-12, font=dict(size=11, color=INK_MUDA), row=i, col=1)
    folga = 0.12 * (y_max - y_min or 1.0)
    for i in range(1, len(fs) + 1):
        fig.update_yaxes(range=[y_min - folga, y_max + folga], zeroline=True, zerolinecolor=INK, zerolinewidth=1.5, showgrid=False, tickfont=dict(size=11), row=i, col=1)
        fig.update_xaxes(showline=False, ticks="", showgrid=False, row=i, col=1)
    passo = 1 if H <= 20 else 5
    fig.update_xaxes(title_text="Ano", tickmode="linear", dtick=passo, range=[-0.7, H + 0.7], row=len(fs), col=1)
    if passo > 1:
        fig.update_xaxes(minor=dict(dtick=1, ticks="outside", ticklen=3, tickcolor=EIXO), ticks="outside", row=len(fs), col=1)
    if subtitulo is None:
        subtitulo = "↑ entradas · ↓ investimentos e saídas" + (" · ↻ início de um novo ciclo: a última entrada do ciclo anterior e o novo investimento aparecem lado a lado" if metodo == "mmc" else "")
    fig.update_layout(
        title=dict(text=titulo or (f"Diagramas de fluxo de caixa empilhados até o horizonte comum ({H} anos)" if metodo == "mmc" else "Diagramas de fluxo de caixa"), subtitle=dict(text=subtitulo, font=dict(size=11, color=INK_2))),
        height=altura or (140 + 190 * len(fs)), margin=dict(t=100), legend=dict(orientation="h", y=1.0, yanchor="bottom", x=1.0, xanchor="right"),
    )
    return fig


def grafico_fluxos_empilhados(fluxos, metodo: str = "mmc", titulo: str | None = None, altura: int | None = None, estilo: str = "setas") -> go.Figure:
    """Projetos replicados até o horizonte comum, um painel por projeto.

    ``estilo="setas"`` (padrão) desenha o diagrama clássico com setas, mantendo
    separadas as entradas e os investimentos nos anos de reinvestimento;
    ``estilo="barras"`` desenha o fluxo líquido em barras.
    """
    if estilo == "setas":
        return grafico_diagrama_fluxo(fluxos, metodo=metodo, titulo=titulo, altura=altura)
    fs = equalizar(fluxos, metodo)
    fig = make_subplots(rows=len(fs), cols=1, shared_xaxes=True, vertical_spacing=0.08, subplot_titles=[f.nome for f in fs])
    for i, f in enumerate(fs, start=1):
        v = f.media().valores
        fig.add_bar(x=f.t, y=v, name=f.nome, marker=dict(color=[cor_de(f.nome) if x >= 0 else COR_SAIDA for x in v], line=dict(width=0)), showlegend=False, hovertemplate="Ano %{x}: %{y:,.0f}<extra></extra>", row=i, col=1)
    fig.update_layout(title=titulo or f"Fluxos empilhados até o horizonte comum ({fs[0].n} anos)", height=altura or 420, bargap=0.3)
    fig.update_xaxes(title_text="Ano", row=len(fs), col=1)
    return fig


# --------------------------------------------------------------------------- #
# Perfil do VPL e taxa de Fisher
# --------------------------------------------------------------------------- #
def grafico_perfil_vpl(fluxos, taxa: float, taxas=None, fisher: bool = True, metodo: str | None = None, titulo: str | None = None, altura: int = 460) -> go.Figure:
    """Curvas VPL × taxa de desconto, TIR de cada projeto e interseções (taxas de Fisher)."""
    fs = como_fluxos(fluxos)
    if metodo:
        fs = equalizar(fs, metodo, taxa=taxa)
    fs = [f.media() for f in fs]
    taxas = np.linspace(0.0, 0.45, 181) if taxas is None else np.asarray(taxas, dtype=float)
    fig = go.Figure()
    for f in fs:
        y = vpl_perfil(f, taxas)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = tir(f)
        rotulo = f.nome if np.isnan(r) else f"{f.nome} (TIR {_pct(r)})"
        fig.add_scatter(x=taxas, y=y, mode="lines", name=rotulo, line=dict(color=cor_de(f.nome), width=2), hovertemplate=f"{f.nome}<br>taxa %{{x:.1%}}<br>VPL %{{y:,.0f}}<extra></extra>")
        if not np.isnan(r):
            fig.add_scatter(x=[r], y=[0], mode="markers", marker=dict(size=9, color=cor_de(f.nome), line=dict(color=SUPERFICIE, width=2)), showlegend=False, hovertemplate=f"TIR {f.nome}: %{{x:.2%}}<extra></extra>")
    if fisher and len(fs) > 1:
        import itertools

        for a, b in itertools.combinations(fs, 2):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                rs = taxa_fisher(a, b, todas=True)
            for r in np.atleast_1d(rs):
                if max(taxas.min(), 1e-6) <= r <= taxas.max():
                    y = vpl(a, r)
                    fig.add_scatter(x=[r], y=[y], mode="markers", marker=dict(symbol="diamond", size=10, color=INK, line=dict(color=SUPERFICIE, width=2)), name="Taxa de Fisher", showlegend=False, hovertemplate=f"Fisher {a.nome} × {b.nome}<br>taxa %{{x:.2%}}<br>VPL %{{y:,.0f}}<extra></extra>")
    fig.add_vline(x=taxa, line=dict(color=INK_MUDA, width=1, dash="dot"))
    fig.add_annotation(x=taxa, y=1, yref="paper", text=f"TMA {_pct(taxa)}", showarrow=False, xanchor="left", xshift=4, font=dict(size=11, color=INK_2))
    fig.update_layout(title=titulo or "Perfil do VPL e taxas de Fisher", xaxis=dict(title="Taxa de desconto", tickformat=".0%"), yaxis_title="VPL ($)", height=altura)
    return fig


# --------------------------------------------------------------------------- #
# Indicadores
# --------------------------------------------------------------------------- #
_FORMATOS = {"vpl": "moeda", "rau": "moeda", "investimento": "moeda", "tir": "pct", "tirm": "pct", "roia": "pct", "ibc": "num2", "payback": "anos", "payback_descontado": "anos"}
_MAIOR_MELHOR = {"payback": False, "payback_descontado": False}


def _rotulo(v: float, formato: str) -> str:
    if np.isnan(v):
        return "—"
    if formato == "moeda":
        return _moeda(v)
    if formato == "pct":
        return _pct(v)
    if formato == "anos":
        return f"{v:.1f} anos".replace(".", ",")
    return f"{v:.2f}".replace(".", ",")


def grafico_indicador(df: pd.DataFrame, coluna: str, titulo: str | None = None, referencia: float | None = None, rotulo_ref: str = "", altura: int = 320, maior_melhor: bool | None = None) -> go.Figure:
    """Barras horizontais de um indicador por projeto; o melhor em destaque."""
    formato = _FORMATOS.get(coluna, "num2")
    maior_melhor = _MAIOR_MELHOR.get(coluna, True) if maior_melhor is None else maior_melhor
    s = df[coluna].astype(float)
    ordem = s.sort_values(ascending=maior_melhor)
    melhor = ordem.index[-1] if maior_melhor else ordem.index[0]
    fig = go.Figure()
    fig.add_bar(
        y=list(ordem.index), x=ordem.values, orientation="h",
        marker=dict(color=[cor_de(str(i)) for i in ordem.index], line=dict(width=0)),
        text=[_rotulo(v, formato) + (" ★" if i == melhor else "") for i, v in ordem.items()],
        textposition="outside", textfont=dict(color=INK_2, size=12), cliponaxis=False,
        hovertemplate="%{y}: %{text}<extra></extra>", width=0.55,
    )
    if referencia is not None:
        fig.add_vline(x=referencia, line=dict(color=INK_MUDA, width=1, dash="dot"))
        fig.add_annotation(x=referencia, y=1, yref="paper", text=rotulo_ref or _rotulo(referencia, formato), showarrow=False, xanchor="left", xshift=4, font=dict(size=11, color=INK_2))
    fig.update_layout(title=titulo or coluna.upper(), height=altura, xaxis=dict(showgrid=True, gridcolor=GRADE, tickformat=".0%" if formato == "pct" else None), yaxis=dict(showgrid=False, zeroline=False), margin=dict(r=90))
    return fig


def grafico_painel_indicadores(df: pd.DataFrame, colunas: Sequence[str] = ("vpl", "rau", "tir", "tirm", "ibc", "payback_descontado"), referencias: Mapping[str, float] | None = None, titulo: str = "Indicadores por projeto", altura: int = 360) -> go.Figure:
    """Pequenos múltiplos: um painel por indicador, projetos coloridos de forma consistente."""
    from .metricas import COLUNAS_INDICADORES

    referencias = referencias or {}
    n = len(colunas)
    fig = make_subplots(rows=1, cols=n, subplot_titles=[COLUNAS_INDICADORES.get(c, c) for c in colunas], horizontal_spacing=0.06)
    ordem_proj = list(df.index)[::-1]
    for j, c in enumerate(colunas, start=1):
        formato = _FORMATOS.get(c, "num2")
        vals = df.loc[ordem_proj, c].astype(float)
        maior = _MAIOR_MELHOR.get(c, True)
        melhor = vals.idxmax() if maior else vals.idxmin()
        fig.add_bar(
            y=[str(i) for i in ordem_proj], x=vals.values, orientation="h",
            marker=dict(color=[cor_de(str(i)) for i in ordem_proj], line=dict(width=0)),
            text=[_rotulo(v, formato) + (" ★" if i == melhor else "") for i, v in vals.items()],
            textposition="outside", textfont=dict(size=11, color=INK_2), cliponaxis=False,
            showlegend=False, hovertemplate="%{y}: %{text}<extra></extra>", width=0.6, row=1, col=j,
        )
        if c in referencias:
            fig.add_vline(x=referencias[c], line=dict(color=INK_MUDA, width=1, dash="dot"), row=1, col=j)
        vmax = float(np.nanmax(np.append(vals.values, referencias.get(c, 0.0))))
        vmin = float(min(0.0, np.nanmin(vals.values)))
        fig.update_xaxes(showgrid=True, gridcolor=GRADE, tickformat=".0%" if formato == "pct" else None, showticklabels=False, range=[vmin * 1.1, vmax * 1.45], row=1, col=j)
        fig.update_yaxes(showgrid=False, zeroline=False, showticklabels=(j == 1), row=1, col=j)
    fig.update_layout(title=titulo, height=altura, margin=dict(r=40, t=90))
    return fig


def grafico_matriz_marginal(analise: pd.DataFrame, taxa: float | None = None, titulo: str | None = None, altura: int = 420) -> go.Figure:
    """Matriz desafiante (linha) × defensor (coluna) da análise marginal.

    A cor segue a decisão à TMA (azul: VPL marginal > 0 → desafiante vence;
    vermelho: defensor vence) e o texto mostra a TIR marginal (``*`` indica
    mais de uma TIR; ``—`` quando os investimentos são iguais).
    Recebe a tabela de :func:`greenseer.comparacao.analise_marginal`.
    """
    taxa = analise.attrs.get("taxa", taxa) if taxa is None else taxa
    nomes = list(dict.fromkeys(list(analise["Defensor"]) + list(analise["Desafiante"])))
    linhas_ = [n_ for n_ in nomes if n_ in set(analise["Desafiante"])]
    colunas_ = [n_ for n_ in nomes if n_ in set(analise["Defensor"])]
    n = len(nomes)
    pos_l = {k: i for i, k in enumerate(linhas_)}
    pos_c = {k: i for i, k in enumerate(colunas_)}
    z = np.full((len(linhas_), len(colunas_)), np.nan)
    texto = [["" for _ in colunas_] for _ in linhas_]
    hover = [["" for _ in colunas_] for _ in linhas_]
    for _, r in analise.iterrows():
        i, j = pos_l[r["Desafiante"]], pos_c[r["Defensor"]]
        z[i, j] = r["VPL marginal"]
        t = "—" if np.isnan(r["TIR marginal"]) else _pct(r["TIR marginal"]) + ("*" if r["Nº de TIRs"] > 1 else "")
        texto[i][j] = f"TIR {t}<br>VPL {_moeda(r['VPL marginal'])}"
        hover[i][j] = (
            f"desafiante {r['Desafiante']} × defensor {r['Defensor']}<br>horizonte {int(r['Horizonte'])} anos<br>"
            f"investimento marginal {_moeda(r['Investimento marginal'])}<br>TIRs: {r['TIRs']}<br>VPL marginal {_moeda(r['VPL marginal'])}<br>"
            f"RAU marginal {_moeda(r['RAU marginal'])}<br><b>preferido: {r['Preferido']}</b>"
        )
    amp = float(np.nanmax(np.abs(z))) or 1.0
    fig = go.Figure(
        go.Heatmap(
            z=z, x=colunas_, y=linhas_, text=texto, texttemplate="%{text}", textfont=dict(size=12), customdata=hover,
            colorscale=[[0, "#e34948"], [0.5, "#f0efec"], [1, "#2a78d6"]], zmid=0.0, zmin=-amp, zmax=amp,
            colorbar=dict(title="VPL marginal", tickformat=",.0f", thickness=12, outlinewidth=0),
            hoverongaps=False, hovertemplate="%{customdata}<extra></extra>", xgap=3, ygap=3,
        )
    )
    sub = f" à TMA de {_pct(taxa)}" if taxa is not None else ""
    fig.update_layout(
        title=dict(
            text=titulo or f"Análise marginal{sub}: azul = desafiante (linha) vence o defensor (coluna)",
            subtitle=dict(text="* fluxo marginal com mais de uma TIR — a decisão usa o sinal do VPL marginal · — investimentos iguais: compare os VPL diretamente", font=dict(size=11, color=INK_2)),
        ),
        height=altura, margin=dict(t=100), xaxis=dict(title="Defensor (menor investimento)", side="bottom"), yaxis=dict(title="Desafiante (maior investimento)", autorange="reversed", showgrid=False),
    )
    return fig


# --------------------------------------------------------------------------- #
# Cenários, sensibilidade
# --------------------------------------------------------------------------- #
def grafico_cenarios(resultado: pd.DataFrame, metrica: str = "vpl", resumo: pd.DataFrame | None = None, titulo: str | None = None, altura: int = 380, referencia: float | None = 0.0) -> go.Figure:
    """Faixa pessimista→otimista por projeto, com marcador por cenário e valor esperado (losango)."""
    formato = _FORMATOS.get(metrica, "num2")
    projetos = list(dict.fromkeys(resultado["Projeto"]))
    cenarios = list(dict.fromkeys(resultado["Cenário"]))
    simbolos = {c: s for c, s in zip(cenarios, ["triangle-down", "circle", "triangle-up", "square", "x"])}
    fig = go.Figure()
    for p in projetos:
        sub = resultado[resultado["Projeto"] == p]
        fig.add_scatter(x=[sub[metrica].min(), sub[metrica].max()], y=[p, p], mode="lines", line=dict(color=cor_de(p), width=3), showlegend=False, hoverinfo="skip")
    for c in cenarios:
        sub = resultado[resultado["Cenário"] == c]
        fig.add_scatter(x=sub[metrica], y=sub["Projeto"], mode="markers", name=c, marker=dict(symbol=simbolos.get(c, "circle"), size=12, color=[cor_de(p) for p in sub["Projeto"]], line=dict(color=SUPERFICIE, width=2)), hovertemplate="%{y} · " + c + "<br>%{text}<extra></extra>", text=[_rotulo(v, formato) for v in sub[metrica]])
    if resumo is not None and "Valor esperado" in resumo:
        fig.add_scatter(x=resumo.loc[projetos, "Valor esperado"], y=projetos, mode="markers", name="Valor esperado", marker=dict(symbol="diamond", size=13, color=INK, line=dict(color=SUPERFICIE, width=2)), hovertemplate="%{y}<br>E = %{text}<extra></extra>", text=[_rotulo(v, formato) for v in resumo.loc[projetos, "Valor esperado"]])
    if referencia is not None:
        fig.add_vline(x=referencia, line=dict(color=INK_MUDA, width=1, dash="dot"))
    fig.update_layout(title=titulo or f"{metrica.upper()} por cenário", height=altura, xaxis=dict(showgrid=True, gridcolor=GRADE, tickformat=".0%" if formato == "pct" else None), yaxis=dict(showgrid=False, zeroline=False, autorange="reversed"))
    return fig


def grafico_tornado(df: pd.DataFrame, titulo: str | None = None, altura: int | None = None) -> go.Figure:
    """Tornado da sensibilidade um-a-um: impacto do valor baixo e alto de cada parâmetro."""
    base = float(df["Base"].iloc[0])
    metrica = df.attrs.get("metrica", "vpl")
    formato = _FORMATOS.get(metrica, "num2")
    d = df.iloc[::-1]
    fig = go.Figure()
    fig.add_bar(y=d["Parâmetro"], x=d["Métrica (baixo)"] - base, base=base, orientation="h", name="Valor baixo", marker=dict(color=COR_SAIDA, line=dict(width=0)), text=[_rotulo(v, formato) for v in d["Métrica (baixo)"]], textposition="outside", textfont=dict(size=11, color=INK_2), cliponaxis=False, hovertemplate="%{y}<br>valor baixo → %{text}<extra></extra>", width=0.55)
    fig.add_bar(y=d["Parâmetro"], x=d["Métrica (alto)"] - base, base=base, orientation="h", name="Valor alto", marker=dict(color=COR_ENTRADA, line=dict(width=0)), text=[_rotulo(v, formato) for v in d["Métrica (alto)"]], textposition="outside", textfont=dict(size=11, color=INK_2), cliponaxis=False, hovertemplate="%{y}<br>valor alto → %{text}<extra></extra>", width=0.55)
    fig.add_vline(x=base, line=dict(color=INK, width=1))
    fig.add_annotation(x=base, y=1, yref="paper", text=f"base {_rotulo(base, formato)}", showarrow=False, xanchor="left", xshift=4, yshift=8, font=dict(size=11, color=INK_2))
    lo = float(min(d["Métrica (baixo)"].min(), d["Métrica (alto)"].min(), base))
    hi = float(max(d["Métrica (baixo)"].max(), d["Métrica (alto)"].max(), base))
    folga = 0.18 * (hi - lo or 1.0)
    fig.update_layout(title=titulo or f"Sensibilidade do {metrica.upper()}", barmode="overlay", height=altura or (140 + 42 * len(df)), xaxis=dict(showgrid=True, gridcolor=GRADE, tickformat=".0%" if formato == "pct" else None, range=[lo - folga, hi + folga]), yaxis=dict(showgrid=False, zeroline=False, automargin=True), margin=dict(r=60, l=40))
    return fig


# --------------------------------------------------------------------------- #
# Monte Carlo
# --------------------------------------------------------------------------- #
def grafico_distribuicoes(resultado, metrica: str = "vpl", projetos: Sequence[str] | None = None, limite: float | None = None, bins: int = 60, titulo: str | None = None, altura: int = 420, modo: str = "histograma") -> go.Figure:
    """Histogramas (ou violinos) sobrepostos da métrica simulada; legenda com P(< limite)."""
    df = resultado.metrica(metrica)
    projetos = list(projetos or df.columns)
    if limite is None:
        limite = {"tir": resultado.taxa, "tirm": resultado.taxa, "ibc": 1.0}.get(metrica, 0.0)
    formato = _FORMATOS.get(metrica, "num2")
    fig = go.Figure()
    lo, hi = np.nanpercentile(df[projetos].values, [0.5, 99.5])
    for p in projetos:
        x = df[p].dropna()
        prob = float(np.mean(x < limite))
        nome = f"{p} · P(< {_rotulo(limite, formato)}) = {_pct(prob)}"
        if modo == "violino":
            fig.add_violin(y=x, name=nome, line=dict(color=cor_de(p), width=1.5), fillcolor=cor_de(p), opacity=0.5, meanline_visible=True, box_visible=True, points=False, hoverinfo="y+name")
        else:
            bordas = np.linspace(lo, hi, bins + 1)
            contagens, _ = np.histogram(x, bins=bordas)  # valores fora da faixa 0,5–99,5% ficam de fora
            centros = 0.5 * (bordas[:-1] + bordas[1:])
            fig.add_bar(x=centros, y=contagens / len(x), width=(hi - lo) / bins, name=nome, marker=dict(color=cor_de(p), line=dict(width=0)), opacity=0.55, hovertemplate=p + "<br>%{x:,.0f}<br>freq. %{y:.1%}<extra></extra>")
    if modo == "violino":
        fig.add_hline(y=limite, line=dict(color=INK, width=1))
        fig.update_layout(yaxis=dict(tickformat=".0%" if formato == "pct" else None, showgrid=True, gridcolor=GRADE))
    else:
        fig.add_vline(x=limite, line=dict(color=INK, width=1))
        fig.update_layout(barmode="overlay", bargap=0.15, xaxis=dict(tickformat=".0%" if formato == "pct" else None, showgrid=False), yaxis=dict(title="Frequência", tickformat=".0%"))
    fig.update_layout(title=titulo or f"Distribuição simulada do {metrica.upper()}", height=altura, legend=dict(orientation="v", y=1, x=1.0, xanchor="right", yanchor="top"))
    return fig


def grafico_cdf(resultado, metrica: str = "vpl", projetos: Sequence[str] | None = None, limite: float | None = None, titulo: str | None = None, altura: int = 420) -> go.Figure:
    """Curvas de probabilidade acumulada (lê-se diretamente P(métrica ≤ x))."""
    df = resultado.metrica(metrica)
    projetos = list(projetos or df.columns)
    if limite is None:
        limite = {"tir": resultado.taxa, "tirm": resultado.taxa, "ibc": 1.0}.get(metrica, 0.0)
    formato = _FORMATOS.get(metrica, "num2")
    fig = go.Figure()
    for p in projetos:
        x = np.sort(df[p].dropna().values)
        y = np.arange(1, len(x) + 1) / len(x)
        if len(x) > 1_000:  # reduz a 1.000 pontos sem perda visual
            q = np.linspace(0, 1, 1_001)
            x, y = np.quantile(x, q), q
        fig.add_scatter(x=x, y=y, mode="lines", name=p, line=dict(color=cor_de(p), width=2), hovertemplate=p + "<br>%{x:,.0f}<br>P(≤ x) = %{y:.1%}<extra></extra>")
    fig.add_vline(x=limite, line=dict(color=INK, width=1))
    fig.update_layout(title=titulo or f"Probabilidade acumulada do {metrica.upper()}", height=altura, xaxis=dict(tickformat=".0%" if formato == "pct" else None, showgrid=True, gridcolor=GRADE), yaxis=dict(title="P(≤ x)", tickformat=".0%"))
    return fig


def grafico_risco_retorno(resumo: pd.DataFrame, x: str = "Desvio padrão", y: str = "Média", tamanho: str | None = None, titulo: str = "Risco × retorno", altura: int = 440, formato_x: str = "moeda", formato_y: str = "moeda") -> go.Figure:
    """Dispersão retorno esperado × risco; cada ponto é um projeto ou carteira."""
    fig = go.Figure()
    tam = None
    if tamanho and tamanho in resumo:
        t = resumo[tamanho].astype(float)
        tam = 10 + 30 * (t - t.min()) / max(t.max() - t.min(), 1e-9)
    for i, (nome, r) in enumerate(resumo.iterrows()):
        fig.add_scatter(x=[r[x]], y=[r[y]], mode="markers+text", text=[str(nome)], textposition="top center", textfont=dict(size=11, color=INK_2), name=str(nome), marker=dict(size=float(tam.iloc[i]) if tam is not None else 14, color=cor_de(str(nome)), line=dict(color=SUPERFICIE, width=2)), showlegend=False, hovertemplate=f"{nome}<br>{y}: {_rotulo(r[y], formato_y)}<br>{x}: {_rotulo(r[x], formato_x)}<extra></extra>")
    fig.update_layout(title=titulo, height=altura, xaxis=dict(title=x, showgrid=True, gridcolor=GRADE, tickformat=".0%" if formato_x == "pct" else None), yaxis=dict(title=y, tickformat=".0%" if formato_y == "pct" else None))
    return fig


def grafico_carteiras(resumo: pd.DataFrame, coluna: str = "Média", faixa: tuple[str, str] | None = ("P5", "P95"), titulo: str | None = None, altura: int = 400, formato: str = "moeda") -> go.Figure:
    """Carteiras ordenadas: ponto no valor esperado com faixa P5–P95 (ou barras, sem faixa)."""
    d = resumo.iloc[::-1]
    melhor = resumo[coluna].astype(float).idxmax()
    nomes = [str(i) for i in d.index]
    fig = go.Figure()
    tem_faixa = bool(faixa) and faixa[0] in d and faixa[1] in d
    if tem_faixa:
        for nome, r in d.iterrows():
            cor = COR_DESTAQUE if nome == melhor else COR_ENTRADA
            fig.add_scatter(x=[r[faixa[0]], r[faixa[1]]], y=[str(nome), str(nome)], mode="lines", line=dict(color=cor, width=3), opacity=0.45, showlegend=False, hoverinfo="skip")
            fig.add_scatter(x=[r[coluna]], y=[str(nome)], mode="markers+text", marker=dict(size=13, color=cor, line=dict(color=SUPERFICIE, width=2)), text=[_rotulo(r[coluna], formato) + (" ★" if nome == melhor else "")], textposition="middle right", textfont=dict(size=11, color=INK_2), showlegend=False, hovertemplate=f"{nome}<br>{coluna} {_rotulo(r[coluna], formato)}<br>{faixa[0]} {_rotulo(r[faixa[0]], formato)} · {faixa[1]} {_rotulo(r[faixa[1]], formato)}<extra></extra>")
        fig.add_annotation(x=0, y=1.02, xref="paper", yref="paper", text=f"ponto = {coluna.lower()} · faixa = {faixa[0]}–{faixa[1]}", showarrow=False, xanchor="left", font=dict(size=11, color=INK_2))
    else:
        fig.add_bar(y=nomes, x=d[coluna].values, orientation="h", marker=dict(color=[COR_DESTAQUE if i == melhor else COR_ENTRADA for i in d.index], line=dict(width=0)), text=[_rotulo(v, formato) + (" ★" if i == melhor else "") for i, v in d[coluna].items()], textposition="outside", textfont=dict(size=11, color=INK_2), cliponaxis=False, hovertemplate="%{y}: %{text}<extra></extra>", width=0.55)
    fig.add_vline(x=0, line=dict(color=INK, width=1))
    fig.update_layout(title=titulo or ("Carteiras: valor esperado e intervalo de confiança" if tem_faixa else "Carteiras"), height=altura, xaxis=dict(showgrid=True, gridcolor=GRADE, tickformat=".0%" if formato == "pct" else None), yaxis=dict(showgrid=False, zeroline=False, automargin=True), margin=dict(r=110))
    return fig


def grafico_leque(resultado, projeto: str, acumulado: bool = True, descontado: bool = True, titulo: str | None = None, altura: int = 400) -> go.Figure:
    """Leque P5–P95 / P25–P75 e mediana do fluxo acumulado (descontado) ao longo do tempo."""
    df = resultado.leque(projeto, acumulado=acumulado, descontado=descontado)
    cor = cor_de(projeto)
    x = list(df.index)
    fig = go.Figure()
    fig.add_scatter(x=x + x[::-1], y=list(df["P95"]) + list(df["P5"])[::-1], fill="toself", fillcolor=cor, opacity=0.15, line=dict(width=0), mode="lines", name="P5–P95", hoverinfo="skip")
    fig.add_scatter(x=x + x[::-1], y=list(df["P75"]) + list(df["P25"])[::-1], fill="toself", fillcolor=cor, opacity=0.3, line=dict(width=0), mode="lines", name="P25–P75", hoverinfo="skip")
    fig.add_scatter(x=x, y=df["P50"], mode="lines+markers", name="Mediana", line=dict(color=cor, width=2), marker=dict(size=7), hovertemplate="Ano %{x}<br>mediana %{y:,.0f}<extra></extra>")
    fig.update_layout(title=titulo or f"{projeto}: fluxo acumulado{' descontado' if descontado else ''} — leque de incerteza", height=altura, xaxis=dict(title="Ano", tickmode="array", tickvals=x), yaxis_title="Acumulado ($)")
    return fig


def grafico_prob_melhor(prob: pd.Series, titulo: str = "Probabilidade de ser o melhor projeto", altura: int = 300) -> go.Figure:
    """Barras com a probabilidade de cada projeto/carteira ser o melhor."""
    d = prob.sort_values()
    fig = go.Figure(go.Bar(y=[str(i) for i in d.index], x=d.values, orientation="h", marker=dict(color=[cor_de(str(i)) for i in d.index], line=dict(width=0)), text=[_pct(v) for v in d.values], textposition="outside", textfont=dict(size=11, color=INK_2), cliponaxis=False, hovertemplate="%{y}: %{text}<extra></extra>", width=0.55))
    fig.update_layout(title=titulo, height=altura, xaxis=dict(tickformat=".0%", showgrid=True, gridcolor=GRADE, range=[0, max(1.0, float(d.max()) * 1.15)]), yaxis=dict(showgrid=False, zeroline=False), margin=dict(r=70))
    return fig


# --------------------------------------------------------------------------- #
# Exportação
# --------------------------------------------------------------------------- #
def exportar(fig: go.Figure, caminho_base: str, formatos: Sequence[str] = ("html", "png"), largura: int = 1100, altura: int | None = None, escala: float = 2.0) -> list[str]:
    """Grava a figura como HTML interativo e/ou PNG (requer kaleido + Chrome)."""
    import os

    os.makedirs(os.path.dirname(caminho_base) or ".", exist_ok=True)
    gerados = []
    if "html" in formatos:
        p = caminho_base + ".html"
        fig.write_html(p, include_plotlyjs="cdn", full_html=True)
        gerados.append(p)
    for ext in [f for f in formatos if f != "html"]:
        p = f"{caminho_base}.{ext}"
        try:
            fig.write_image(p, width=largura, height=altura or fig.layout.height or 450, scale=escala)
            gerados.append(p)
        except Exception as exc:  # kaleido/Chrome ausentes
            warnings.warn(f"não foi possível exportar {ext}: {exc}")
    return gerados


# --------------------------------------------------------------------------- #
# Árvores de cenários, distribuições discretas e curva de indiferença
# --------------------------------------------------------------------------- #
def grafico_arvore(analise, projeto: str, metrica: str = "vpl", titulo: str | None = None, altura: int | None = None) -> go.Figure:
    """Árvore de hipóteses → alternativas com probabilidades nas arestas e a métrica nas folhas."""
    from .relatorios import fmt_moeda, fmt_pct

    formato = _FORMATOS.get(metrica, "num")
    rot = lambda v: _rotulo(v, formato)
    arvore = analise.arvores[projeto]
    res = analise.resultado[analise.resultado["Projeto"] == projeto].reset_index(drop=True)
    resumo = analise.resumo(metrica).loc[projeto]
    por_hip = analise.resumo_por_hipotese(metrica)
    por_hip = por_hip[por_hip["Projeto"] == projeto].set_index("Hipótese")
    cor = cor_de(projeto)
    n_folhas = len(res)
    ys = np.linspace(1.0, 0.0, n_folhas) if n_folhas > 1 else np.array([0.5])
    folhas = {row["Cenário"]: (2.0, ys[i]) for i, (_, row) in enumerate(res.iterrows())}
    hips = {}
    for h in arvore.hipoteses:
        yy = [folhas[c][1] for c in res["Cenário"] if res.loc[res["Cenário"] == c, "Hipótese"].iloc[0] == h.nome]
        hips[h.nome] = (1.0, float(np.mean(yy)))
    raiz = (0.0, float(np.mean([v[1] for v in hips.values()])))
    fig = go.Figure()
    # arestas
    xs, ys_ = [], []
    for h in arvore.hipoteses:
        xs += [raiz[0], hips[h.nome][0], None]; ys_ += [raiz[1], hips[h.nome][1], None]
    for _, row in res.iterrows():
        a, b = hips[row["Hipótese"]], folhas[row["Cenário"]]
        xs += [a[0], b[0], None]; ys_ += [a[1], b[1], None]
    fig.add_scatter(x=xs, y=ys_, mode="lines", line=dict(color=EIXO, width=1.5), hoverinfo="skip", showlegend=False)
    # rótulos de probabilidade nas arestas
    for h in arvore.hipoteses:
        fig.add_annotation(x=(raiz[0] + hips[h.nome][0]) / 2, y=(raiz[1] + hips[h.nome][1]) / 2, text=f"P = {_pct(h.probabilidade)}", showarrow=False, font=dict(size=11, color=INK_2), bgcolor=SUPERFICIE, yshift=8)
    for _, row in res.iterrows():
        a, b = hips[row["Hipótese"]], folhas[row["Cenário"]]
        if row["P(alt | hip)"] is not None and not (isinstance(row["P(alt | hip)"], float) and np.isnan(row["P(alt | hip)"])) and row["P(alt | hip)"] < 1:
            fig.add_annotation(x=(a[0] + b[0]) / 2, y=(a[1] + b[1]) / 2, text=_pct(row["P(alt | hip)"]), showarrow=False, font=dict(size=10.5, color=INK_2), bgcolor=SUPERFICIE)
    # nós
    fig.add_scatter(x=[raiz[0]], y=[raiz[1]], mode="markers+text", marker=dict(size=22, color=cor, line=dict(color=SUPERFICIE, width=2)), text=[f"<b>{projeto}</b>"], textposition="middle left", textfont=dict(size=13, color=INK), hovertemplate=f"<b>{projeto}</b><br>E[{metrica.upper()}] = {rot(resumo['Valor esperado'])}<br>DP = {rot(resumo['Desvio padrão'])}<br>CV = {resumo['CV']:.2f}<extra></extra>".replace(".", ","), showlegend=False)
    # duas anotações separadas: um texto com dois cifrões seria interpretado como LaTeX pelo Plotly
    fig.add_annotation(x=raiz[0], y=raiz[1], text=f"E = {rot(resumo['Valor esperado'])}", showarrow=False, xanchor="center", yshift=-24, font=dict(size=11, color=INK_2))
    fig.add_annotation(x=raiz[0], y=raiz[1], text=f"DP = {rot(resumo['Desvio padrão'])} · CV = {resumo['CV']:.2f}".replace("CV = 0.", "CV = 0,"), showarrow=False, xanchor="center", yshift=-40, font=dict(size=11, color=INK_2))
    for h in arvore.hipoteses:
        x, y = hips[h.nome]
        e_cond = por_hip.loc[h.nome, "E[métrica | hipótese]"] if h.nome in por_hip.index else np.nan
        fig.add_scatter(x=[x], y=[y], mode="markers", marker=dict(size=16, color=cor, opacity=0.85, line=dict(color=SUPERFICIE, width=2)), hovertemplate=f"<b>{h.nome}</b><br>P = {_pct(h.probabilidade)}<br>E[{metrica.upper()} | hipótese] = {rot(e_cond)}<extra></extra>", showlegend=False)
        fig.add_annotation(x=x, y=y, text=f"<b>{h.nome}</b><br>E = {rot(e_cond)}", showarrow=False, yshift=26 if len(arvore.hipoteses) > 1 else 0, font=dict(size=11, color=INK), align="center")
    for _, row in res.iterrows():
        x, y = folhas[row["Cenário"]]
        v = row[metrica]
        alt = row["Cenário"].split(" · ", 1)[1] if " · " in row["Cenário"] else "caso-base"
        fig.add_scatter(x=[x], y=[y], mode="markers", marker=dict(size=11, color=COR_ENTRADA if v >= 0 else COR_SAIDA, line=dict(color=SUPERFICIE, width=1.5)), hovertemplate=f"<b>{row['Cenário']}</b><br>P conjunta = {_pct(row['Probabilidade'])}<br>{metrica.upper()} = {rot(v)}<extra></extra>", showlegend=False)
        fig.add_annotation(x=x, y=y, text=f"{alt} · <b>{rot(v)}</b> · p = {_pct(row['Probabilidade'])}", showarrow=False, xanchor="left", xshift=12, font=dict(size=11, color=INK_2 if v >= 0 else COR_SAIDA), align="left")
    fig.update_layout(
        title=titulo or f"{projeto} — árvore de cenários ({metrica.upper()} em cada folha)",
        xaxis=dict(visible=False, range=[-0.6, 3.4]), yaxis=dict(visible=False, range=[-0.12, 1.14]),
        height=altura or max(380, 36 * n_folhas + 120), margin=dict(l=20, r=20, t=70, b=20), showlegend=False, hovermode="closest",
    )
    return fig


def grafico_distribuicao_discreta(analise, projeto: str, metrica: str = "vpl", titulo: str | None = None, altura: int = 380) -> go.Figure:
    """Distribuição de probabilidade da métrica (um ponto por cenário), com o valor esperado e a faixa ± DP."""
    formato = _FORMATOS.get(metrica, "num")
    res = analise.resultado[analise.resultado["Projeto"] == projeto].sort_values(metrica)
    resumo = analise.resumo(metrica).loc[projeto]
    e, dp = resumo["Valor esperado"], resumo["Desvio padrão"]
    v = res[metrica].to_numpy(dtype=float); p = res["Probabilidade"].to_numpy(dtype=float)
    largura = 0.012 * (max(v.max(), e + dp) - min(v.min(), e - dp) or 1.0)
    fig = go.Figure()
    fig.add_vrect(x0=e - dp, x1=e + dp, fillcolor=cor_de(projeto), opacity=0.08, line_width=0, annotation_text="E ± DP", annotation_position="top left", annotation_font=dict(size=11, color=INK_2))
    fig.add_bar(x=v, y=p, width=largura, marker=dict(color=[cor_de(projeto) if x >= 0 else COR_SAIDA for x in v], line=dict(width=0)), customdata=res["Cenário"], hovertemplate="%{customdata}<br>" + metrica.upper() + " %{x:,.0f}<br>probabilidade %{y:.1%}<extra></extra>", showlegend=False)
    fig.add_vline(x=e, line=dict(color=INK, width=1.5))
    fig.add_annotation(x=e, y=1, yref="paper", text=f"E = {_rotulo(e, formato)}", showarrow=False, xanchor="left", xshift=4, font=dict(size=11, color=INK))
    if v.min() < 0 < v.max():
        fig.add_vline(x=0, line=dict(color=INK_MUDA, width=1, dash="dot"))
    fig.update_layout(title=titulo or f"{projeto} — distribuição do {metrica.upper()} por cenário (DP = {_rotulo(dp, formato)}, CV = {resumo['CV']:.2f})".replace("CV = 0.", "CV = 0,"), xaxis=dict(title=metrica.upper(), tickformat=".0%" if formato == "pct" else None, showgrid=False), yaxis=dict(title="Probabilidade", tickformat=".0%"), height=altura, bargap=0)
    return fig


def grafico_curva_indiferenca(curva, pontos=None, extrapolar: bool = False, titulo: str | None = None, altura: int = 420, cv_max: float | None = None) -> go.Figure:
    """Curva CV → taxa ajustada ao risco, com o ponto de cada projeto e a taxa resultante."""
    from .cenarios import taxa_ajustada_ao_risco

    pontos = {} if pontos is None else (pontos.to_dict() if hasattr(pontos, "to_dict") else dict(pontos))
    xs_c = [c[0] for c in sorted(curva)]
    xmax = max(cv_max or 0, max(xs_c), *(1.1 * v for v in pontos.values())) if pontos else max(cv_max or 0, max(xs_c))
    xs = np.linspace(0, xmax, 241)
    ys = taxa_ajustada_ao_risco(xs, curva, extrapolar)
    fig = go.Figure()
    fig.add_scatter(x=xs, y=ys, mode="lines", line=dict(color=INK_2, width=2), name="Curva de indiferença", hovertemplate="CV %{x:.2f} → %{y:.1%}<extra></extra>")
    fig.add_scatter(x=xs_c, y=[c[1] for c in sorted(curva)], mode="markers", marker=dict(symbol="diamond", size=9, color=INK_2, line=dict(color=SUPERFICIE, width=1.5)), name="Pontos informados", hovertemplate="CV %{x:.2f} → %{y:.1%}<extra></extra>")
    if extrapolar and xmax > max(xs_c):
        fig.add_vrect(x0=max(xs_c), x1=xmax, fillcolor=INK_MUDA, opacity=0.07, line_width=0, annotation_text="extrapolação", annotation_position="top left", annotation_font=dict(size=11, color=INK_2))
    for nome, cv in pontos.items():
        tx = float(taxa_ajustada_ao_risco(cv, curva, extrapolar))
        cor = cor_de(str(nome))
        fig.add_scatter(x=[0, cv, cv], y=[tx, tx, 0], mode="lines", line=dict(color=cor, width=1, dash="dot"), hoverinfo="skip", showlegend=False)
        fig.add_scatter(x=[cv], y=[tx], mode="markers+text", marker=dict(size=13, color=cor, line=dict(color=SUPERFICIE, width=2)), text=[f"{nome}: CV {cv:.2f} → {_pct(tx)}".replace(f"{cv:.2f}", f"{cv:.2f}".replace(".", ","))], textposition="top left", textfont=dict(size=11.5, color=INK), name=str(nome), hovertemplate=f"<b>{nome}</b><br>CV {cv:.2f}<br>taxa ajustada {_pct(tx)}<extra></extra>")
    fig.update_layout(title=titulo or "Curva de indiferença: coeficiente de variação → taxa de desconto ajustada ao risco", xaxis=dict(title="Coeficiente de variação (risco)", showgrid=True, gridcolor=GRADE, range=[0, xmax * 1.02]), yaxis=dict(title="Taxa de desconto ajustada", tickformat=".0%", rangemode="tozero"), height=altura, legend=dict(orientation="h", y=1.0, yanchor="bottom", x=1, xanchor="right"))
    return fig
