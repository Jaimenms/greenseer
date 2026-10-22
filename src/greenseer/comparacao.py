"""Comparação de projetos: análise marginal, vidas diferentes e carteiras.

* :func:`equalizar` — iguala horizontes (empilhamento até o MMC das vidas,
  extensão com zeros ou truncamento com valor econômico remanescente);
* :func:`fluxo_marginal` / :func:`analise_marginal` — fluxo *desafiante −
  defensor* para todos os pares e sua TIR marginal (taxa de Fisher);
* :func:`matriz_tir_marginal` — matriz com a TIR marginal de cada par;
* :func:`campeao_incremental` — seleção sequencial por análise incremental;
* :func:`carteiras` — todas as combinações de projetos independentes sujeitas
  a restrição de quantidade ou orçamento;
* :func:`ranking` — posição de cada projeto segundo cada critério.
"""

from __future__ import annotations

import itertools
import warnings
from typing import Literal, Sequence

import numpy as np
import pandas as pd

from .cashflow import FluxoDeCaixa, como_fluxos, mmc, somar
from .metricas import ibc, indicadores, payback, rau, tir, tirm, vp, vpl

MetodoHorizonte = Literal["mmc", "estender", "truncar"]


# --------------------------------------------------------------------------- #
# Horizontes
# --------------------------------------------------------------------------- #
def equalizar(
    fluxos,
    metodo: MetodoHorizonte = "mmc",
    horizonte: int | None = None,
    taxa: float | None = None,
) -> list[FluxoDeCaixa]:
    """Iguala o horizonte de um conjunto de fluxos.

    * ``mmc`` — replica cada projeto até o mínimo múltiplo comum das vidas
      (ou ``horizonte``, que deve ser múltiplo de todas): hipótese de
      repetição, coerente com o RAU;
    * ``estender`` — completa com zeros até a maior vida (ou ``horizonte``):
      hipótese de não repetição, coerente com o VPL simples;
    * ``truncar`` — corta no menor horizonte (ou ``horizonte``) e lança, no
      corte, o valor presente dos fluxos remanescentes (valor econômico
      residual; exige ``taxa``).
    """
    fluxos = como_fluxos(fluxos)
    vidas = [f.n for f in fluxos]
    if metodo == "mmc":
        H = mmc(*vidas) if horizonte is None else int(horizonte)
        return [f.replicar_ate(H) for f in fluxos]
    if metodo == "estender":
        H = max(vidas) if horizonte is None else int(horizonte)
        return [f.estender(H) for f in fluxos]
    if metodo == "truncar":
        if taxa is None:
            raise ValueError("truncar exige a taxa de desconto para o valor residual econômico")
        H = min(vidas) if horizonte is None else int(horizonte)
        saida = []
        for f in fluxos:
            if f.n <= H:
                saida.append(f.estender(H))
                continue
            resto = FluxoDeCaixa(f.valores[..., H:], f.nome)
            g = f.truncar(H).copiar()
            g.valores[..., H] += vp(resto, taxa, 1)
            saida.append(g)
        return saida
    raise ValueError(f"método desconhecido: {metodo}")


# --------------------------------------------------------------------------- #
# Análise marginal
# --------------------------------------------------------------------------- #
def fluxo_marginal(
    defensor,
    desafiante,
    metodo: MetodoHorizonte = "mmc",
    horizonte: int | None = None,
    taxa: float | None = None,
) -> FluxoDeCaixa:
    """Fluxo marginal ``desafiante − defensor`` em horizonte equalizado."""
    a, b = equalizar([defensor, desafiante], metodo, horizonte, taxa)
    m = b - a
    m.nome = f"{b.nome} − {a.nome}"
    return m


def _ordenar_por_investimento(fluxos: list[FluxoDeCaixa]) -> list[FluxoDeCaixa]:
    return sorted(fluxos, key=lambda f: (float(np.atleast_1d(f.investimento_inicial)[0]), f.nome))


def analise_marginal(
    fluxos,
    taxa: float,
    metodo: MetodoHorizonte = "mmc",
    horizonte: int | None = None,
    taxa_reinvestimento: float | None = None,
) -> pd.DataFrame:
    """TIR, VPL e RAU do fluxo marginal para todos os pares de projetos.

    Os projetos são ordenados por investimento inicial; em cada par o de menor
    investimento é o *defensor* e o de maior o *desafiante*. O desafiante é
    preferido se a TIR marginal supera a taxa mínima (equivalente a VPL
    marginal > 0 para fluxos marginais convencionais).
    """
    fluxos = _ordenar_por_investimento(como_fluxos(fluxos))
    linhas = []
    for a, b in itertools.combinations(fluxos, 2):
        m = fluxo_marginal(a, b, metodo, horizonte, taxa)
        inv_marg = float(m.investimento_inicial)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raizes = tir(m, todas=True)
        mesmo_investimento = abs(inv_marg) < 1e-9
        if mesmo_investimento or len(raizes) == 0:
            t_marg = float("nan")
        else:
            t_marg = float(min(raizes, key=abs))
        v_marg = vpl(m, taxa)
        linhas.append(
            {
                "Defensor": a.nome,
                "Desafiante": b.nome,
                "Horizonte": m.n,
                "Investimento marginal": inv_marg,
                "VPL marginal": v_marg,
                "TIR marginal": t_marg,
                "TIRM marginal": tirm(m, taxa if taxa_reinvestimento is None else taxa_reinvestimento),
                "RAU marginal": rau(m, taxa),
                "Nº de TIRs": int(len(raizes)),
                "TIRs": "mesmo investimento" if mesmo_investimento else (", ".join(f"{100 * r:.1f}%" for r in raizes) or "nenhuma"),
                "Preferido": b.nome if v_marg > 0 else a.nome,
            }
        )
    df = pd.DataFrame(linhas)
    df.attrs["taxa"] = taxa
    df.attrs["metodo"] = metodo
    return df


def matriz_marginal(fluxos, taxa: float, metodo: MetodoHorizonte = "mmc") -> pd.DataFrame:
    """Tabela desafiante (linha) × defensor (coluna) com VPL/TIR marginal e decisão (triângulo inferior)."""
    df = analise_marginal(fluxos, taxa, metodo)
    nomes = list(dict.fromkeys(list(df["Defensor"]) + list(df["Desafiante"])))
    ordem = sorted(nomes, key=lambda n: df.loc[df["Desafiante"] == n, "Investimento marginal"].sum())
    fluxos_ord = _ordenar_por_investimento(como_fluxos(fluxos))
    ordem = [f.nome for f in fluxos_ord]
    M = pd.DataFrame(index=pd.Index(ordem, name="Desafiante ↓ / Defensor →"), columns=ordem, dtype=object)
    for _, r in df.iterrows():
        tir_txt = "—" if np.isnan(r["TIR marginal"]) else f"{100 * r['TIR marginal']:.1f}%" + ("*" if r["Nº de TIRs"] > 1 else "")
        M.loc[r["Desafiante"], r["Defensor"]] = f"TIR {tir_txt} · VPL {r['VPL marginal']:,.0f} → {r['Preferido']}"
    return M.fillna("")


def matriz_tir_marginal(
    fluxos, taxa: float, metodo: MetodoHorizonte = "mmc", valor: str = "TIR marginal"
) -> pd.DataFrame:
    """Matriz simétrica (projeto × projeto) com a TIR (ou VPL/RAU) marginal do par."""
    fluxos = _ordenar_por_investimento(como_fluxos(fluxos))
    nomes = [f.nome for f in fluxos]
    df = analise_marginal(fluxos, taxa, metodo)
    M = pd.DataFrame(np.nan, index=nomes, columns=nomes)
    for _, r in df.iterrows():
        M.loc[r["Desafiante"], r["Defensor"]] = r[valor]
        M.loc[r["Defensor"], r["Desafiante"]] = r[valor]
    M.index.name = "Projeto"
    return M


def campeao_incremental(
    fluxos, taxa: float, metodo: MetodoHorizonte = "mmc", horizonte: int | None = None
) -> tuple[str, pd.DataFrame]:
    """Seleção por análise incremental sequencial (mutuamente excludentes).

    Ordena por investimento; o defensor inicial é o menor projeto *viável*
    (VPL > 0); cada desafiante substitui o defensor se a TIR marginal supera a
    taxa (VPL marginal > 0). Devolve o campeão e a trilha de decisões.
    """
    fluxos = _ordenar_por_investimento(como_fluxos(fluxos))
    defensor = None
    trilha = []
    for f in fluxos:
        if defensor is None:
            v = vpl(f, taxa)
            trilha.append(
                {"Defensor": "—", "Desafiante": f.nome, "VPL marginal": v, "TIR marginal": tir(f), "Vence": f.nome if v > 0 else "—"}
            )
            if v > 0:
                defensor = f
            continue
        m = fluxo_marginal(defensor, f, metodo, horizonte, taxa)
        v = vpl(m, taxa)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            t = tir(m)
        vence = f if v > 0 else defensor
        trilha.append({"Defensor": defensor.nome, "Desafiante": f.nome, "VPL marginal": v, "TIR marginal": t, "Vence": vence.nome})
        defensor = vence
    return (defensor.nome if defensor is not None else "—"), pd.DataFrame(trilha)


# --------------------------------------------------------------------------- #
# Carteiras (projetos independentes)
# --------------------------------------------------------------------------- #
def carteiras(
    fluxos,
    taxa: float,
    k: int | Sequence[int] | None = None,
    orcamento: float | None = None,
    taxa_reinvestimento: float | None = None,
    incluir_vazia: bool = False,
) -> tuple[pd.DataFrame, dict[str, FluxoDeCaixa]]:
    """Enumera combinações de projetos independentes e as avalia.

    Parameters
    ----------
    k
        Tamanho(s) de carteira permitido(s) (ex.: ``2`` → "apenas dois
        projetos"). ``None`` = qualquer tamanho.
    orcamento
        Limite para a soma dos investimentos iniciais.

    Returns
    -------
    (tabela, fluxos)
        Tabela com VPL (aditivo), soma dos RAU (hipótese de repetição), TIR e
        TIRM do fluxo agregado, IBC, paybacks e investimento; e o dicionário
        com o fluxo agregado de cada carteira.
    """
    fluxos = como_fluxos(fluxos)
    tamanhos = range(0 if incluir_vazia else 1, len(fluxos) + 1) if k is None else ([k] if isinstance(k, int) else list(k))
    linhas, agregados = [], {}
    ind_individuais = {f.nome: indicadores(f, taxa, taxa_reinvestimento) for f in fluxos}
    for tam in tamanhos:
        for combo in itertools.combinations(fluxos, tam):
            if not combo:
                continue
            inv = sum(float(np.atleast_1d(f.investimento_inicial)[0]) for f in combo)
            if orcamento is not None and inv > orcamento + 1e-9:
                continue
            nome = " + ".join(f.nome for f in combo)
            agg = somar(combo, nome)
            agregados[nome] = agg
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                t_agg = tir(agg)
            linhas.append(
                {
                    "Carteira": nome,
                    "Nº projetos": len(combo),
                    "Investimento": inv,
                    "VPL": vpl(agg, taxa),
                    "RAU (soma)": sum(ind_individuais[f.nome]["rau"] for f in combo),
                    "TIR": t_agg,
                    "TIRM": tirm(agg, taxa if taxa_reinvestimento is None else taxa_reinvestimento),
                    "IBC": ibc(agg, taxa),
                    "Payback ajustado": payback(agg, taxa),
                    "Horizonte": agg.n,
                }
            )
    df = pd.DataFrame(linhas).sort_values("VPL", ascending=False).reset_index(drop=True)
    return df, agregados


# --------------------------------------------------------------------------- #
# Ranking por critério
# --------------------------------------------------------------------------- #
def ranking(fluxos, taxa: float, taxa_reinvestimento: float | None = None) -> pd.DataFrame:
    """Indicadores + posição de cada projeto em cada critério (1 = melhor)."""
    from .metricas import tabela_indicadores

    df = tabela_indicadores(fluxos, taxa, taxa_reinvestimento)
    maior_melhor = ["vpl", "tir", "tirm", "rau", "ibc", "roia"]
    menor_melhor = ["payback", "payback_descontado"]
    for c in maior_melhor:
        df[f"rank_{c}"] = df[c].rank(ascending=False, method="min").astype(int)
    for c in menor_melhor:
        df[f"rank_{c}"] = df[c].rank(ascending=True, method="min").astype(int)
    return df
