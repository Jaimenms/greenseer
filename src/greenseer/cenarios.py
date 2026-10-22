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
    """Conjunto nomeado de alterações sobre o caso-base.

    * ``parametros`` — valores que substituem os do caso-base;
    * ``variacoes`` — ``{parametro: ±fração}``: multiplica o valor-base por
      ``(1 + fração)`` ("CPV 2% maior" → ``{"p2_cpv": +0.02}``);
    * ``variacoes_pp`` — ``{parametro: ±valor}``: soma ao valor-base (pontos
      percentuais ou unidades).
    ``hipotese`` e ``probabilidade_condicional`` são preenchidos quando o
    cenário nasce da expansão de uma :class:`ArvoreDeCenarios`.
    """

    nome: str
    parametros: dict = field(default_factory=dict)
    probabilidade: float | None = None
    descricao: str = ""
    variacoes: dict = field(default_factory=dict)
    variacoes_pp: dict = field(default_factory=dict)
    hipotese: str | None = None
    probabilidade_condicional: float | None = None


def aplicar_cenario(base: Mapping, cenario: Cenario) -> dict:
    """Parâmetros efetivos de um cenário: base + substituições + variações."""
    p = {**base, **cenario.parametros}
    for k, v in cenario.variacoes.items():
        if k not in p:
            raise KeyError(f"variação sobre parâmetro desconhecido: {k}")
        p[k] = p[k] * (1.0 + float(v))
    for k, v in cenario.variacoes_pp.items():
        if k not in p:
            raise KeyError(f"variação sobre parâmetro desconhecido: {k}")
        p[k] = p[k] + float(v)
    return p


@dataclass
class Hipotese:
    """Ramo de primeiro nível de uma árvore de cenários (ex.: pessimista 10%).

    Se tiver ``alternativas`` (lista de :class:`Cenario` com probabilidades
    *condicionais* que somam 1), cada uma vira uma folha com probabilidade
    conjunta ``P(hipótese) × P(alternativa | hipótese)``. Sem alternativas, a
    hipótese é ela própria a folha, com as alterações informadas.
    """

    nome: str
    probabilidade: float
    alternativas: list = field(default_factory=list)
    parametros: dict = field(default_factory=dict)
    variacoes: dict = field(default_factory=dict)
    variacoes_pp: dict = field(default_factory=dict)
    descricao: str = ""


@dataclass
class ArvoreDeCenarios:
    """Árvore de hipóteses → alternativas, ponderada por probabilidades.

    É a estrutura do caso Gama (Parte 2): três hipóteses (pessimista / mais
    provável / otimista) e, dentro das extremas, alternativas com
    probabilidades condicionais. :meth:`expandir` devolve a lista plana de
    cenários com as probabilidades conjuntas, que é o que
    :class:`AnaliseDeCenarios` consome.
    """

    hipoteses: list
    nome: str = ""
    parametro: str | None = None

    def validar(self, tol: float = 1e-9) -> None:
        soma = sum(h.probabilidade for h in self.hipoteses)
        if abs(soma - 1.0) > tol:
            raise ValueError(f"probabilidades das hipóteses somam {soma:.4f} (esperado 1)")
        for h in self.hipoteses:
            if h.alternativas:
                sc = sum((a.probabilidade or 0.0) for a in h.alternativas)
                if abs(sc - 1.0) > tol:
                    raise ValueError(f"probabilidades condicionais em '{h.nome}' somam {sc:.4f} (esperado 1)")

    def expandir(self) -> list[Cenario]:
        self.validar()
        folhas: list[Cenario] = []
        for h in self.hipoteses:
            if not h.alternativas:
                folhas.append(Cenario(h.nome, dict(h.parametros), h.probabilidade, h.descricao, dict(h.variacoes), dict(h.variacoes_pp), h.nome, 1.0))
                continue
            for a in h.alternativas:
                folhas.append(Cenario(
                    f"{h.nome} · {a.nome}", {**h.parametros, **a.parametros}, h.probabilidade * (a.probabilidade or 0.0), a.descricao or h.descricao,
                    {**h.variacoes, **a.variacoes}, {**h.variacoes_pp, **a.variacoes_pp}, h.nome, a.probabilidade,
                ))
        return folhas

    def tabela(self) -> pd.DataFrame:
        """Hipótese, P(hipótese), alternativa, P(alt | hip), P conjunta e as alterações."""
        linhas = []
        for c in self.expandir():
            alt = c.nome.split(" · ", 1)[1] if " · " in c.nome else "—"
            hip = next(h for h in self.hipoteses if h.nome == c.hipotese)
            desc = ", ".join([f"{k} = {v}" for k, v in c.parametros.items()] + [f"{k} {v:+.1%}" for k, v in c.variacoes.items()] + [f"{k} {v:+g} p.p." for k, v in c.variacoes_pp.items()]) or "caso-base"
            linhas.append({"Hipótese": c.hipotese, "P(hipótese)": hip.probabilidade, "Alternativa": alt, "P(alt | hip)": c.probabilidade_condicional, "P conjunta": c.probabilidade, "Alteração": desc, "Cenário": c.nome})
        return pd.DataFrame(linhas)

    def como_distribuicao(self, base: Mapping, parametro: str):
        """Valores e probabilidades que ``parametro`` assume nas folhas (distribuição discreta)."""
        valores = [aplicar_cenario(base, c)[parametro] for c in self.expandir()]
        probs = [c.probabilidade for c in self.expandir()]
        return np.asarray(valores, dtype=float), np.asarray(probs, dtype=float)


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


def taxa_ajustada_ao_risco(cv, curva: Sequence[tuple[float, float]], extrapolar: bool = False):
    """Taxa de desconto ajustada ao risco por interpolação linear numa curva CV → taxa.

    ``curva`` = lista de pares ``(cv, taxa)`` como no gráfico das notas (ex.:
    ``[(0.1, 0.18), (0.3, 0.19), (0.5, 0.20), (0.7, 0.22), (0.9, 0.25)]``).
    Fora do intervalo a taxa é mantida constante; com ``extrapolar=True`` o
    último (ou primeiro) segmento é prolongado linearmente.
    """
    pontos = sorted(curva)
    xs = np.asarray([p[0] for p in pontos], dtype=float)
    ys = np.asarray([p[1] for p in pontos], dtype=float)
    x = np.asarray(cv, dtype=float)
    r = np.interp(x, xs, ys)
    if extrapolar and len(xs) > 1:
        decl_fim = (ys[-1] - ys[-2]) / (xs[-1] - xs[-2])
        decl_ini = (ys[1] - ys[0]) / (xs[1] - xs[0])
        r = np.where(x > xs[-1], ys[-1] + decl_fim * (x - xs[-1]), r)
        r = np.where(x < xs[0], ys[0] + decl_ini * (x - xs[0]), r)
    return float(r) if r.ndim == 0 else r


# --------------------------------------------------------------------------- #
# Cenários
# --------------------------------------------------------------------------- #
class AnaliseDeCenarios:
    """Avalia um ou vários modelos sob cenários ponderados por probabilidades.

    Parameters
    ----------
    modelos
        ``{nome: modelo}`` ou um único modelo.
    base
        Parâmetros do caso-base (compartilhados por todos os modelos).
    cenarios
        Lista de :class:`Cenario`, uma :class:`ArvoreDeCenarios` (expandida
        automaticamente) ou ``{nome_do_modelo: lista | árvore}`` quando cada
        projeto tem a sua própria árvore (caso Gama, Parte 2: CPV em P2 e
        preço em P4).
    taxa
        Taxa mínima de atratividade usada em VPL/RAU/IBC.
    """

    def __init__(self, modelos, base: Mapping, cenarios, taxa: float, taxa_reinvestimento: float | None = None):
        self.modelos = _normalizar_modelos(modelos)
        self.base = dict(base)
        self.taxa = taxa
        self.taxa_reinvestimento = taxa_reinvestimento
        self.arvores: dict[str, ArvoreDeCenarios] = {}
        self.cenarios_por_modelo: dict[str, list[Cenario]] = {}
        if isinstance(cenarios, Mapping) and not isinstance(cenarios, ArvoreDeCenarios):
            for nome in self.modelos:
                if nome not in cenarios:
                    raise KeyError(f"cenários não informados para o modelo '{nome}'")
                self.cenarios_por_modelo[nome] = self._expandir(nome, cenarios[nome])
        else:
            for nome in self.modelos:
                self.cenarios_por_modelo[nome] = self._expandir(nome, cenarios)
        self._resultado: pd.DataFrame | None = None
        self._fluxos: dict[tuple[str, str], FluxoDeCaixa] = {}

    def _expandir(self, nome, cen):
        if isinstance(cen, ArvoreDeCenarios):
            self.arvores[nome] = cen
            return cen.expandir()
        return list(cen)

    # compatibilidade: cenários "globais" = os do primeiro modelo
    @property
    def cenarios(self) -> list[Cenario]:
        return next(iter(self.cenarios_por_modelo.values()))

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
        return aplicar_cenario(self.base, cenario)

    def executar(self) -> pd.DataFrame:
        """Tabela longa: uma linha por (projeto, cenário) com todos os indicadores."""
        linhas = []
        for nome, modelo in self.modelos.items():
            for c in self.cenarios_por_modelo[nome]:
                fluxo = como_fluxo(modelo(self.parametros(c)))
                self._fluxos[(nome, c.nome)] = fluxo
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    ind = indicadores(fluxo, self.taxa, self.taxa_reinvestimento)
                linhas.append({"Projeto": nome, "Cenário": c.nome, "Hipótese": c.hipotese, "P(alt | hip)": c.probabilidade_condicional, "Probabilidade": c.probabilidade, **{k: ind[k] for k in METRICAS}})
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
        """Pivô projeto × cenário para uma métrica (NaN onde o cenário não se aplica ao projeto)."""
        df = self.resultado.pivot(index="Projeto", columns="Cenário", values=metrica)
        ordem = list(dict.fromkeys(c.nome for cs in self.cenarios_por_modelo.values() for c in cs))
        return df.reindex(columns=[c for c in ordem if c in df.columns]).reindex(list(self.modelos))

    @staticmethod
    def _pesos(sub: pd.DataFrame) -> np.ndarray:
        p = sub["Probabilidade"].to_numpy(dtype=float)
        if np.isnan(p).any():
            return np.full(len(sub), 1.0 / len(sub))
        return p / p.sum()

    def resumo(self, metrica: str = "vpl") -> pd.DataFrame:
        """Valor esperado, desvio padrão, CV, mínimo, máximo e P(métrica < 0) por projeto."""
        linhas = {}
        for proj, sub in self.resultado.groupby("Projeto", sort=False):
            v = sub[metrica].to_numpy(dtype=float)
            p = self._pesos(sub)
            e = float(np.sum(v * p))
            dp = float(np.sqrt(np.sum(p * (v - e) ** 2)))
            linhas[proj] = {"Valor esperado": e, "Desvio padrão": dp, "CV": dp / abs(e) if e else np.nan, "Mínimo": float(np.nanmin(v)), "Máximo": float(np.nanmax(v)), "P(< 0)": float(np.sum(p[v < 0]))}
        out = pd.DataFrame(linhas).T.reindex(list(self.modelos))
        out.index.name = "Projeto"
        return out

    def resumo_por_hipotese(self, metrica: str = "vpl") -> pd.DataFrame:
        """Para árvores: P(hipótese), valor esperado condicional, mínimo, máximo e contribuição ao valor esperado."""
        linhas = []
        for (proj, hip), sub in self.resultado.dropna(subset=["Hipótese"]).groupby(["Projeto", "Hipótese"], sort=False):
            p_hip = float(sub["Probabilidade"].sum())
            pc = sub["Probabilidade"].to_numpy(dtype=float) / p_hip
            v = sub[metrica].to_numpy(dtype=float)
            e_cond = float(np.sum(v * pc))
            linhas.append({"Projeto": proj, "Hipótese": hip, "P(hipótese)": p_hip, "Nº alternativas": len(sub), "E[métrica | hipótese]": e_cond, "Mínimo": float(v.min()), "Máximo": float(v.max()), "Contribuição P × E": p_hip * e_cond})
        return pd.DataFrame(linhas)

    def tabela_dispersao(self, projeto: str, metrica: str = "vpl") -> pd.DataFrame:
        """Tabela no formato das notas: Ei, pi, E, Ei − E, (Ei − E)², pi·(Ei − E)² e totais."""
        sub = self.resultado[self.resultado["Projeto"] == projeto]
        v = sub[metrica].to_numpy(dtype=float)
        p = self._pesos(sub)
        e = float(np.sum(v * p))
        df = pd.DataFrame({"Cenário": sub["Cenário"].values, "pi": p, "Ei": v, "pi · Ei": p * v, "Ei − E": v - e, "(Ei − E)²": (v - e) ** 2, "pi · (Ei − E)²": p * (v - e) ** 2})
        total = {"Cenário": "Total", "pi": p.sum(), "Ei": np.nan, "pi · Ei": e, "Ei − E": np.nan, "(Ei − E)²": np.nan, "pi · (Ei − E)²": float(np.sum(p * (v - e) ** 2))}
        df = pd.concat([df, pd.DataFrame([total])], ignore_index=True)
        df.attrs.update(valor_esperado=e, variancia=total["pi · (Ei − E)²"], desvio_padrao=float(np.sqrt(total["pi · (Ei − E)²"])), cv=float(np.sqrt(total["pi · (Ei − E)²"])) / abs(e) if e else np.nan)
        return df

    def taxas_ajustadas(self, curva: Sequence[tuple[float, float]], metrica: str = "vpl", extrapolar: bool = False) -> pd.Series:
        """Taxa ajustada ao risco de cada projeto a partir do CV da métrica."""
        cv = self.resumo(metrica)["CV"]
        return pd.Series(taxa_ajustada_ao_risco(cv.values, curva, extrapolar), index=cv.index, name="Taxa ajustada")

    def reavaliar(self, taxas: Mapping[str, float], cenario: str | None = None) -> pd.DataFrame:
        """Recalcula VPL/RAU/IBC com uma taxa específica por projeto, no caso-base (ou num cenário nomeado)."""
        linhas = {}
        for proj, modelo in self.modelos.items():
            if cenario is None:
                f = como_fluxo(modelo(self.base))
            else:
                cen = next(c for c in self.cenarios_por_modelo[proj] if c.nome == cenario)
                f = self._fluxos.get((proj, cenario)) or como_fluxo(modelo(self.parametros(cen)))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ind = indicadores(f, float(taxas[proj]), self.taxa_reinvestimento)
            linhas[proj] = {"Taxa": float(taxas[proj]), "VPL": ind["vpl"], "RAU": ind["rau"], "IBC": ind["ibc"], "TIR": ind["tir"]}
        out = pd.DataFrame(linhas).T
        out.index.name = "Projeto"
        return out

    def avaliacao_ajustada(self, curva: Sequence[tuple[float, float]], metrica: str = "vpl", extrapolar: bool = False) -> pd.DataFrame:
        """Resumo completo: E, DP, CV → taxa ajustada → VPL, RAU e IBC do caso-base à taxa ajustada, e a decisão."""
        res = self.resumo(metrica)
        taxas = self.taxas_ajustadas(curva, metrica, extrapolar)
        reav = self.reavaliar(taxas)
        base_tma = self.reavaliar({p: self.taxa for p in self.modelos})
        out = pd.DataFrame({
            "VPL à TMA": base_tma["VPL"], "Valor esperado": res["Valor esperado"], "Desvio padrão": res["Desvio padrão"], "CV": res["CV"], "P(< 0)": res["P(< 0)"],
            "Taxa ajustada": taxas, "VPL ajustado": reav["VPL"], "RAU ajustado": reav["RAU"], "IBC ajustado": reav["IBC"], "TIR": reav["TIR"],
        })
        out["Decisão"] = np.where(out["VPL ajustado"] > 0, "aceitar", "rejeitar")
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
