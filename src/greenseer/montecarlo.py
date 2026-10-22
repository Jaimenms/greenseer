"""Simulação de Monte Carlo vetorizada.

As distribuições geram colunas ``(S, 1)``; o modelo é avaliado **uma única
vez** com todos os parâmetros vetorizados e o *broadcasting* produz ``S``
fluxos de caixa de uma só vez. Com isso dezenas de milhares de simulações de
vários projetos levam poucos segundos.

Amostragem por hipercubo latino (padrão) ou Monte Carlo simples; correlação
entre parâmetros via cópula gaussiana.
"""

from __future__ import annotations

import itertools
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import qmc

from .cashflow import FluxoDeCaixa, como_fluxo, somar
from .cenarios import Modelo, _normalizar_modelos
from .metricas import COLUNAS_INDICADORES, ibc, payback, rau, tir, tirm, vpl


# --------------------------------------------------------------------------- #
# Distribuições
# --------------------------------------------------------------------------- #
class Distribuicao:
    """Base das distribuições: implementa ``ppf(u)`` (inversa da acumulada)."""

    def ppf(self, u: np.ndarray) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def amostrar(self, rng: np.random.Generator, s: int) -> np.ndarray:
        return self.ppf(rng.random(s)).reshape(s, 1)

    @property
    def dimensao(self) -> int:
        return 1

    def media(self) -> float:
        u = (np.arange(20_001) + 0.5) / 20_001
        return float(np.mean(self.ppf(u)))


@dataclass
class Triangular(Distribuicao):
    minimo: float
    moda: float
    maximo: float

    def ppf(self, u):
        c = (self.moda - self.minimo) / (self.maximo - self.minimo)
        return stats.triang.ppf(u, c, loc=self.minimo, scale=self.maximo - self.minimo)


@dataclass
class Pert(Distribuicao):
    """PERT (beta modificada): mais massa na moda que a triangular."""

    minimo: float
    moda: float
    maximo: float
    lamb: float = 4.0

    def ppf(self, u):
        r = self.maximo - self.minimo
        a = 1 + self.lamb * (self.moda - self.minimo) / r
        b = 1 + self.lamb * (self.maximo - self.moda) / r
        return stats.beta.ppf(u, a, b, loc=self.minimo, scale=r)


@dataclass
class Normal(Distribuicao):
    media_: float
    dp: float

    def ppf(self, u):
        return stats.norm.ppf(u, loc=self.media_, scale=self.dp)


@dataclass
class LogNormal(Distribuicao):
    """Log-normal parametrizada pela média e desvio padrão da própria variável."""

    media_: float
    dp: float

    def ppf(self, u):
        s2 = np.log(1 + (self.dp / self.media_) ** 2)
        mu = np.log(self.media_) - s2 / 2
        return stats.lognorm.ppf(u, np.sqrt(s2), scale=np.exp(mu))


@dataclass
class Uniforme(Distribuicao):
    minimo: float
    maximo: float

    def ppf(self, u):
        return stats.uniform.ppf(u, loc=self.minimo, scale=self.maximo - self.minimo)


@dataclass
class Discreta(Distribuicao):
    valores: Sequence[float]
    probabilidades: Sequence[float]

    def ppf(self, u):
        p = np.asarray(self.probabilidades, dtype=float)
        acum = np.cumsum(p / p.sum())
        idx = np.searchsorted(acum, np.asarray(u), side="right")
        idx = np.clip(idx, 0, len(p) - 1)
        return np.asarray(self.valores, dtype=float)[idx]


@dataclass
class Fixa(Distribuicao):
    valor: float

    def ppf(self, u):
        return np.full_like(np.asarray(u, dtype=float), float(self.valor))


@dataclass
class PorPeriodo(Distribuicao):
    """Uma distribuição independente por período → amostra ``(S, n)``.

    Ex.: ``PorPeriodo([Normal(0.10, 0.03)] * 5)`` para o crescimento de cada ano.
    """

    distribuicoes: Sequence[Distribuicao]

    @property
    def dimensao(self) -> int:
        return len(self.distribuicoes)

    def ppf(self, u):
        u = np.asarray(u)
        cols = [d.ppf(u[:, j]) for j, d in enumerate(self.distribuicoes)]
        return np.stack(cols, axis=1)

    def amostrar(self, rng, s):
        return self.ppf(rng.random((s, self.dimensao)))


# --------------------------------------------------------------------------- #
# Simulação
# --------------------------------------------------------------------------- #
def _matriz_correlacao(nomes: Sequence[str], correlacoes: Mapping) -> np.ndarray:
    d = len(nomes)
    R = np.eye(d)
    pos = {n: i for i, n in enumerate(nomes)}
    for (a, b), rho in correlacoes.items():
        if a not in pos or b not in pos:
            raise KeyError(f"correlação entre '{a}' e '{b}': parâmetro desconhecido")
        R[pos[a], pos[b]] = R[pos[b], pos[a]] = float(rho)
    # garante matriz definida positiva (ajuste por autovalores)
    w, V = np.linalg.eigh(R)
    if w.min() < 1e-8:
        w = np.clip(w, 1e-8, None)
        R = V @ np.diag(w) @ V.T
        dd = np.sqrt(np.diag(R))
        R = R / np.outer(dd, dd)
    return R


class MonteCarlo:
    """Simulação de Monte Carlo de um ou vários modelos.

    Parameters
    ----------
    modelos
        ``{nome: modelo}`` ou modelo único. Todos recebem o mesmo dicionário
        de parâmetros amostrados (incertezas compartilhadas, como alíquota ou
        preço de mercado, afetam os projetos em conjunto).
    incertezas
        ``{parametro: Distribuicao}``.
    fixos
        Parâmetros determinísticos.
    n_simulacoes, seed
        Tamanho da amostra e semente.
    correlacoes
        ``{(param_a, param_b): rho}`` — cópula gaussiana.
    amostragem
        ``"lhs"`` (hipercubo latino) ou ``"mc"``.
    taxa, taxa_reinvestimento
        Taxas para VPL/RAU/IBC e TIRM.
    """

    def __init__(
        self,
        modelos,
        incertezas: Mapping[str, Distribuicao],
        fixos: Mapping | None = None,
        n_simulacoes: int = 10_000,
        seed: int | None = 42,
        correlacoes: Mapping[tuple[str, str], float] | None = None,
        amostragem: str = "lhs",
        taxa: float = 0.0,
        taxa_reinvestimento: float | None = None,
    ):
        self.modelos = _normalizar_modelos(modelos)
        self.incertezas = dict(incertezas)
        self.fixos = dict(fixos or {})
        self.n = int(n_simulacoes)
        self.seed = seed
        self.correlacoes = dict(correlacoes or {})
        self.amostragem = amostragem
        self.taxa = taxa
        self.taxa_reinvestimento = taxa_reinvestimento

    def amostrar(self) -> dict[str, np.ndarray]:
        """Gera as amostras dos parâmetros incertos (colunas ``(S, 1)`` ou ``(S, n)``)."""
        rng = np.random.default_rng(self.seed)
        nomes = list(self.incertezas)
        dims = [self.incertezas[k].dimensao for k in nomes]
        d = sum(dims)
        if self.amostragem == "lhs":
            u = qmc.LatinHypercube(d=d, seed=rng).random(self.n)
        elif self.amostragem == "mc":
            u = rng.random((self.n, d))
        else:
            raise ValueError("amostragem deve ser 'lhs' ou 'mc'")
        if self.correlacoes:
            escalares = [k for k, dim in zip(nomes, dims) if dim == 1]
            R = _matriz_correlacao(escalares, self.correlacoes)
            L = np.linalg.cholesky(R)
            idx = [sum(dims[:i]) for i, k in enumerate(nomes) if dims[i] == 1]
            z = stats.norm.ppf(np.clip(u[:, idx], 1e-12, 1 - 1e-12))
            z = z @ L.T
            u[:, idx] = stats.norm.cdf(z)
        amostras = {}
        pos = 0
        for k, dim in zip(nomes, dims):
            bloco = u[:, pos : pos + dim]
            pos += dim
            dist = self.incertezas[k]
            amostras[k] = dist.ppf(bloco[:, 0]).reshape(self.n, 1) if dim == 1 else dist.ppf(bloco)
        return amostras

    def executar(self) -> "ResultadoMonteCarlo":
        amostras = self.amostrar()
        params = {**self.fixos, **amostras}
        fluxos, saidas = {}, {}
        tr = self.taxa if self.taxa_reinvestimento is None else self.taxa_reinvestimento
        for nome, modelo in self.modelos.items():
            f = como_fluxo(modelo(params))
            if not f.vetorizado:
                f = FluxoDeCaixa(np.broadcast_to(f.valores, (self.n, f.valores.shape[-1])).copy(), f.nome)
            f.nome = nome
            fluxos[nome] = f
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                saidas[nome] = pd.DataFrame(
                    {
                        "vpl": vpl(f, self.taxa),
                        "tir": tir(f),
                        "tirm": tirm(f, tr),
                        "rau": rau(f, self.taxa),
                        "ibc": ibc(f, self.taxa),
                        "payback": payback(f),
                        "payback_descontado": payback(f, self.taxa),
                    }
                )
        entradas = {}
        for k, v in amostras.items():
            if v.shape[1] == 1:
                entradas[k] = v[:, 0]
            else:
                for j in range(v.shape[1]):
                    entradas[f"{k}[{j + 1}]"] = v[:, j]
        return ResultadoMonteCarlo(pd.DataFrame(entradas), saidas, fluxos, self.taxa, self.fixos)


@dataclass
class ResultadoMonteCarlo:
    """Amostras de entrada, indicadores por projeto e fluxos simulados."""

    entradas: pd.DataFrame
    saidas: dict[str, pd.DataFrame]
    fluxos: dict[str, FluxoDeCaixa]
    taxa: float
    fixos: dict = field(default_factory=dict)

    @property
    def projetos(self) -> list[str]:
        return list(self.saidas)

    @property
    def n(self) -> int:
        return len(self.entradas) if len(self.entradas.columns) else next(iter(self.saidas.values())).shape[0]

    def metrica(self, metrica: str = "vpl") -> pd.DataFrame:
        """Matriz S × projetos de uma métrica."""
        return pd.DataFrame({p: df[metrica].values for p, df in self.saidas.items()})

    # ------------------------------------------------------------------ #
    @staticmethod
    def _resumir(col: pd.Series | np.ndarray, limite: float, nivel: float) -> dict:
        v = np.asarray(col, dtype=float)
        v_ok = v[~np.isnan(v)]
        if v_ok.size == 0:
            return {}
        media = v_ok.mean()
        dp = v_ok.std(ddof=1) if v_ok.size > 1 else 0.0
        p_lo = np.percentile(v_ok, 100 * nivel)
        cauda = v_ok[v_ok <= p_lo]
        return {
            "Média": media,
            "Desvio padrão": dp,
            "CV": dp / abs(media) if media else np.nan,
            "Mínimo": v_ok.min(),
            f"P{int(100 * nivel)}": p_lo,
            "Mediana": np.median(v_ok),
            f"P{int(100 * (1 - nivel))}": np.percentile(v_ok, 100 * (1 - nivel)),
            "Máximo": v_ok.max(),
            f"P(< {limite:g})": float(np.mean(v_ok < limite)),
            f"CVaR {int(100 * nivel)}%": cauda.mean() if cauda.size else np.nan,
            "n válidas": int(v_ok.size),
        }

    def resumo(self, metrica: str = "vpl", limite: float | None = None, nivel: float = 0.05) -> pd.DataFrame:
        """Estatísticas por projeto. ``limite`` padrão: 0 (VPL/RAU), taxa (TIR/TIRM), 1 (IBC)."""
        if limite is None:
            limite = {"tir": self.taxa, "tirm": self.taxa, "ibc": 1.0}.get(metrica, 0.0)
        df = self.metrica(metrica)
        out = pd.DataFrame({p: self._resumir(df[p], limite, nivel) for p in df.columns}).T
        out.index.name = "Projeto"
        return out

    def probabilidade(self, metrica: str = "vpl", limite: float = 0.0, sentido: str = "<") -> pd.Series:
        df = self.metrica(metrica)
        op = {"<": np.less, "<=": np.less_equal, ">": np.greater, ">=": np.greater_equal}[sentido]
        return pd.Series({p: float(np.nanmean(op(df[p].values, limite))) for p in df.columns}, name=f"P({metrica} {sentido} {limite:g})")

    def prob_superar(self, metrica: str = "vpl") -> pd.DataFrame:
        """P(linha > coluna) para cada par de projetos."""
        df = self.metrica(metrica)
        M = pd.DataFrame(np.nan, index=df.columns, columns=df.columns)
        for a, b in itertools.permutations(df.columns, 2):
            M.loc[a, b] = float(np.nanmean(df[a].values > df[b].values))
        M.index.name = "Projeto"
        return M

    def prob_melhor(self, metrica: str = "vpl") -> pd.Series:
        """Probabilidade de cada projeto ser o melhor na métrica."""
        df = self.metrica(metrica)
        vencedor = df.idxmax(axis=1)
        return vencedor.value_counts(normalize=True).reindex(df.columns).fillna(0.0).rename("P(melhor)")

    def sensibilidade(self, projeto: str | None = None, metrica: str = "vpl", metodo: str = "spearman") -> pd.DataFrame:
        """Correlação (Spearman) entre cada entrada e a métrica — insumo do tornado."""
        alvo = self.metrica(metrica)
        projetos = [projeto] if projeto else self.projetos
        out = {}
        for p in projetos:
            y = alvo[p]
            corr = {}
            for c in self.entradas.columns:
                x = self.entradas[c]
                if x.nunique() <= 1 or y.nunique() <= 1:
                    corr[c] = 0.0
                else:
                    corr[c] = float(x.corr(y, method=metodo))
            out[p] = corr
        df = pd.DataFrame(out)
        df.index.name = "Parâmetro"
        if projeto:
            df = df.reindex(df[projeto].abs().sort_values(ascending=False).index)
        return df

    def carteiras(self, k: int | Sequence[int] | None = None, orcamento: float | None = None, nivel: float = 0.05, metrica: str = "vpl"):
        """Distribuição do VPL (soma) de cada combinação de projetos independentes.

        Devolve (tabela resumo ordenada por média, DataFrame S × carteira).
        """
        df = self.metrica(metrica)
        inv = {p: float(np.nanmean(self.fluxos[p].investimento_inicial)) for p in self.projetos}
        tamanhos = range(1, len(self.projetos) + 1) if k is None else ([k] if isinstance(k, int) else list(k))
        colunas = {}
        for tam in tamanhos:
            for combo in itertools.combinations(self.projetos, tam):
                total_inv = sum(inv[p] for p in combo)
                if orcamento is not None and total_inv > orcamento + 1e-9:
                    continue
                nome = " + ".join(combo)
                colunas[nome] = df[list(combo)].sum(axis=1).values
        dist = pd.DataFrame(colunas)
        limite = {"tir": self.taxa, "tirm": self.taxa, "ibc": 1.0}.get(metrica, 0.0)
        resumo = pd.DataFrame({c: self._resumir(dist[c], limite, nivel) for c in dist.columns}).T
        resumo.insert(0, "Investimento", [sum(inv[p] for p in c.split(" + ")) for c in resumo.index])
        resumo.insert(0, "Nº projetos", [len(c.split(" + ")) for c in resumo.index])
        melhor = dist.idxmax(axis=1).value_counts(normalize=True)
        resumo["P(melhor)"] = melhor.reindex(resumo.index).fillna(0.0).values
        resumo.index.name = "Carteira"
        return resumo.sort_values("Média", ascending=False), dist

    def leque(self, projeto: str, acumulado: bool = True, percentis=(5, 25, 50, 75, 95), descontado: bool = False) -> pd.DataFrame:
        """Percentis do fluxo (ou do fluxo acumulado) por período — gráfico de leque."""
        f = self.fluxos[projeto]
        v = f.valores
        if descontado:
            v = v * (1 + self.taxa) ** (-f.t)
        if acumulado:
            v = np.cumsum(v, axis=1)
        out = pd.DataFrame({f"P{p}": np.percentile(v, p, axis=0) for p in percentis}, index=pd.Index(f.t, name="ano"))
        out["Média"] = v.mean(axis=0)
        return out
