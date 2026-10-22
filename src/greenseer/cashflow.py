"""Fluxo de caixa como série temporal (período 0..n), escalar ou vetorizado.

Um :class:`FluxoDeCaixa` guarda os valores em ``valores`` com forma ``(n+1,)``
— um fluxo — ou ``(S, n+1)`` — ``S`` fluxos amostrados (Monte Carlo). Todas
as operações preservam a dimensão de amostras.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import lcm
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


@dataclass
class FluxoDeCaixa:
    """Série de fluxos de caixa, com o instante zero na posição 0.

    Convenção de sinal: entradas positivas, saídas (investimentos) negativas.
    """

    valores: np.ndarray
    nome: str = ""
    periodo: str = "ano"
    rotulos: dict = field(default_factory=dict, repr=False)

    def __post_init__(self):
        self.valores = np.asarray(self.valores, dtype=float)
        if self.valores.ndim not in (1, 2):
            raise ValueError("valores deve ter forma (n+1,) ou (S, n+1)")

    # ------------------------------------------------------------------ #
    # Propriedades básicas
    # ------------------------------------------------------------------ #
    @property
    def n(self) -> int:
        """Número de períodos (vida do fluxo)."""
        return self.valores.shape[-1] - 1

    @property
    def vetorizado(self) -> bool:
        return self.valores.ndim == 2

    @property
    def amostras(self) -> int | None:
        return self.valores.shape[0] if self.vetorizado else None

    @property
    def t(self) -> np.ndarray:
        return np.arange(self.n + 1)

    @property
    def investimento_inicial(self):
        """Desembolso no instante zero (número positivo)."""
        return -self.valores[..., 0]

    def __getitem__(self, t):
        return self.valores[..., t]

    def __len__(self):
        return self.valores.shape[-1]

    def __repr__(self):
        v = self.valores if not self.vetorizado else self.valores.mean(axis=0)
        corpo = ", ".join(f"{x:,.0f}" for x in v[: min(len(v), 8)])
        if len(v) > 8:
            corpo += ", …"
        extra = f", S={self.amostras}" if self.vetorizado else ""
        return f"FluxoDeCaixa({self.nome!r}, n={self.n}{extra}: [{corpo}])"

    # ------------------------------------------------------------------ #
    # Conversões
    # ------------------------------------------------------------------ #
    def amostra(self, i: int) -> "FluxoDeCaixa":
        """Extrai a i-ésima amostra como fluxo 1-D."""
        if not self.vetorizado:
            return self
        return FluxoDeCaixa(self.valores[i], self.nome, self.periodo)

    def media(self) -> "FluxoDeCaixa":
        """Fluxo médio entre as amostras (ou o próprio fluxo se 1-D)."""
        if not self.vetorizado:
            return self
        return FluxoDeCaixa(self.valores.mean(axis=0), self.nome, self.periodo)

    def to_series(self) -> pd.Series:
        v = self.media().valores
        return pd.Series(v, index=pd.Index(self.t, name=self.periodo), name=self.nome or "fluxo")

    def to_frame(self) -> pd.DataFrame:
        if self.vetorizado:
            return pd.DataFrame(self.valores, columns=pd.Index(self.t, name=self.periodo))
        return self.to_series().to_frame()

    def copiar(self, nome: str | None = None) -> "FluxoDeCaixa":
        return FluxoDeCaixa(self.valores.copy(), self.nome if nome is None else nome, self.periodo)

    # ------------------------------------------------------------------ #
    # Aritmética (com alinhamento por preenchimento de zeros)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _valores_de(outro):
        return outro.valores if isinstance(outro, FluxoDeCaixa) else np.asarray(outro, dtype=float)

    def _combinar(self, outro, op, nome):
        a = self.valores
        b = self._valores_de(outro)
        if b.ndim == 0:
            return FluxoDeCaixa(op(a, b), nome, self.periodo)
        a, b = alinhar(a, b)
        return FluxoDeCaixa(op(a, b), nome, self.periodo)

    def __add__(self, outro):
        nome = f"{self.nome} + {getattr(outro, 'nome', '')}".strip(" +")
        return self._combinar(outro, np.add, nome)

    __radd__ = __add__

    def __sub__(self, outro):
        nome = f"{self.nome} − {getattr(outro, 'nome', '')}".strip(" −")
        return self._combinar(outro, np.subtract, nome)

    def __neg__(self):
        return FluxoDeCaixa(-self.valores, f"−{self.nome}", self.periodo)

    def __mul__(self, k):
        return FluxoDeCaixa(self.valores * float(k), self.nome, self.periodo)

    __rmul__ = __mul__

    # ------------------------------------------------------------------ #
    # Transformações temporais
    # ------------------------------------------------------------------ #
    def deslocar(self, k: int) -> "FluxoDeCaixa":
        """Adia o fluxo ``k`` períodos (insere zeros no início)."""
        if k < 0:
            raise ValueError("k deve ser ≥ 0")
        z = np.zeros(self.valores.shape[:-1] + (k,))
        return FluxoDeCaixa(np.concatenate([z, self.valores], axis=-1), self.nome, self.periodo)

    def truncar(self, n: int) -> "FluxoDeCaixa":
        """Mantém apenas os períodos 0..n."""
        return FluxoDeCaixa(self.valores[..., : n + 1], self.nome, self.periodo)

    def estender(self, n: int) -> "FluxoDeCaixa":
        """Preenche com zeros até o período ``n``."""
        if n < self.n:
            return self.truncar(n)
        z = np.zeros(self.valores.shape[:-1] + (n - self.n,))
        return FluxoDeCaixa(np.concatenate([self.valores, z], axis=-1), self.nome, self.periodo)

    def replicar(self, ciclos: int) -> "FluxoDeCaixa":
        """Empilha ``ciclos`` repetições consecutivas do projeto.

        O instante final de um ciclo coincide com o instante zero do seguinte
        (os valores se somam), como no empilhamento até o mínimo múltiplo
        comum das vidas úteis.
        """
        if ciclos < 1:
            raise ValueError("ciclos deve ser ≥ 1")
        n = self.n
        saida = np.zeros(self.valores.shape[:-1] + (n * ciclos + 1,))
        for k in range(ciclos):
            saida[..., k * n : k * n + n + 1] += self.valores
        return FluxoDeCaixa(saida, self.nome, self.periodo)

    def replicar_ate(self, horizonte: int) -> "FluxoDeCaixa":
        """Replica até um horizonte múltiplo da vida (ex.: o MMC das vidas)."""
        if horizonte % self.n:
            raise ValueError(f"horizonte {horizonte} não é múltiplo da vida {self.n}")
        return self.replicar(horizonte // self.n)

    def acumulado(self) -> np.ndarray:
        return np.cumsum(self.valores, axis=-1)


# ---------------------------------------------------------------------- #
# Utilidades
# ---------------------------------------------------------------------- #
def alinhar(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Preenche com zeros para que ``a`` e ``b`` tenham o mesmo comprimento."""
    la, lb = a.shape[-1], b.shape[-1]
    L = max(la, lb)
    if la < L:
        a = np.concatenate([a, np.zeros(a.shape[:-1] + (L - la,))], axis=-1)
    if lb < L:
        b = np.concatenate([b, np.zeros(b.shape[:-1] + (L - lb,))], axis=-1)
    return a, b


def somar(fluxos: Iterable[FluxoDeCaixa], nome: str = "") -> FluxoDeCaixa:
    """Soma fluxos (projetos independentes) alinhando pelo instante zero."""
    fluxos = list(fluxos)
    if not fluxos:
        raise ValueError("lista de fluxos vazia")
    total = fluxos[0].copiar()
    for f in fluxos[1:]:
        total = total + f
    total.nome = nome or " + ".join(f.nome for f in fluxos)
    return total


def mmc(*vidas: int) -> int:
    """Mínimo múltiplo comum das vidas úteis."""
    r = 1
    for v in vidas:
        r = lcm(r, int(v))
    return r


def como_fluxo(obj, nome: str | None = None) -> FluxoDeCaixa:
    """Converte projeto, array ou fluxo em :class:`FluxoDeCaixa`."""
    if isinstance(obj, FluxoDeCaixa):
        return obj if nome is None else obj.copiar(nome)
    if hasattr(obj, "fluxo") and callable(obj.fluxo):
        f = obj.fluxo()
        return f if nome is None else f.copiar(nome)
    return FluxoDeCaixa(np.asarray(obj, dtype=float), nome or "")


def como_fluxos(objs, taxa=None) -> list[FluxoDeCaixa]:
    """Normaliza lista/dicionário de projetos ou fluxos em lista de fluxos nomeados."""
    if isinstance(objs, dict):
        return [como_fluxo(v, k) for k, v in objs.items()]
    saida = []
    for i, o in enumerate(objs):
        f = como_fluxo(o)
        if not f.nome:
            f = f.copiar(f"Fluxo {i + 1}")
        saida.append(f)
    return saida
