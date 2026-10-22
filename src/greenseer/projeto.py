"""Modelagem de projetos de investimento e construção do fluxo de caixa.

Segue a estrutura das notas de Orçamento de Capital:

.. code-block:: text

    Receitas
    (−) Custos / Despesas
    (=) EBITDA
    (−) Depreciação
    (=) Lucro Operacional
    (−) IR
    (=) NOPAT
    (+) Depreciação
    (=) Fluxo de Caixa Operacional
    (−) Investimento
    (−/+) Capital de giro
    (+) Valor residual (líquido de IR sobre o ganho de capital)
    (=) Fluxo de Caixa Global  (FCG)

e, quando há financiamento, o Fluxo de Caixa do Capital Próprio (FCCP):
``LAIR = EBIT − juros``, ``LL = LAIR − IR``, ``FCCP = LL + depreciação −
amortização + entrada do financiamento − investimento ...``.

Todas as linhas aceitam as especificações de :mod:`greenseer.series`, de modo
que um projeto pode ser descrito por quantidades × preços, percentuais da
receita, séries explícitas ou funções — escalares ou vetorizadas (Monte Carlo).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

from .cashflow import FluxoDeCaixa
from .series import Contexto, Spec, escalar, novo_contexto, resolver

ModoIR = Literal["compensar", "zero", "acumular"]


# --------------------------------------------------------------------------- #
# Componentes
# --------------------------------------------------------------------------- #
@dataclass
class Depreciacao:
    """Política de depreciação.

    * ``linear`` — (base − residual_contabil) / anos, durante ``anos`` períodos
      (padrão: vida útil) sobre a ``base`` (padrão: investimento total);
    * ``explicita`` — série informada (ex.: "depreciação anual de $30.000");
    * ``nenhuma``.
    """

    metodo: Literal["linear", "explicita", "nenhuma"] = "linear"
    anos: int | None = None
    valor_anual: Spec | None = None
    base: Spec | None = None
    residual_contabil: Spec = 0.0

    @classmethod
    def linear(cls, anos: int | None = None, base: Spec | None = None, residual_contabil: Spec = 0.0):
        return cls("linear", anos=anos, base=base, residual_contabil=residual_contabil)

    @classmethod
    def explicita(cls, valor_anual: Spec, base: Spec | None = None):
        return cls("explicita", valor_anual=valor_anual, base=base)

    @classmethod
    def nenhuma(cls):
        return cls("nenhuma")


@dataclass
class ValorResidual:
    """Valor residual ao final da vida útil.

    * ``mercado(valor, tributar_ganho=True)`` — venda do ativo; o ganho sobre o
      valor contábil é tributado à alíquota de IR do projeto (se o valor for
      menor que o contábil, a perda gera benefício fiscal);
    * ``perpetuidade(taxa, crescimento=0, base="nopat")`` — valor da empresa
      em marcha: ``NOPAT_n / (taxa − g)``, convenção das notas de Orçamento de
      Capital (série perpétua do NOPAT, pois na perpetuidade a depreciação é
      reinvestida para manter o ativo);
    * ``contabil()`` — valor contábil remanescente, sem tributação;
    * ``nenhum()``.
    """

    tipo: Literal["mercado", "perpetuidade", "contabil", "nenhum"] = "nenhum"
    valor: Spec = 0.0
    tributar_ganho: bool = True
    taxa: Spec | None = None
    crescimento: Spec = 0.0
    base: Literal["nopat", "fco"] = "nopat"

    @classmethod
    def mercado(cls, valor: Spec, tributar_ganho: bool = True):
        return cls("mercado", valor=valor, tributar_ganho=tributar_ganho)

    @classmethod
    def perpetuidade(cls, taxa: Spec, crescimento: Spec = 0.0, base: Literal["nopat", "fco"] = "nopat"):
        return cls("perpetuidade", taxa=taxa, crescimento=crescimento, base=base)

    @classmethod
    def contabil(cls):
        return cls("contabil")

    @classmethod
    def nenhum(cls):
        return cls("nenhum")


@dataclass
class Financiamento:
    """Financiamento de terceiros com planilha SAC, PRICE ou americano.

    ``carencia`` = períodos iniciais só com juros. A entrada do recurso ocorre
    no instante zero.
    """

    valor: Spec
    taxa: float
    prazo: int
    sistema: Literal["SAC", "PRICE", "AMERICANO"] = "SAC"
    carencia: int = 0

    def planilha(self, n: int) -> dict[str, np.ndarray]:
        """Arrays sobre 0..n: saldo_inicial, juros, amortizacao, prestacao, saldo_final."""
        valor = np.asarray(escalar(self.valor), dtype=float)
        i = float(self.taxa)
        fim = self.carencia + self.prazo
        saldo = valor.copy() if valor.ndim else np.asarray(valor, dtype=float)
        juros, amort = [np.zeros_like(saldo)], [np.zeros_like(saldo)]
        saldo_ini, saldo_fim = [np.zeros_like(saldo)], [saldo.copy()]
        pmt = None
        for t in range(1, n + 1):
            j = saldo * i
            if t <= self.carencia or t > fim:
                a = np.zeros_like(saldo)
                if t > fim:
                    j = np.zeros_like(saldo)
            elif self.sistema == "SAC":
                a = valor / self.prazo
            elif self.sistema == "PRICE":
                if pmt is None:
                    pmt = valor * i / (1 - (1 + i) ** (-self.prazo)) if i else valor / self.prazo
                a = pmt - j
            elif self.sistema == "AMERICANO":
                a = valor.copy() if t == fim else np.zeros_like(saldo)
            else:
                raise ValueError(f"sistema desconhecido: {self.sistema}")
            saldo_ini.append(saldo.copy())
            juros.append(j)
            amort.append(a)
            saldo = saldo - a
            saldo_fim.append(saldo.copy())

        def empilhar(lista):
            arr = np.stack(lista, axis=-1)
            if arr.ndim == 3:  # (S, 1, n+1) → (S, n+1)
                arr = arr.reshape(arr.shape[0], arr.shape[-1])
            return arr

        saida = {
            "saldo_inicial": empilhar(saldo_ini),
            "juros": empilhar(juros),
            "amortizacao": empilhar(amort),
            "saldo_final": empilhar(saldo_fim),
        }
        saida["prestacao"] = saida["juros"] + saida["amortizacao"]
        entrada = np.zeros_like(saida["juros"])
        entrada[..., 0] = valor.reshape(-1) if valor.ndim else float(valor)
        saida["entrada"] = entrada
        return saida

    def tabela(self, n: int) -> pd.DataFrame:
        p = self.planilha(n)
        if p["juros"].ndim == 2:
            p = {k: v.mean(axis=0) for k, v in p.items()}
        df = pd.DataFrame(
            {
                "Saldo inicial": p["saldo_inicial"],
                "Juros": p["juros"],
                "Amortização": p["amortizacao"],
                "Prestação": p["prestacao"],
                "Saldo final": p["saldo_final"],
            },
            index=pd.Index(range(n + 1), name="ano"),
        )
        return df


# --------------------------------------------------------------------------- #
# Projeto
# --------------------------------------------------------------------------- #
def _com_zero(x: np.ndarray) -> np.ndarray:
    """Prefixa a coluna do instante zero (zeros) a uma série 1..n."""
    z = np.zeros(x.shape[:-1] + (1,))
    return np.concatenate([z, x], axis=-1)


def _resolver_investimento(spec: Spec, n: int, ctx: Contexto) -> np.ndarray:
    """Investimento sobre 0..n. Número/coluna → tudo em t=0; dict → esparso; vetor n+1 → explícito."""
    if isinstance(spec, Mapping) or callable(spec):
        return resolver(spec, n, ctx, inicio=0)
    arr = np.asarray(spec, dtype=float)
    if arr.ndim == 0 or (arr.ndim == 1 and arr.shape[0] == 1):
        saida = np.zeros(n + 1)
        saida[0] = float(arr.reshape(-1)[0])
        return saida
    if arr.ndim == 1 and arr.shape[0] == n + 1:
        return arr.copy()
    if arr.ndim == 2 and arr.shape[1] == 1:
        saida = np.zeros((arr.shape[0], n + 1))
        saida[:, 0] = arr[:, 0]
        return saida
    if arr.ndim == 2 and arr.shape[1] == n + 1:
        return arr.copy()
    raise ValueError(
        "investimento deve ser número (t=0), coluna (S,1), dicionário {t: valor} "
        f"ou vetor de comprimento n+1={n + 1}; recebido forma {arr.shape}"
    )


def calcular_ir(lucro: np.ndarray, aliquota, modo: ModoIR = "compensar") -> np.ndarray:
    """Imposto de renda sobre o lucro tributável, período a período.

    * ``compensar`` — prejuízo gera IR negativo (benefício fiscal imediato,
      hipótese usual para projetos marginais de uma empresa lucrativa);
    * ``zero`` — não há IR em períodos de prejuízo, sem compensação futura;
    * ``acumular`` — prejuízos acumulados compensam lucros futuros.
    """
    aliq = escalar(aliquota)
    if modo == "compensar":
        return lucro * aliq
    if modo == "zero":
        return np.maximum(lucro, 0.0) * aliq
    if modo == "acumular":
        lucro2 = np.atleast_2d(lucro).astype(float)
        aliq2 = np.broadcast_to(np.asarray(aliq, dtype=float).reshape(-1, 1) if np.ndim(aliq) else aliq, (lucro2.shape[0], 1))
        saldo = np.zeros(lucro2.shape[0])
        ir = np.zeros_like(lucro2)
        for t in range(lucro2.shape[1]):
            l = lucro2[:, t]
            comp = np.minimum(saldo, np.maximum(l, 0.0))
            base = l - comp
            saldo = saldo - comp + np.maximum(-l, 0.0)
            ir[:, t] = np.maximum(base, 0.0) * aliq2[:, 0]
        return ir[0] if lucro.ndim == 1 else ir
    raise ValueError(f"modo de IR desconhecido: {modo}")


@dataclass
class Demonstrativo:
    """Resultado do cálculo de um projeto: todas as linhas sobre 0..n."""

    nome: str
    n: int
    linhas: dict[str, np.ndarray]
    itens_receita: dict[str, np.ndarray]
    itens_custo: dict[str, np.ndarray]
    fluxo_global: FluxoDeCaixa
    fluxo_proprio: FluxoDeCaixa | None = None
    financiamento: dict[str, np.ndarray] | None = None

    @property
    def vetorizado(self) -> bool:
        return self.fluxo_global.vetorizado

    def tabela(self, amostra: int | None = None, detalhar: bool = True, capital_proprio: bool = True) -> pd.DataFrame:
        """DataFrame no formato do curso (linhas = contas, colunas = anos).

        Para projetos vetorizados usa a média das amostras ou a amostra indicada.
        """

        def pegar(arr):
            arr = np.asarray(arr, dtype=float)
            if arr.ndim == 2:
                return arr[amostra] if amostra is not None else arr.mean(axis=0)
            return arr

        linhas: dict[str, np.ndarray] = {}
        L = self.linhas
        if detalhar and len(self.itens_receita) > 1:
            for k, v in self.itens_receita.items():
                linhas[f"    Receita · {k}"] = pegar(v)
        linhas["Receitas"] = pegar(L["receita"])
        if detalhar and len(self.itens_custo) > 1:
            for k, v in self.itens_custo.items():
                linhas[f"    Custo · {k}"] = pegar(v)
        linhas["(−) Custos e despesas"] = pegar(L["custo"])
        linhas["(=) EBITDA"] = pegar(L["ebitda"])
        linhas["(−) Depreciação"] = pegar(L["depreciacao"])
        linhas["(=) Lucro operacional (EBIT)"] = pegar(L["ebit"])
        linhas["(−) IR"] = pegar(L["ir"])
        linhas["(=) NOPAT"] = pegar(L["nopat"])
        linhas["(+) Depreciação "] = pegar(L["depreciacao"])
        linhas["(=) Fluxo de caixa operacional"] = pegar(L["fco"])
        linhas["(−) Investimento"] = pegar(L["investimento"])
        if np.any(L["capital_de_giro"] != 0):
            linhas["(−/+) Capital de giro"] = pegar(L["capital_de_giro"])
        if np.any(L["valor_residual"] != 0):
            linhas["(+) Valor residual"] = pegar(L["valor_residual"])
        if np.any(L["outros"] != 0):
            linhas["(+) Outros fluxos"] = pegar(L["outros"])
        linhas["(=) Fluxo de caixa global"] = pegar(L["fcg"])
        if capital_proprio and self.financiamento is not None:
            linhas["(+) Entrada do financiamento"] = pegar(L["entrada_financiamento"])
            linhas["(−) Juros"] = pegar(L["juros"])
            linhas["(=) LAIR"] = pegar(L["lair"])
            linhas["(−) IR (capital próprio)"] = pegar(L["ir_proprio"])
            linhas["(=) Lucro líquido"] = pegar(L["lucro_liquido"])
            linhas["(−) Amortização"] = pegar(L["amortizacao"])
            linhas["(=) Fluxo de caixa do capital próprio"] = pegar(L["fccp"])
        linhas["Memo: valor contábil do ativo"] = pegar(L["valor_contabil"])
        df = pd.DataFrame(linhas).T
        df.columns = pd.Index(range(self.n + 1), name="ano")
        df.index.name = self.nome
        return df


@dataclass
class Projeto:
    """Projeto de investimento descrito por suas premissas.

    Parameters
    ----------
    nome, vida_util
        Identificação e número de períodos (anos).
    investimento
        Desembolso(s) de capital (positivo = saída): número (em t=0),
        ``{t: valor}`` ou vetor sobre 0..n.
    receitas, custos
        Especificação única ou dicionário ``{nome: especificação}``. Custos
        positivos = desembolso. Custos podem referenciar a receita
        (``Percentual(0.6)``) ou itens anteriores.
    aliquota_ir
        Alíquota do imposto de renda.
    depreciacao
        :class:`Depreciacao`, série explícita, ``None`` (linear na vida útil).
    valor_residual
        :class:`ValorResidual` ou número (= valor de mercado tributado).
    capital_de_giro
        Nível de NCG necessário em cada período 1..n (investido no período
        anterior, recuperado em n). Número = nível constante.
    financiamento
        :class:`Financiamento` para obter também o fluxo do capital próprio.
    ir_sobre_prejuizo
        ``compensar`` (padrão), ``zero`` ou ``acumular``.
    outros_fluxos
        ``{nome: especificação sobre 0..n}`` de fluxos já líquidos de IR.
    """

    nome: str
    vida_util: int
    investimento: Spec = 0.0
    receitas: Spec | Mapping[str, Spec] = 0.0
    custos: Spec | Mapping[str, Spec] = 0.0
    aliquota_ir: Spec = 0.0
    depreciacao: Depreciacao | Spec | None = None
    valor_residual: ValorResidual | Spec | None = None
    capital_de_giro: Spec | None = None
    financiamento: Financiamento | None = None
    ir_sobre_prejuizo: ModoIR = "compensar"
    outros_fluxos: Mapping[str, Spec] | None = None
    descricao: str = ""
    _cache: Demonstrativo | None = field(default=None, repr=False, compare=False)

    # ------------------------------------------------------------------ #
    def _resolver_itens(self, spec, n, ctx, total_chave, prefixo) -> tuple[dict, np.ndarray]:
        itens = spec if isinstance(spec, Mapping) else {prefixo: spec}
        resolvidos: dict[str, np.ndarray] = {}
        total = np.zeros(n)
        ctx[total_chave] = total
        for k, v in itens.items():
            arr = resolver(v, n, ctx)
            resolvidos[k] = arr
            ctx[k] = arr
            total = total + arr
            ctx[total_chave] = total
        return resolvidos, total

    def calcular(self) -> Demonstrativo:
        """Executa o DRE e devolve o :class:`Demonstrativo` completo."""
        n = int(self.vida_util)
        ctx = novo_contexto(n)
        aliq = escalar(self.aliquota_ir)

        itens_rec, receita = self._resolver_itens(self.receitas, n, ctx, "receita", "Receita")
        itens_cus, custo = self._resolver_itens(self.custos, n, ctx, "custo", "Custo")
        ebitda = receita - custo
        ctx["ebitda"] = ebitda

        investimento = _resolver_investimento(self.investimento, n, ctx)
        inv_total = investimento.sum(axis=-1, keepdims=investimento.ndim == 2)

        # depreciação e valor contábil
        dep_spec = self.depreciacao
        if dep_spec is None:
            dep_spec = Depreciacao.linear()
        elif not isinstance(dep_spec, Depreciacao):
            dep_spec = Depreciacao.explicita(dep_spec)
        base = inv_total if dep_spec.base is None else escalar(dep_spec.base)
        if dep_spec.metodo == "linear":
            anos = n if dep_spec.anos is None else int(dep_spec.anos)
            anual = (base - escalar(dep_spec.residual_contabil)) / anos
            mascara = (np.arange(1, n + 1) <= anos).astype(float)
            depreciacao = anual * mascara
        elif dep_spec.metodo == "explicita":
            depreciacao = resolver(dep_spec.valor_anual, n, ctx)
        else:
            depreciacao = np.zeros(n)
        depreciacao = np.broadcast_to(depreciacao, np.broadcast_shapes(depreciacao.shape, ebitda.shape)).copy()
        valor_contabil = np.asarray(base, dtype=float) - np.cumsum(_com_zero(depreciacao), axis=-1)

        ebit = ebitda - depreciacao
        ir = calcular_ir(ebit, aliq, self.ir_sobre_prejuizo)
        nopat = ebit - ir
        fco = nopat + depreciacao
        ctx.update(ebit=ebit, nopat=nopat, fco=fco)

        # capital de giro
        if self.capital_de_giro is None:
            ncg = np.zeros(n + 1)
        else:
            nivel = resolver(self.capital_de_giro, n, ctx)
            ncg = np.zeros(nivel.shape[:-1] + (n + 1,))
            ncg[..., 0] = -nivel[..., 0]
            if n > 1:
                ncg[..., 1:n] = -(nivel[..., 1:] - nivel[..., :-1])
            ncg[..., n] = ncg[..., n] + nivel[..., -1]

        # valor residual
        vr_spec = self.valor_residual
        if vr_spec is None:
            vr_spec = ValorResidual.nenhum()
        elif not isinstance(vr_spec, ValorResidual):
            vr_spec = ValorResidual.mercado(vr_spec)
        vc_n = valor_contabil[..., -1]
        if vr_spec.tipo == "mercado":
            valor = np.asarray(escalar(vr_spec.valor), dtype=float)
            valor = valor.reshape(-1) if valor.ndim == 2 else valor
            ir_vr = (valor - vc_n) * aliq if vr_spec.tributar_ganho else 0.0
            ir_vr = np.asarray(ir_vr).reshape(-1) if np.ndim(ir_vr) == 2 else ir_vr
            vr_n = valor - ir_vr
        elif vr_spec.tipo == "perpetuidade":
            if vr_spec.taxa is None:
                raise ValueError("ValorResidual.perpetuidade exige a taxa de desconto")
            k = np.asarray(escalar(vr_spec.taxa), dtype=float)
            g = np.asarray(escalar(vr_spec.crescimento), dtype=float)
            k = k.reshape(-1) if k.ndim == 2 else k
            g = g.reshape(-1) if g.ndim == 2 else g
            base_vr = nopat[..., -1] if vr_spec.base == "nopat" else fco[..., -1]
            vr_n = base_vr / (k - g)
        elif vr_spec.tipo == "contabil":
            vr_n = vc_n
        else:
            vr_n = 0.0
        forma = np.broadcast_shapes(np.shape(vr_n), fco.shape[:-1] + (1,)) if np.ndim(vr_n) else fco.shape[:-1] + (1,)
        valor_residual = np.zeros(forma[:-1] + (n + 1,)) if forma else np.zeros(n + 1)
        valor_residual[..., n] = vr_n

        # outros fluxos (líquidos de IR), sobre 0..n
        outros = np.zeros(n + 1)
        if self.outros_fluxos:
            for k, v in self.outros_fluxos.items():
                outros = outros + resolver(v, n, ctx, inicio=0)

        fco0 = _com_zero(fco)
        fcg = fco0 - investimento + ncg + valor_residual + outros
        fcg = np.asarray(fcg, dtype=float)

        linhas = {
            "receita": _com_zero(receita),
            "custo": _com_zero(custo),
            "ebitda": _com_zero(ebitda),
            "depreciacao": _com_zero(depreciacao),
            "ebit": _com_zero(ebit),
            "ir": _com_zero(ir),
            "nopat": _com_zero(nopat),
            "fco": fco0,
            "investimento": -investimento + np.zeros_like(fcg),
            "capital_de_giro": ncg + np.zeros_like(fcg),
            "valor_residual": valor_residual + np.zeros_like(fcg),
            "outros": outros + np.zeros_like(fcg),
            "fcg": fcg,
            "valor_contabil": valor_contabil + np.zeros_like(fcg),
        }
        # garante forma consistente (S, n+1) quando qualquer linha é vetorizada
        if fcg.ndim == 2:
            linhas = {k: np.broadcast_to(v, fcg.shape).copy() for k, v in linhas.items()}

        fluxo_global = FluxoDeCaixa(fcg, self.nome)
        fluxo_proprio = None
        plan = None
        if self.financiamento is not None:
            plan = self.financiamento.planilha(n)
            juros = plan["juros"][..., 1:]
            lair = ebit - juros
            ir_cp = calcular_ir(lair, aliq, self.ir_sobre_prejuizo)
            ll = lair - ir_cp
            fccp = (
                _com_zero(ll + depreciacao)
                - plan["amortizacao"]
                + plan["entrada"]
                - investimento
                + ncg
                + valor_residual
                + outros
            )
            fccp = np.asarray(fccp, dtype=float)
            extras = {
                "entrada_financiamento": plan["entrada"],
                "juros": plan["juros"],
                "lair": _com_zero(lair),
                "ir_proprio": _com_zero(ir_cp),
                "lucro_liquido": _com_zero(ll),
                "amortizacao": plan["amortizacao"],
                "fccp": fccp,
            }
            if fccp.ndim == 2:
                extras = {k: np.broadcast_to(v, fccp.shape).copy() for k, v in extras.items()}
                linhas = {k: np.broadcast_to(v, fccp.shape).copy() for k, v in linhas.items()}
            linhas.update(extras)
            fluxo_proprio = FluxoDeCaixa(fccp, f"{self.nome} (capital próprio)")

        demo = Demonstrativo(
            nome=self.nome,
            n=n,
            linhas=linhas,
            itens_receita={k: _com_zero(v) for k, v in itens_rec.items()},
            itens_custo={k: _com_zero(v) for k, v in itens_cus.items()},
            fluxo_global=fluxo_global,
            fluxo_proprio=fluxo_proprio,
            financiamento=plan,
        )
        self._cache = demo
        return demo

    # ------------------------------------------------------------------ #
    def demonstrativo(self, **kw) -> pd.DataFrame:
        """Tabela DRE → fluxo de caixa (ver :meth:`Demonstrativo.tabela`)."""
        return self.calcular().tabela(**kw)

    def fluxo(self, tipo: Literal["global", "proprio"] = "global") -> FluxoDeCaixa:
        """Fluxo de caixa global (FCG) ou do capital próprio (FCCP)."""
        demo = self.calcular()
        if tipo == "global":
            return demo.fluxo_global
        if tipo == "proprio":
            if demo.fluxo_proprio is None:
                raise ValueError("projeto sem financiamento: não há fluxo do capital próprio")
            return demo.fluxo_proprio
        raise ValueError(f"tipo desconhecido: {tipo}")

    def indicadores(self, taxa, taxa_reinvestimento=None, tipo: str = "global") -> dict:
        from .metricas import indicadores

        return indicadores(self.fluxo(tipo), taxa, taxa_reinvestimento)

    def com(self, **alteracoes) -> "Projeto":
        """Cópia do projeto com campos alterados (útil para cenários)."""
        from dataclasses import replace

        return replace(self, _cache=None, **alteracoes)
