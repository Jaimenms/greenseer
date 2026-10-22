# %% [markdown]
"""
# 1 · Fundamentos: fluxo de caixa, indicadores e RAU

Este notebook apresenta a biblioteca **greenseer** reproduzindo os exemplos das notas do curso
(Orçamento de Capital, Análise de Investimentos, *Cuidado com a TIR*). Cada número calculado aqui pode
ser conferido com os slides — é a forma de validar a ferramenta antes de usá-la no estudo de caso.

**Roteiro**

1. Do DRE ao fluxo de caixa global (FCG) e ao fluxo do capital próprio (FCCP)
2. Indicadores: VPL, payback ajustado, IBC, TIR, TIRM
3. Vidas úteis diferentes: empilhamento até o MMC, fluxo marginal e **RAU**
4. *Cuidado com a TIR*: múltiplas raízes e a hipótese de reinvestimento
"""

# %%
import numpy as np
import pandas as pd

from greenseer import *
from greenseer import viz

viz.configurar_notebook()
pd.set_option("display.float_format", lambda x: f"{x:,.2f}")

# %% [markdown]
"""
## 1. Do DRE ao fluxo de caixa

A classe `Projeto` descreve um investimento pelas suas premissas (receitas, custos, depreciação, IR, valor
residual, capital de giro, financiamento) e monta a cadeia das notas de Orçamento de Capital:

```
Receitas → (−) Custos → EBITDA → (−) Depreciação → Lucro operacional → (−) IR → NOPAT
→ (+) Depreciação → FCO → (−) Investimento → (∓) Capital de giro → (+) Valor residual → FCG
```

**Exemplo dos slides** — projeto novo: custo $20.000 + instalação $8.000 (depreciação linear em 7 anos),
NCG $6.000, EBITDA $20.000/ano, IR 30%, valor residual de mercado $1.000 e financiamento de $15.000 a 10% a.a.
pelo sistema SAC em 5 anos.
"""

# %%
novo = Projeto(
    "Projeto novo", vida_util=7,
    investimento=28_000,                      # 20.000 + 8.000 de instalação, em t = 0
    receitas=20_000, custos=0.0,              # EBITDA projetado constante
    aliquota_ir=0.30,
    capital_de_giro=6_000,                    # investido em t = 0 e recuperado no fim
    valor_residual=ValorResidual.mercado(1_000),
    financiamento=Financiamento(15_000, taxa=0.10, prazo=5, sistema="SAC"),
)
estilizar_demonstrativo(novo.demonstrativo(), "Projeto novo — DRE → FCG → FCCP")

# %% [markdown]
"""
O FCG de $15.200 e o fluxo do capital próprio ($11.150, $11.360, …) coincidem com os slides. No diagrama,
cada ano é decomposto em setas empilhadas: investimento e capital de giro para baixo no ano 0; no ano 7, a
recuperação do capital de giro e o valor residual líquido de IR ($1.000 − 30% × ganho = $700) empilham sobre o
fluxo operacional.
"""

# %%
viz.grafico_fluxo(novo, titulo="Projeto novo — fluxo de caixa global por componente")

# %%
viz.grafico_fluxo(novo, tipo="proprio", titulo="Projeto novo — fluxo do capital próprio: entrada e amortização do financiamento")

# %% [markdown]
"""
A planilha do financiamento também fica disponível:
"""

# %%
estilizar(novo.financiamento.tabela(7), tipos={c: "moeda" for c in ["Saldo inicial", "Juros", "Amortização", "Prestação", "Saldo final"]}, destacar=[])

# %% [markdown]
"""
### Fluxo marginal (novo − anterior)

O projeto anterior (ativo já existente) tem EBITDA decrescente e depreciação de $1.000/ano. O que interessa na
decisão de substituição é o **fluxo marginal** — e o mesmo mecanismo serve para comparar dois projetos quaisquer.
"""

# %%
anterior = Projeto(
    "Projeto anterior", vida_util=7, investimento=0.0,
    receitas=[18_000, 16_000, 16_000, 15_000, 15_000, 13_000, 13_000], aliquota_ir=0.30, depreciacao=1_000,
)
marginal = fluxo_marginal(anterior, novo, metodo="estender")
pd.DataFrame({"Novo (FCG)": novo.fluxo().valores, "Anterior (FCG)": anterior.fluxo().valores, "Marginal": marginal.valores},
             index=pd.Index(range(8), name="ano")).T.style.format("{:,.0f}")

# %% [markdown]
"""
## 2. Indicadores de avaliação

Exemplo dos slides de Análise de Investimentos: projetos A e B, WACC = 10%.
"""

# %%
K = 0.10
A = FluxoDeCaixa([-60_000] + [20_000] * 6, "A")
B = FluxoDeCaixa([-72_000, 45_000, 22_000, 20_000, 13_000, 13_000, 13_000], "B")
estilizar(tabela_indicadores([A, B], K), titulo="Indicadores a 10% a.a. (slides: VPL 27.080 / 26.381, payback 3,75 / 2,86, IBC 1,45 / 1,37)")

# %% [markdown]
"""
As pequenas diferenças em relação aos slides vêm do arredondamento dos fatores de valor atual (3 casas) usado
em sala. O gráfico abaixo mostra o fluxo e o acumulado descontado, com o ponto de payback ajustado:
"""

# %%
viz.grafico_fluxo(A, taxa=K, acumulado=True, titulo="Projeto A — fluxo de caixa e payback ajustado a 10%")

# %% [markdown]
"""
Cada indicador também existe como função isolada (`vpl`, `tir`, `tirm`, `payback`, `ibc`, `rau`, `roia`,
`taxa_fisher`). Todas aceitam um `FluxoDeCaixa`, um `Projeto` ou um array:
"""

# %%
print(f"VPL(A) = {vpl(A, K):,.0f}   TIR(A) = {tir(A):.2%}   payback ajustado = {payback(A, K):.2f} anos   IBC = {ibc(A, K):.3f}")
print(f"VPL(B) = {vpl(B, K):,.0f}   TIR(B) = {tir(B):.2%}   payback ajustado = {payback(B, K):.2f} anos   IBC = {ibc(B, K):.3f}")

# %% [markdown]
"""
## 3. Vidas úteis diferentes: empilhamento, fluxo marginal e RAU

Segundo exemplo dos slides: A (4 anos) e B (6 anos). Comparar VPLs de horizontes diferentes é inadequado.
As notas mostram dois caminhos equivalentes:

* **Empilhar** cada projeto até o mínimo múltiplo comum das vidas (12 anos), montar o fluxo marginal B − A
  e calcular a TIR marginal;
* calcular o **RAU** (Resultado Anual Uniforme) = equivalente anual uniforme do fluxo − equivalente anual
  uniforme do investimento = VPL × (A/P, i, n).
"""

# %%
A2 = FluxoDeCaixa([-64_000] + [21_000] * 4, "A")
B2 = FluxoDeCaixa([-122_000, 29_000, 28_000, 27_000, 26_000, 25_000, 44_000], "B")
tab = tabela_indicadores([A2, B2], K)
estilizar(tab[["investimento", "vida", "vpl", "tir", "rau"]], titulo="A tem a maior TIR; B tem o maior RAU (slides: 810 vs. 1.351)")

# %%
viz.grafico_fluxos_empilhados([A2, B2], metodo="mmc", titulo="Empilhamento até o MMC das vidas (12 anos)")

# %%
marg = fluxo_marginal(A2, B2, metodo="mmc")
print("Fluxo marginal B − A (12 anos):", np.round(marg.valores / 1000, 1), "(em $ mil)")
print(f"TIR marginal = {tir(marg):.2%}  (slides: 11,33% por interpolação)  >  WACC 10%  ⇒  B é preferível")
print(f"VPL marginal a 10% = {vpl(marg, K):,.0f}   |   RAU(B) − RAU(A) = {rau(B2, K) - rau(A2, K):,.0f}   |   RAU do fluxo marginal = {rau(marg, K):,.0f}")

# %% [markdown]
"""
Repare que o **RAU é invariante ao empilhamento**: o RAU de A sobre 12 anos (três ciclos) é o mesmo RAU de um
ciclo de 4 anos. Por isso ele é o atalho correto para comparar vidas diferentes sob a hipótese de repetição —
e o VPL marginal do empilhamento é exatamente o valor presente da diferença de RAU.
"""

# %%
print(f"RAU(A, 1 ciclo) = {rau(A2, K):,.2f}   RAU(A, 3 ciclos) = {rau(A2.replicar(3), K):,.2f}")
print(f"RAU(B, 1 ciclo) = {rau(B2, K):,.2f}   RAU(B, 2 ciclos) = {rau(B2.replicar(2), K):,.2f}")

# %%
viz.grafico_perfil_vpl([A2, B2], K, taxas=np.linspace(0, 0.25, 101), metodo="mmc",
                       titulo="Perfil do VPL (fluxos empilhados a 12 anos) — a interseção é a taxa de Fisher")

# %% [markdown]
"""
A função `analise_marginal` faz esse procedimento para **todos os pares** de uma lista de projetos (ordenando
por investimento, defensor × desafiante) — é o que faltou na resolução de referência do estudo de caso e o que
veremos no próximo notebook.

## 4. *Cuidado com a TIR*

A TIR supõe que os fluxos intermediários são reinvestidos à própria TIR. A **TIRM** troca essa hipótese por
uma taxa de reinvestimento explícita (o custo de capital). Exemplo dos slides: investimento de $180.000 e
TIR de 32%; com reinvestimento a 25% a TIRM cai para 29,6%, e a 18% para 27,0%.
"""

# %%
T = FluxoDeCaixa([-180_000, 26_200, 61_200, 138_200, 103_200, 129_200], "T")
print(f"TIR = {tir(T):.2%}   TIRM (reinv. 25%) = {tirm(T, 0.25):.2%}   TIRM (reinv. 18%) = {tirm(T, 0.18):.2%}")

# %% [markdown]
"""
Fluxos não convencionais (mais de uma inversão de sinal) podem ter **várias TIR** — o caso típico é um fluxo
marginal entre projetos empilhados. `tir(..., todas=True)` devolve todas as raízes; a decisão deve usar o sinal
do VPL marginal à TMA, não "a" TIR:
"""

# %%
nc = FluxoDeCaixa([-100, 230, -132], "não convencional")
print("Raízes:", [f"{r:.1%}" for r in tir(nc, todas=True)], "| VPL a 15% =", round(vpl(nc, 0.15), 3))
viz.grafico_perfil_vpl([nc], 0.15, taxas=np.linspace(0, 0.35, 141), fisher=False, titulo="Duas TIR: o VPL só é positivo entre 10% e 20%")

# %% [markdown]
"""
## Resumo da API

| Tarefa | Função / classe |
|---|---|
| Descrever premissas e obter o fluxo | `Projeto(...)`, `.demonstrativo()`, `.fluxo("global"/"proprio")` |
| Séries flexíveis | `Crescimento`, `Variacoes`, `Percentual`, `Produto`, `Soma`, `Degraus` |
| Indicadores | `vpl`, `vp`, `tir`, `tirm`, `payback`, `ibc`, `rau`, `roia`, `taxa_fisher`, `tabela_indicadores` |
| Vidas diferentes / marginal | `equalizar`, `fluxo_marginal`, `analise_marginal`, `matriz_marginal`, `campeao_incremental` |
| Carteiras | `carteiras(fluxos, taxa, k=2, orcamento=...)` |
| Risco | `Cenario`, `AnaliseDeCenarios`, `sensibilidade`, `ponto_de_equilibrio`, `MonteCarlo` |
| Apresentação | `viz.grafico_*`, `viz.exportar`, `estilizar`, `estilizar_demonstrativo` |
"""
