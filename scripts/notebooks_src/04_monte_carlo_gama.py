# %% [markdown]
"""
# 4 · Caso Gama — simulação de Monte Carlo

Cenários discretos mostram três estados do mundo. A simulação de Monte Carlo sorteia milhares de combinações
das premissas incertas e produz a **distribuição** do VPL, da TIR e do RAU de cada projeto — e de cada carteira
de dois projetos. Com isso respondemos ao board em termos de probabilidade: *qual a chance de o projeto
destruir valor? qual a chance de P4 ser melhor que P3? qual carteira domina?*

A simulação é **vetorizada**: cada modelo é avaliado uma única vez com os parâmetros em forma de colunas
`(S, 1)`, e o *broadcasting* produz os `S` fluxos de caixa de uma vez (20.000 simulações × 4 projetos em
menos de um segundo).
"""

# %%
import numpy as np
import pandas as pd

from greenseer import *
from greenseer import viz
from caso_gama import BASE, K, MODELOS, ROTULOS   # modelos parametrizados (idênticos ao notebook 3)

viz.configurar_notebook()
pd.set_option("display.float_format", lambda x: f"{x:,.2f}")
SAIDA = "../outputs/04"

# %% [markdown]
"""
## 1. Distribuições das premissas

Para cada parâmetro incerto escolhemos uma distribuição com mínimo, moda e máximo compatíveis com os cenários
do notebook 3. `Pert` concentra mais massa perto da moda que a `Triangular`; `Normal` para preços de mercado;
`Uniforme` quando não há informação além dos limites. Volumes de P1 e P4 recebem correlação de 0,5 (mesmo
mercado).
"""

# %%
incertezas = {
    "p1_unidades":      Triangular(10_500, 12_500, 14_000),
    "p1_custo_var":     Pert(5.50, 5.80, 6.30),
    "p2_fator_receita": Triangular(0.88, 1.00, 1.08),
    "p2_cpv":           Triangular(0.58, 0.60, 0.64),
    "p3_crescimento":   Pert(0.04, 0.10, 0.13),
    "p3_cpv_delta":     Normal(0.0, 0.02),
    "p3_valor_residual": Uniforme(30_000, 50_000),
    "p4_unidades":      Triangular(27_000, 30_000, 32_000),
    "p4_preco":         Normal(20.0, 0.8),
    "p4_custo_fixo":    Pert(195_000, 200_000, 220_000),
    "p4_custo_var":     Triangular(8.50, 9.00, 9.80),
}
correlacoes = {("p1_unidades", "p4_unidades"): 0.5}

tabela_dist = pd.DataFrame({ROTULOS[k]: {"Distribuição": type(d).__name__, "Parâmetros": ", ".join(f"{v:g}" for v in d.__dict__.values()), "Média": d.media()} for k, d in incertezas.items()}).T
tabela_dist.style.format({"Média": "{:,.3f}"}).set_caption("Premissas incertas")

# %% [markdown]
"""
## 2. Simulação
"""

# %%
mc = MonteCarlo(MODELOS, incertezas, fixos=BASE, n_simulacoes=20_000, seed=2026, correlacoes=correlacoes, taxa=K)
res = mc.executar()
print(f"{res.n:,} simulações · correlação obtida entre volumes P1/P4: {res.entradas[['p1_unidades', 'p4_unidades']].corr(method='spearman').iloc[0, 1]:.2f}")
estilizar(res.resumo("vpl"), titulo="VPL simulado por projeto")

# %%
estilizar(res.resumo("tir"), tipos={c: "pct" for c in ["Média", "Mínimo", "P5", "Mediana", "P95", "Máximo", "CVaR 5%"]} | {"Desvio padrão": "pct"}, titulo="TIR simulada por projeto (P(< 18%) = probabilidade de não remunerar o acionista)")

# %%
estilizar(res.resumo("rau"), titulo="RAU simulado por projeto — comparável entre vidas diferentes")

# %%
fig = viz.grafico_distribuicoes(res, "vpl", titulo="Distribuição do VPL por projeto (20.000 simulações)")
viz.exportar(fig, f"{SAIDA}/distribuicao_vpl")
fig

# %%
fig = viz.grafico_cdf(res, "vpl", titulo="Probabilidade acumulada do VPL — leia P(VPL ≤ x) diretamente")
viz.exportar(fig, f"{SAIDA}/cdf_vpl")
fig

# %% [markdown]
"""
## 3. Probabilidades para a decisão
"""

# %%
probs = pd.DataFrame({
    "P(VPL < 0)": res.probabilidade("vpl", 0.0, "<"),
    "P(TIR < TMA)": res.probabilidade("tir", K, "<"),
    "P(RAU < 0)": res.probabilidade("rau", 0.0, "<"),
    "P(melhor VPL)": res.prob_melhor("vpl"),
    "P(melhor RAU)": res.prob_melhor("rau"),
})
estilizar(probs, tipos={c: "pct" for c in probs.columns}, destacar=["P(VPL < 0)", "P(melhor VPL)", "P(melhor RAU)"], titulo="Probabilidades por projeto")

# %%
estilizar(res.prob_superar("vpl"), tipos={c: "pct" for c in res.projetos}, destacar=[], titulo="P(linha tem VPL maior que coluna)")

# %%
fig = viz.grafico_prob_melhor(res.prob_melhor("vpl"), titulo="Probabilidade de cada projeto ter o maior VPL")
viz.exportar(fig, f"{SAIDA}/prob_melhor")
fig

# %% [markdown]
"""
## 4. O que move o resultado (tornado por correlação)

A correlação de Spearman entre cada premissa sorteada e o VPL mostra quais incertezas realmente importam —
é a versão probabilística do tornado do notebook 3.
"""

# %%
sens = res.sensibilidade(metrica="vpl").rename(index=ROTULOS)
estilizar_divergente(sens, amplitude=1.0, titulo="Correlação (Spearman) entre premissas e VPL")

# %%
fig = viz.grafico_leque(res, "P4", titulo="P4 — fluxo acumulado descontado: leque P5–P95")
viz.exportar(fig, f"{SAIDA}/leque_p4")
fig

# %%
fig = viz.grafico_leque(res, "P3", titulo="P3 — fluxo acumulado descontado: leque P5–P95")
viz.exportar(fig, f"{SAIDA}/leque_p3")
fig

# %% [markdown]
"""
## 5. Carteiras de dois projetos

Somando o VPL simulado de cada par (os projetos são independentes, mas as premissas podem ser correlacionadas)
obtemos a distribuição do VPL de cada carteira.
"""

# %%
cart, dist_cart = res.carteiras(k=2)
estilizar(cart, tipos={"Nº projetos": "int"}, titulo="Carteiras de dois projetos — VPL simulado")

# %%
fig = viz.grafico_carteiras(cart, coluna="Média", faixa=("P5", "P95"), titulo="Carteiras: VPL esperado e intervalo P5–P95")
viz.exportar(fig, f"{SAIDA}/carteiras")
fig

# %%
fig = viz.grafico_risco_retorno(cart, x="Desvio padrão", y="Média", titulo="Carteiras: retorno esperado × risco (desvio padrão do VPL)")
viz.exportar(fig, f"{SAIDA}/carteiras_risco_retorno")
fig

# %%
fig = viz.grafico_prob_melhor(cart["P(melhor)"], titulo="Probabilidade de cada carteira ter o maior VPL")
viz.exportar(fig, f"{SAIDA}/carteiras_prob_melhor")
fig

# %% [markdown]
"""
## 6. Parecer para o board

| Pergunta | Caso-base (notebook 2) | Com risco (cenários + Monte Carlo) |
|---|---|---|
| Melhor projeto isolado | P4 (maior VPL e RAU) | **P3**: maior VPL esperado, ~13% de chance de VPL negativo; P4 tem ~50% |
| Dois melhores | P4 + P3 (VPL) ou P4 + P2 (RAU) | **P2 + P3**: maior VPL esperado, maior chance de ser a melhor carteira (26%) e só 7% de chance de perda |
| Projeto a evitar | P1 (menor VPL) | P4, a menos que preço e volume sejam contratados: ponto de equilíbrio a 1,8% de queda de preço |

**Mensagens-chave**

1. O ranking determinístico esconde que P4 é uma aposta alavancada: custo fixo alto e margem fina fazem o
   VPL variar de −$400 mil a +$500 mil, com quase 50% de chance de ser negativo.
2. P3 e P2 têm distribuições concentradas; juntos formam a carteira de maior VPL esperado, e P1 + P3 é a
   alternativa de menor risco com retorno quase igual.
3. Se o board quiser P4, a decisão deve vir acompanhada de mitigação (contratos de preço/volume, redução de
   custo fixo) — a simulação permite recalcular a probabilidade de perda com essas medidas.

Todos os gráficos deste notebook foram exportados para `outputs/04/` em HTML interativo e PNG para a apresentação.
"""
