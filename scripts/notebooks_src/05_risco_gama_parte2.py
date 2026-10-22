# %% [markdown]
"""
# 5 · Caso Gama (Parte 2) — risco ponderado por probabilidades e taxa ajustada

Na Parte 1 a Empresa Gama escolheu os dois melhores projetos: **P2 e P4** (os de maior RAU, critério
coerente com a hipótese de repetição; P4 também é o campeão da análise marginal). A Parte 2 pede a avaliação
desses dois projetos **levando em conta o risco econômico**, com uma estrutura clássica das notas de Análise de
Risco:

* para cada projeto, uma **variável crítica** (CPV em P2, preço de venda em P4) com três hipóteses —
  pessimista (10%), mais provável (85%) e otimista (5%);
* dentro das hipóteses extremas, **quatro alternativas** com probabilidades condicionais (70 / 15 / 10 / 5% no
  ramo pessimista; 60 / 20 / 15 / 5% no otimista);
* uma **curva de indiferença** que converte o coeficiente de variação do VPL numa taxa de desconto ajustada ao
  risco.

Não há simulação aqui: o risco é **ponderado pelas probabilidades** da árvore (valor esperado, desvio padrão e
CV calculados exatamente, como nas notas), e a pergunta final é: *qual seria, nesta situação, a decisão final?*

| Projeto | Variável | Pessimista (10%) | Mais provável (85%) | Otimista (5%) |
|---|---|---|---|---|
| P2 | Custo dos produtos vendidos | +2% (70%), +5% (15%), +7% (10%), +10% (5%) | enunciado | −2% (60%), −5% (20%), −8% (15%), −10% (5%) |
| P4 | Preço de venda | −2% (70%), −3% (15%), −4% (10%), −5% (5%) | enunciado | +1% (60%), +2% (20%), +3% (15%), +4% (5%) |
"""

# %%
import numpy as np
import pandas as pd

from greenseer import *
from greenseer import viz
from caso_gama import BASE, K, MODELOS, CURVA_INDIFERENCA, arvore_p2, arvore_p4

viz.configurar_notebook()
pd.set_option("display.float_format", lambda x: f"{x:,.2f}")
SAIDA = "../outputs/05"

# ponto de partida: os dois projetos escolhidos, no caso-base
modelos = {"P2": MODELOS["P2"], "P4": MODELOS["P4"]}
estilizar(tabela_indicadores([modelos[p](BASE) for p in modelos], K)[["investimento", "vida", "vpl", "tir", "rau", "ibc", "payback_descontado"]], titulo="Caso-base (Parte 1): P2 e P4 à TMA de 18%")

# %% [markdown]
"""
## 1. Modelar a árvore de probabilidades

A biblioteca descreve a árvore com três objetos:

* `Hipotese(nome, probabilidade, alternativas=[...])` — ramo de primeiro nível;
* `Cenario(nome, probabilidade=p_condicional, variacoes={parametro: ±fração})` — alternativa dentro da hipótese.
  `variacoes` multiplica o valor-base por (1 + fração): "CPV 2% maior" vira `{"p2_cpv": +0.02}`;
* `ArvoreDeCenarios([...])` — valida que as probabilidades somam 1 em cada nível e **expande** a árvore nas
  folhas, com a probabilidade conjunta P(hipótese) × P(alternativa | hipótese).

O mesmo modelo parametrizado dos notebooks 3 e 4 (`caso_gama.py`) é reavaliado em cada folha. Abaixo, a árvore
do P2 escrita por extenso (a do P4 segue o mesmo padrão e está em `caso_gama.arvore_p4`).
"""

# %%
def alternativa(nome, variacao, prob):
    return Cenario(nome, probabilidade=prob, variacoes={"p2_cpv": variacao})

arvore_cpv = ArvoreDeCenarios(
    [
        Hipotese("Pessimista", 0.10, [alternativa("CPV +2%", 0.02, 0.70), alternativa("CPV +5%", 0.05, 0.15), alternativa("CPV +7%", 0.07, 0.10), alternativa("CPV +10%", 0.10, 0.05)]),
        Hipotese("Mais provável", 0.85),
        Hipotese("Otimista", 0.05, [alternativa("CPV −2%", -0.02, 0.60), alternativa("CPV −5%", -0.05, 0.20), alternativa("CPV −8%", -0.08, 0.15), alternativa("CPV −10%", -0.10, 0.05)]),
    ],
    nome="P2 · CPV", parametro="p2_cpv",
)
arvore_preco = arvore_p4()
tab = arvore_cpv.tabela()
estilizar(tab, tipos={"P(hipótese)": "pct", "P(alt | hip)": "pct", "P conjunta": "pct"}, destacar=[], titulo="P2 — folhas da árvore e probabilidades conjuntas (somam 100%)")

# %% [markdown]
"""
**Uma escolha de interpretação.** "Acréscimo de 2% no CPV" foi lido como variação *relativa* do percentual
(60% → 61,2% da receita). A leitura alternativa, 2 pontos percentuais (60% → 62%), é mais severa; a seção 6
mostra que ela não muda a decisão. Para o preço do P4 não há ambiguidade.

Os valores que a variável assume em cada folha, com as probabilidades, formam uma distribuição discreta:
"""

# %%
valores, probs = arvore_cpv.como_distribuicao(BASE, "p2_cpv")
pd.DataFrame({"CPV (% da receita)": valores, "Probabilidade": probs}, index=[c.nome for c in arvore_cpv.expandir()]).style.format({"CPV (% da receita)": "{:.2%}", "Probabilidade": "{:.2%}"})

# %% [markdown]
"""
## 2. VPL em cada folha

`AnaliseDeCenarios` aceita uma árvore por projeto (`{"P2": árvore_cpv, "P4": árvore_preco}`): cada modelo é
avaliado nas suas próprias folhas, todas à TMA de 18%.
"""

# %%
ac = AnaliseDeCenarios(modelos, BASE, {"P2": arvore_cpv, "P4": arvore_preco}, taxa=K)
resultado = ac.executar()
estilizar(resultado[["Projeto", "Hipótese", "Cenário", "Probabilidade", "vpl", "tir", "rau", "payback_descontado"]], tipos={"Probabilidade": "pct"}, destacar=[], titulo="VPL, TIR, RAU e payback em cada folha (TMA 18%)")

# %%
fig = viz.grafico_arvore(ac, "P2", titulo="P2 — árvore de cenários do CPV: probabilidade nas arestas, VPL nas folhas")
viz.exportar(fig, f"{SAIDA}/arvore_p2")
fig

# %%
fig = viz.grafico_arvore(ac, "P4", titulo="P4 — árvore de cenários do preço de venda")
viz.exportar(fig, f"{SAIDA}/arvore_p4")
fig

# %% [markdown]
"""
## 3. Valor esperado, desvio padrão e coeficiente de variação

Exatamente como nas notas: E = Σ pᵢ·VPLᵢ, DP = √Σ pᵢ·(VPLᵢ − E)², CV = DP / E. A tabela de dispersão reproduz
o quadro de cálculo do curso, folha a folha.
"""

# %%
td2 = ac.tabela_dispersao("P2")
estilizar(td2, tipos={"pi": "pct", "Ei": "moeda", "pi · Ei": "moeda", "Ei − E": "moeda", "(Ei − E)²": "num", "pi · (Ei − E)²": "num"}, destacar=[], titulo=f"P2 — E = {fmt_moeda(td2.attrs['valor_esperado'])} · DP = {fmt_moeda(td2.attrs['desvio_padrao'])} · CV = {td2.attrs['cv']:.2f}")

# %%
td4 = ac.tabela_dispersao("P4")
estilizar(td4, tipos={"pi": "pct", "Ei": "moeda", "pi · Ei": "moeda", "Ei − E": "moeda", "(Ei − E)²": "num", "pi · (Ei − E)²": "num"}, destacar=[], titulo=f"P4 — E = {fmt_moeda(td4.attrs['valor_esperado'])} · DP = {fmt_moeda(td4.attrs['desvio_padrao'])} · CV = {td4.attrs['cv']:.2f}")

# %%
resumo = ac.resumo("vpl")
estilizar(resumo, titulo="Resumo por projeto")

# %%
estilizar(ac.resumo_por_hipotese("vpl"), tipos={"P(hipótese)": "pct", "Nº alternativas": "int", "E[métrica | hipótese]": "moeda", "Mínimo": "moeda", "Máximo": "moeda", "Contribuição P × E": "moeda"}, destacar=[], titulo="Leitura por hipótese: valor esperado condicional e contribuição ao valor esperado")

# %%
fig = viz.grafico_distribuicao_discreta(ac, "P2")
viz.exportar(fig, f"{SAIDA}/distribuicao_p2")
fig

# %%
fig = viz.grafico_distribuicao_discreta(ac, "P4")
viz.exportar(fig, f"{SAIDA}/distribuicao_p4")
fig

# %% [markdown]
"""
**Leitura.** O ramo pessimista pesa mais que o otimista nos dois projetos (10% contra 5%), por isso o valor
esperado fica um pouco abaixo do VPL do caso-base. A diferença está na dispersão: uma queda de poucos por cento
no preço do P4 derruba o VPL muito mais do que o mesmo percentual no CPV do P2, porque o P4 opera com custo fixo
de $200.000 e margem estreita. O CV do P4 é, portanto, bem maior — e é o CV que a curva de indiferença traduz em
taxa.

## 4. Curva de indiferença → taxa ajustada ao risco

O gráfico do enunciado foi lido ponto a ponto: CV 0,1 → 18% (a própria TMA), 0,3 → 19%, 0,5 → 20%,
0,7 → 22%, 0,9 → 25%. Entre os pontos a taxa é interpolada linearmente; abaixo de 0,1 vale 18%.
"""

# %%
taxas = ac.taxas_ajustadas(CURVA_INDIFERENCA)
fig = viz.grafico_curva_indiferenca(CURVA_INDIFERENCA, resumo["CV"], extrapolar=False)
viz.exportar(fig, f"{SAIDA}/curva_indiferenca")
fig

# %% [markdown]
"""
## 5. Reavaliação à taxa ajustada e decisão

Cada projeto é descontado à sua taxa ajustada: VPL, RAU e IBC do caso-base recalculados. O projeto continua
aceitável se o VPL ajustado permanece positivo; entre os dois, o de maior RAU ajustado é o preferido.
"""

# %%
avaliacao = ac.avaliacao_ajustada(CURVA_INDIFERENCA)
estilizar(avaliacao, tipos={"VPL à TMA": "moeda", "VPL ajustado": "moeda", "RAU ajustado": "moeda", "IBC ajustado": "num", "Taxa ajustada": "pct", "P(< 0)": "pct"}, destacar=["Valor esperado", "VPL ajustado", "RAU ajustado"], titulo="Avaliação ajustada ao risco")

# %%
fluxos_base = [modelos[p](BASE).fluxo() for p in modelos]
fig = viz.grafico_perfil_vpl(fluxos_base, K, taxas=np.linspace(0.10, 0.30, 161), fisher=True, titulo="Perfil do VPL de P2 e P4 entre 10% e 30%: a TMA e as taxas ajustadas")
for p, tx in taxas.items():
    fig.add_vline(x=tx, line=dict(color=viz.cor_de(p), width=1.5, dash="dash"))
    fig.add_annotation(x=tx, y=0.96, yref="paper", text=f"{p}: {tx:.1%}".replace(".", ","), showarrow=False, xanchor="left", xshift=4, font=dict(size=11, color=viz.cor_de(p)))
viz.exportar(fig, f"{SAIDA}/perfil_ajustado")
fig

# %% [markdown]
"""
## 6. A decisão resiste às premissas?

Três verificações rápidas antes de fechar o parecer:

1. **Interpretação do CPV** (relativa × pontos percentuais);
2. **Peso do ramo pessimista**: e se a probabilidade da hipótese pessimista fosse maior que 10%?
3. **Extrapolação da curva** além de CV 0,9 (só importa se algum CV ultrapassar esse valor).
"""

# %%
ac_pp = AnaliseDeCenarios({"P2": modelos["P2"]}, BASE, arvore_p2(modo="pp"), taxa=K)
ac_pp.executar()
comp = pd.concat({"CPV relativo (60% → 61,2%)": ac.avaliacao_ajustada(CURVA_INDIFERENCA).loc[["P2"]], "CPV em p.p. (60% → 62%)": ac_pp.avaliacao_ajustada(CURVA_INDIFERENCA)}, names=["Interpretação"])
estilizar(comp[["Valor esperado", "Desvio padrão", "CV", "Taxa ajustada", "VPL ajustado", "RAU ajustado", "Decisão"]], tipos={"Taxa ajustada": "pct", "VPL ajustado": "moeda", "RAU ajustado": "moeda"}, destacar=[], titulo="P2: as duas leituras do 'acréscimo de 2% no CPV'")

# %%
def com_peso_pessimista(arvore, p_pess):
    """Mesma árvore com outra probabilidade para a hipótese pessimista (o resto é redistribuído para a mais provável)."""
    hips = []
    for h in arvore.hipoteses:
        if h.nome == "Pessimista":
            hips.append(Hipotese(h.nome, p_pess, h.alternativas))
        elif h.nome == "Mais provável":
            hips.append(Hipotese(h.nome, 1.0 - p_pess - 0.05, h.alternativas))
        else:
            hips.append(h)
    return ArvoreDeCenarios(hips, nome=arvore.nome, parametro=arvore.parametro)

linhas = []
for p_pess in [0.10, 0.20, 0.30, 0.40, 0.50]:
    aci = AnaliseDeCenarios(modelos, BASE, {"P2": com_peso_pessimista(arvore_cpv, p_pess), "P4": com_peso_pessimista(arvore_preco, p_pess)}, taxa=K)
    aci.executar()
    av = aci.avaliacao_ajustada(CURVA_INDIFERENCA, extrapolar=True)
    for proj in modelos:
        linhas.append({"P(pessimista)": p_pess, "Projeto": proj, "Valor esperado": av.loc[proj, "Valor esperado"], "CV": av.loc[proj, "CV"], "Taxa ajustada": av.loc[proj, "Taxa ajustada"], "VPL ajustado": av.loc[proj, "VPL ajustado"], "RAU ajustado": av.loc[proj, "RAU ajustado"], "Decisão": av.loc[proj, "Decisão"]})
sens = pd.DataFrame(linhas)
sens["P(pessimista)"] = sens["P(pessimista)"].map(lambda v: f"{v:.0%}")
estilizar(sens.pivot(index="P(pessimista)", columns="Projeto", values="VPL ajustado"), tipos={"P2": "moeda", "P4": "moeda"}, destacar=[], titulo="VPL ajustado ao risco quando o ramo pessimista pesa mais (curva extrapolada além de CV 0,9)")

# %%
estilizar(sens.pivot(index="P(pessimista)", columns="Projeto", values="Taxa ajustada"), tipos={"P2": "pct", "P4": "pct"}, destacar=[], titulo="Taxa ajustada correspondente")

# %%
fig = viz.grafico_cenarios(resultado, "vpl", resumo, titulo="VPL por folha da árvore: faixa, pontos e valor esperado (◆)")
viz.exportar(fig, f"{SAIDA}/faixas")
fig

# %% [markdown]
"""
## 7. Parecer

| | P2 | P4 |
|---|---|---|
| VPL à TMA (18%) | $ 30.609 | $ 42.350 |
| Valor esperado do VPL | $ 29.540 | $ 38.296 |
| Desvio padrão · CV | $ 12.612 · 0,43 | $ 22.307 · 0,58 |
| Probabilidade de VPL negativo | 3% | 10% |
| Taxa ajustada ao risco | 19,6% | 20,8% |
| VPL à taxa ajustada | **$ 23.230** | $ 20.581 |
| RAU à taxa ajustada | **$ 7.705** | $ 6.316 |

* **Os dois projetos continuam aceitáveis**: VPL positivo mesmo descontados às taxas ajustadas ao risco.
* **A ordem de preferência se inverte.** No caso-base P4 lidera em VPL e RAU; ajustado ao risco, **P2 passa à
  frente** (VPL ajustado $ 23,2 mil contra $ 20,6 mil; RAU $ 7,7 mil contra $ 6,3 mil). O preço de venda é a
  variável à qual P4 é mais sensível, e todo o ramo pessimista do P4 produz VPL negativo.
* **Robustez.** A leitura em pontos percentuais do CPV piora P2 (CV 0,73, taxa 22,4%), mas ele continua aceito.
  Já o P4 é frágil ao peso do ramo pessimista: se a hipótese pessimista tivesse 20% em vez de 10%, o VPL
  ajustado do P4 ficaria negativo; P2 só cai abaixo de zero com 50%.

**Decisão final:** manter os dois projetos escolhidos na Parte 1, **P2 e P4**, com P2 como prioridade e P4
condicionado a mitigação do risco de preço (contratos de venda, acompanhamento da hipótese pessimista). Se o
board exigir menos exposição, a alternativa é trocar P4 por P3 (segundo colocado da Parte 1 por VPL e terceiro
por RAU), aceitando menor valor esperado em troca de um CV bem menor — o notebook 3 compara essa troca com
cenários mais amplos.
"""
