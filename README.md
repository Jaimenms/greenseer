# greenseer

Biblioteca Python para **análise econômico-financeira e comparação de projetos de investimento**, construída
sobre as notas do curso de Análise de Investimentos (Prof. Geraldo Sardinha Pinto Filho, FDC) e exemplificada
com o estudo de caso da **Empresa Gama**.

O que ela faz:

* monta o fluxo de caixa a partir das premissas (DRE → EBITDA → NOPAT → **FCG**; e **FCCP** quando há
  financiamento), com depreciação, IR sobre prejuízo, capital de giro, valor residual (mercado tributado,
  perpetuidade ou contábil) e planilhas SAC/PRICE/americano;
* calcula todos os indicadores: VPL, VA, TIR (todas as raízes), **TIRM**, payback simples e ajustado, IBC,
  **RAU** (Resultado Anual Uniforme), ROIA e taxa de Fisher;
* compara projetos de **vidas diferentes** (empilhamento até o MMC, extensão ou truncamento com valor
  econômico) e faz a **análise marginal de todos os pares** (defensor × desafiante), além da seleção incremental
  e da enumeração de **carteiras** sob restrição de quantidade ou orçamento;
* **cenários** com probabilidades — listas ou **árvores de hipóteses × alternativas** com probabilidades
  condicionais (`Hipotese`, `ArvoreDeCenarios`) — com valor esperado, desvio padrão, CV, tabela de dispersão no
  formato das notas, taxa ajustada ao risco por curva de indiferença e reavaliação; sensibilidade (tornado) e
  ponto de equilíbrio;
* **simulação de Monte Carlo vetorizada** (distribuições triangular, PERT, normal, log-normal, uniforme,
  discreta, por período; hipercubo latino; correlação por cópula gaussiana) para projetos e carteiras;
* **gráficos para o board** (Plotly, exportáveis em HTML interativo e PNG) e tabelas formatadas.

## Instalação

Requer Python ≥ 3.11 e [uv](https://docs.astral.sh/uv/).

```bash
uv sync --all-extras        # cria .venv com a biblioteca, Jupyter e kaleido
uv run pytest               # testes (validados contra os números das notas e da resolução de referência)
uv run jupyter lab          # abre os notebooks
```

A exportação PNG dos gráficos usa `kaleido`, que precisa de um Chrome/Chromium instalado; sem ele os
notebooks continuam funcionando (só a saída interativa).

## Notebooks (pasta `notebooks/`)

| Notebook | Conteúdo |
|---|---|
| `01_fundamentos.ipynb` | Reproduz os exemplos das notas: FCG/FCCP, indicadores, empilhamento, RAU, TIR marginal, TIRM e múltiplas TIR |
| `02_caso_gama.ipynb` | Caso Gama determinístico: modelagem dos 4 projetos, todos os métodos, vidas diferentes, análise marginal de todos os pares, carteiras de 2 projetos e parecer |
| `03_cenarios_gama.ipynb` | Cenários pessimista/provável/otimista com probabilidades, E/DP/CV, taxa ajustada ao risco, tornado, ponto de equilíbrio, carteiras por cenário |
| `04_monte_carlo_gama.ipynb` | Monte Carlo (20.000 simulações): distribuições de VPL/TIR/RAU, probabilidades de perda e de ser o melhor, tornado probabilístico, leques, carteiras e parecer para o board |
| `05_risco_gama_parte2.ipynb` | Parte 2 do caso: árvore de probabilidades (hipóteses × alternativas) para o CPV de P2 e o preço de P4, valor esperado/DP/CV ponderados pelas probabilidades (sem Monte Carlo), curva de indiferença CV → taxa ajustada ao risco, reavaliação e decisão final |

Os notebooks são gerados a partir de `scripts/notebooks_src/*.py` (formato *percent*, Python válido) com
`uv run python scripts/build_notebooks.py --execute`. Os gráficos usados nas conclusões são exportados para
`outputs/<nº>/` em HTML e PNG.

## Uso rápido

```python
from greenseer import *
from greenseer import viz

K = 0.18
p3 = Projeto(
    "P3", vida_util=7, investimento=225_000,
    receitas=Produto(Crescimento(60_000, 0.10, periodos=5), 6.0),          # unidades × preço
    custos={"CPV": Percentual([.67, .63, .63, .61, .60, .58, .58]),       # % da receita
            "Despesas": Percentual([.25, .23, .23, .23, .21, .17, .17])},
    depreciacao=Depreciacao.explicita(30_000),
    valor_residual=ValorResidual.mercado(42_000),                          # ganho tributado
    aliquota_ir=0.20,
)
p3.demonstrativo()                 # DRE → FCG, ano a ano
p3.fluxo()                         # FluxoDeCaixa
indicadores(p3, K)                 # {'vpl': 36440.2, 'tir': 0.222, 'rau': 9560.5, ...}

tabela_indicadores([p1, p2, p3, p4], K)          # todos os métodos
analise_marginal([p1, p2, p3, p4], K, "mmc")     # fluxo marginal de todos os pares (horizonte = MMC das vidas)
carteiras([p1, p2, p3, p4], K, k=2)              # combinações de 2 projetos
viz.grafico_perfil_vpl([p1, p2, p3, p4], K)      # perfil do VPL e taxas de Fisher
```

### Cenários e Monte Carlo

Descreva o projeto como **função de um dicionário de parâmetros**; o mesmo modelo serve para o caso-base,
para cenários e para a simulação (os parâmetros chegam vetorizados, como colunas `(S, 1)`):

```python
def modelo(p):
    return Projeto("P1", 5, investimento=200_000,
                   receitas=Produto(p["unidades"], p["preco"]),
                   custos=Produto(p["unidades"], p["custo_var"]), aliquota_ir=0.20)

base = dict(unidades=12_500, preco=12.0, custo_var=5.8)
ac = AnaliseDeCenarios({"P1": modelo}, base, [
    Cenario("Pessimista", dict(unidades=10_500, custo_var=6.3), 0.25),
    Cenario("Mais provável", {}, 0.50),
    Cenario("Otimista", dict(unidades=14_000, custo_var=5.5), 0.25),
], taxa=K)
ac.executar(); ac.resumo("vpl")                   # valor esperado, DP, CV, P(VPL<0)

mc = MonteCarlo({"P1": modelo}, {"unidades": Triangular(10_500, 12_500, 14_000),
                                 "preco": Normal(12, 0.5), "custo_var": Pert(5.5, 5.8, 6.3)},
                fixos=base, n_simulacoes=20_000, taxa=K)
res = mc.executar()
res.resumo("vpl"); res.prob_melhor(); res.carteiras(k=2); res.sensibilidade("P1")
viz.grafico_distribuicoes(res); viz.grafico_cdf(res); viz.grafico_leque(res, "P1")
```

## Estrutura do pacote (`src/greenseer/`)

| Módulo | Conteúdo |
|---|---|
| `series.py` | Especificações de séries: número, lista, dict esparso, `(S, n)`; `Crescimento`, `Variacoes`, `Percentual`, `Produto`, `Soma`, `Degraus` |
| `projeto.py` | `Projeto`, `Depreciacao`, `ValorResidual`, `Financiamento`, `Demonstrativo`, `calcular_ir` |
| `cashflow.py` | `FluxoDeCaixa` (1-D ou `(S, n+1)`), aritmética com alinhamento, `replicar`, `estender`, `truncar`, `somar`, `mmc` |
| `metricas.py` | `vp`, `vpl`, `vpl_perfil`, `tir`, `tirm`, `payback`, `ibc`, `rau`, `rau_componentes`, `roia`, `taxa_fisher`, `indicadores`, `tabela_indicadores` |
| `comparacao.py` | `equalizar`, `fluxo_marginal`, `analise_marginal`, `matriz_marginal`, `matriz_tir_marginal`, `campeao_incremental`, `carteiras`, `ranking` |
| `cenarios.py` | `Cenario` (com `variacoes`), `Hipotese`, `ArvoreDeCenarios`, `AnaliseDeCenarios` (uma árvore por projeto; `resumo`, `resumo_por_hipotese`, `tabela_dispersao`, `avaliacao_ajustada`), `sensibilidade`, `ponto_de_equilibrio`, `taxa_ajustada_ao_risco` |
| `montecarlo.py` | Distribuições, `MonteCarlo`, `ResultadoMonteCarlo` (resumo, probabilidades, carteiras, sensibilidade, leque) |
| `viz.py` | Tema e gráficos Plotly: fluxo em barras, diagrama de fluxo com setas (`grafico_diagrama_fluxo`, usado nos fluxos empilhados: ↑ entradas, ↓ investimentos, ambos nos anos de reinvestimento), perfil do VPL/Fisher, painel de indicadores, matriz marginal, cenários, tornado, distribuições, CDF, risco × retorno, carteiras, leque, probabilidade de ser o melhor, árvore de cenários, distribuição discreta, curva de indiferença; `exportar` |
| `relatorios.py` | `estilizar`, `estilizar_demonstrativo`, `tabela_fluxos`, formatação pt-BR |

## Convenções

* Sinais: entradas positivas, saídas negativas no fluxo; nas premissas, investimento e custos são informados
  como números positivos.
* Períodos: `t = 0` é o instante do investimento; séries operacionais cobrem `1..n`.
* Vetorização: um parâmetro incerto entra como coluna NumPy `(S, 1)`; uma trajetória incerta por período como
  `(S, n)`. Tudo o mais faz *broadcasting*.
* Decisão marginal: o desafiante vence quando o **VPL marginal à TMA é positivo**. A TIR marginal é informada,
  mas fluxos marginais podem ter várias TIR ou nenhum investimento marginal (projetos com o mesmo
  investimento) — `analise_marginal` sinaliza os dois casos.
* RAU = VPL × (A/P, i, n); é invariante ao empilhamento e, portanto, o critério correto para vidas
  diferentes sob a hipótese de repetição. Sem repetição, use o VPL (horizonte `estender`).

## Referências

Notas de aula do curso (pasta `references/`): *Orçamento de Capital*, *Análise de Investimentos*, *Análise de
Risco*, *Cuidado com a TIR* (Kelleher & MacCormack, HSM Management, 2005) e o enunciado *Paper Case AInvGama
Parte 1*. A resolução de referência (`Gama - Grupo 7.xlsx`) é usada como oráculo nos testes.
