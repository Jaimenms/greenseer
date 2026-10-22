"""greenseer — análise econômico-financeira e comparação de projetos de investimento.

Módulos
-------
series      especificações flexíveis de séries (crescimento, percentuais, produtos…)
projeto     modelagem DRE → fluxo de caixa global / do capital próprio
cashflow    fluxo de caixa (escalar ou vetorizado), empilhamento, soma
metricas    VPL, TIR, TIRM, payback, IBC, RAU, ROIA, taxa de Fisher
comparacao  análise marginal, equalização de vidas, carteiras, ranking
cenarios    cenários com probabilidades, sensibilidade, taxa ajustada ao risco
montecarlo  simulação vetorizada com distribuições e correlações
viz         gráficos Plotly para apresentação
relatorios  formatação de tabelas
"""

from . import cenarios, comparacao, metricas, montecarlo, projeto, relatorios, series, viz
from .cashflow import FluxoDeCaixa, como_fluxo, como_fluxos, mmc, somar
from .cenarios import AnaliseDeCenarios, ArvoreDeCenarios, Cenario, Hipotese, aplicar_cenario, ponto_de_equilibrio, sensibilidade, taxa_ajustada_ao_risco
from .comparacao import analise_marginal, campeao_incremental, carteiras, equalizar, fluxo_marginal, matriz_marginal, matriz_tir_marginal, ranking
from .metricas import ibc, indicadores, payback, rau, roia, tabela_indicadores, taxa_fisher, tir, tirm, vp, vpl, vpl_perfil
from .montecarlo import Discreta, Fixa, LogNormal, MonteCarlo, Normal, Pert, PorPeriodo, ResultadoMonteCarlo, Triangular, Uniforme
from .projeto import Demonstrativo, Depreciacao, Financiamento, Projeto, ValorResidual
from .relatorios import estilizar, estilizar_demonstrativo, estilizar_divergente, estilizar_gradiente, fmt_moeda, fmt_pct, tabela_fluxos
from .series import Crescimento, Degraus, Percentual, Produto, Soma, Variacoes, unidades_x_preco

__version__ = "0.1.0"

__all__ = [
    "FluxoDeCaixa", "como_fluxo", "como_fluxos", "mmc", "somar",
    "Projeto", "Demonstrativo", "Depreciacao", "ValorResidual", "Financiamento",
    "Crescimento", "Variacoes", "Percentual", "Produto", "Soma", "Degraus", "unidades_x_preco",
    "vp", "vpl", "vpl_perfil", "tir", "tirm", "payback", "ibc", "rau", "roia", "taxa_fisher", "indicadores", "tabela_indicadores",
    "equalizar", "fluxo_marginal", "analise_marginal", "matriz_marginal", "matriz_tir_marginal", "campeao_incremental", "carteiras", "ranking",
    "Cenario", "Hipotese", "ArvoreDeCenarios", "AnaliseDeCenarios", "aplicar_cenario", "sensibilidade", "ponto_de_equilibrio", "taxa_ajustada_ao_risco",
    "MonteCarlo", "ResultadoMonteCarlo", "Triangular", "Pert", "Normal", "LogNormal", "Uniforme", "Discreta", "Fixa", "PorPeriodo",
    "estilizar", "estilizar_demonstrativo", "estilizar_gradiente", "estilizar_divergente", "tabela_fluxos", "fmt_moeda", "fmt_pct",
    "series", "projeto", "metricas", "comparacao", "cenarios", "montecarlo", "viz", "relatorios",
]
