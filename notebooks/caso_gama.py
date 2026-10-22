"""Modelos parametrizados do estudo de caso da Empresa Gama (Prof. Geraldo Sardinha, FDC).

Os quatro projetos são funções de um dicionário de parâmetros ``p`` — o mesmo modelo
serve para o caso-base, para os cenários e para a simulação de Monte Carlo (parâmetros
vetorizados). As definições são idênticas às mostradas no notebook 03.
"""

import numpy as np

from greenseer import Crescimento, Depreciacao, Percentual, Produto, Projeto, ValorResidual, Variacoes

K = 0.18   # remuneração exigida pelos acionistas (TMA)
IR = 0.20  # alíquota de imposto de renda

# Caso-base: exatamente o enunciado
BASE = dict(
    ir=IR,
    p1_unidades=12_500, p1_preco=12.0, p1_custo_var=5.80, p1_investimento=200_000,
    p2_fator_receita=1.0, p2_cpv=0.60, p2_investimento=200_000,
    p3_crescimento=0.10, p3_preco=6.0, p3_cpv_delta=0.0, p3_valor_residual=42_000, p3_investimento=225_000,
    p4_unidades=30_000, p4_preco=20.0, p4_custo_fixo=200_000, p4_custo_var=9.0, p4_investimento=300_000,
)

ROTULOS = {
    "ir": "Alíquota de IR",
    "p1_unidades": "P1 · unidades/ano", "p1_preco": "P1 · preço unitário", "p1_custo_var": "P1 · custo variável unit.", "p1_investimento": "P1 · investimento",
    "p2_fator_receita": "P2 · fator sobre as receitas", "p2_cpv": "P2 · CPV (% da receita)", "p2_investimento": "P2 · investimento",
    "p3_crescimento": "P3 · crescimento anual", "p3_preco": "P3 · preço unitário", "p3_cpv_delta": "P3 · CPV (± p.p.)", "p3_valor_residual": "P3 · valor residual", "p3_investimento": "P3 · investimento",
    "p4_unidades": "P4 · unidades no ano 1", "p4_preco": "P4 · preço no ano 1", "p4_custo_fixo": "P4 · custo fixo anual", "p4_custo_var": "P4 · custo variável unit.", "p4_investimento": "P4 · investimento",
}


def modelo_p1(p):
    """Aquisição de máquinas: +12.500 unidades a $12, custo variável $5,80, 5 anos, sem valor residual."""
    return Projeto(
        "P1", vida_util=5, investimento=p["p1_investimento"],
        receitas=Produto(p["p1_unidades"], p["p1_preco"]),
        custos={"Custo variável": Produto(p["p1_unidades"], p["p1_custo_var"])},
        aliquota_ir=p["ir"],
    )


def modelo_p2(p):
    """Acoplamento de equipamento: receitas marginais dadas, CPV 60% + adm 10% + vendas 9%, despesas gerais."""
    receitas = np.asarray([570_000, 550_000, 510_000, 423_000, 423_000]) * p["p2_fator_receita"]
    return Projeto(
        "P2", vida_util=5, investimento=p["p2_investimento"],
        receitas=receitas,
        custos={
            "CPV": Percentual(p["p2_cpv"]),
            "Despesas administrativas (10%)": Percentual(0.10),
            "Despesas de vendas (9%)": Percentual(0.09),
            "Despesas gerais": [17_200, 19_250, 23_350, 36_330, 36_330],
        },
        aliquota_ir=p["ir"],
    )


def modelo_p3(p):
    """Nova unidade: 60.000 un. crescendo 10% a.a. por 5 anos, preço $6, CPV e despesas em % da receita, 7 anos."""
    cpv = np.asarray([0.67, 0.63, 0.63, 0.61, 0.60, 0.58, 0.58]) + p["p3_cpv_delta"]
    return Projeto(
        "P3", vida_util=7, investimento=p["p3_investimento"],
        receitas=Produto(Crescimento(60_000, p["p3_crescimento"], periodos=5), p["p3_preco"]),
        custos={"CPV": Percentual(cpv), "Despesas operacionais": Percentual([0.25, 0.23, 0.23, 0.23, 0.21, 0.17, 0.17])},
        depreciacao=Depreciacao.explicita(30_000),
        valor_residual=ValorResidual.mercado(p["p3_valor_residual"]),  # ganho sobre o valor contábil tributado
        aliquota_ir=p["ir"],
    )


def modelo_p4(p):
    """Substituição de equipamentos: 30.000 un. com quedas de 2%, 2% e 3%; preço $20 com −5% (ano 4) e −4% (ano 6)."""
    unidades = Variacoes(p["p4_unidades"], {2: -0.02, 3: -0.02, 4: -0.03})
    preco = Variacoes(p["p4_preco"], {4: -0.05, 6: -0.04})
    return Projeto(
        "P4", vida_util=6, investimento=p["p4_investimento"],
        receitas=Produto(unidades, preco),
        custos={"Custo fixo": p["p4_custo_fixo"], "Custo variável": Produto(unidades, p["p4_custo_var"])},
        valor_residual=ValorResidual.perpetuidade(K),  # NOPAT do ano 6 ÷ K
        aliquota_ir=p["ir"],
    )


MODELOS = {"P1": modelo_p1, "P2": modelo_p2, "P3": modelo_p3, "P4": modelo_p4}


def projetos_base():
    """Os quatro projetos no caso-base."""
    return [m(BASE) for m in MODELOS.values()]
