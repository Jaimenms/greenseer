import numpy as np
import pytest

from greenseer import Crescimento, Depreciacao, Percentual, Produto, Projeto, ValorResidual, Variacoes

K = 0.18
IR = 0.20


@pytest.fixture(scope="session")
def taxa():
    return K


@pytest.fixture(scope="session")
def projetos_gama():
    """Os quatro projetos do caso Gama (Prof. Sardinha), como na resolução do Grupo 7."""
    p1 = Projeto("P1", 5, investimento=200_000, receitas=Produto(12_500, 12.0), custos=Produto(12_500, 5.80), aliquota_ir=IR)
    p2 = Projeto(
        "P2", 5, investimento=200_000,
        receitas=[570_000, 550_000, 510_000, 423_000, 423_000],
        custos={"CPV": Percentual(0.60), "Adm": Percentual(0.10), "Vendas": Percentual(0.09), "Gerais": [17_200, 19_250, 23_350, 36_330, 36_330]},
        aliquota_ir=IR,
    )
    p3 = Projeto(
        "P3", 7, investimento=225_000,
        receitas=Produto(Crescimento(60_000, 0.10, periodos=5), 6.0),
        custos={"CPV": Percentual([0.67, 0.63, 0.63, 0.61, 0.60, 0.58, 0.58]), "Opex": Percentual([0.25, 0.23, 0.23, 0.23, 0.21, 0.17, 0.17])},
        depreciacao=Depreciacao.explicita(30_000), valor_residual=ValorResidual.mercado(42_000), aliquota_ir=IR,
    )
    un = Variacoes(30_000, {2: -0.02, 3: -0.02, 4: -0.03})
    p4 = Projeto(
        "P4", 6, investimento=300_000,
        receitas=Produto(un, Variacoes(20.0, {4: -0.05, 6: -0.04})),
        custos={"Fixo": 200_000, "Variável": Produto(un, 9.0)},
        valor_residual=ValorResidual.perpetuidade(K), aliquota_ir=IR,
    )
    return [p1, p2, p3, p4]
