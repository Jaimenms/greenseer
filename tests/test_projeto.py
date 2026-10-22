"""Validação do construtor de fluxo de caixa (DRE) contra as notas e a planilha do Grupo 7."""
import numpy as np
import pytest

from greenseer import Financiamento, Projeto, ValorResidual, indicadores, tabela_indicadores
from greenseer.projeto import calcular_ir


def test_orcamento_de_capital_fcg_e_fccp():
    """Exemplo das notas: EBITDA 20.000, invest. 28.000 (7 anos), IR 30%, SAC 15.000 a 10% em 5 anos."""
    novo = Projeto("Novo", 7, investimento=28_000, receitas=20_000, custos=0.0, aliquota_ir=0.30,
                   capital_de_giro=6_000, valor_residual=ValorResidual.mercado(1_000),
                   financiamento=Financiamento(15_000, 0.10, 5, "SAC"))
    d = novo.calcular()
    fcg = d.fluxo_global.valores
    assert np.allclose(fcg[1:7], 15_200)
    assert fcg[0] == pytest.approx(-34_000)             # investimento + NCG
    assert fcg[7] == pytest.approx(15_200 + 6_000 + 700)  # recupera NCG; VR 1.000 tributado a 30%
    fccp = d.fluxo_proprio.valores
    assert np.allclose(fccp[1:6], [11_150, 11_360, 11_570, 11_780, 11_990])
    assert fccp[6] == pytest.approx(15_200)
    assert fccp[0] == pytest.approx(-34_000 + 15_000)
    plan = novo.financiamento.tabela(7)
    assert list(plan["Juros"][1:6]) == [1_500, 1_200, 900, 600, 300]
    assert list(plan["Prestação"][1:6]) == [4_500, 4_200, 3_900, 3_600, 3_300]


def test_projeto_anterior_e_marginal():
    """EBITDA do projeto anterior → FCG 12.900, 11.500, ...; marginal 2.300, 3.700, ..."""
    ant = Projeto("Anterior", 7, investimento=0.0, receitas=[18_000, 16_000, 16_000, 15_000, 15_000, 13_000, 13_000],
                  aliquota_ir=0.30, depreciacao=1_000)
    f = ant.fluxo().valores
    assert np.allclose(f[1:], [12_900, 11_500, 11_500, 10_800, 10_800, 9_400, 9_400])
    marginal = np.full(7, 15_200) - f[1:]
    assert np.allclose(marginal, [2_300, 3_700, 3_700, 4_400, 4_400, 5_800, 5_800])


def test_caso_gama_grupo7(projetos_gama, taxa):
    p1, p2, p3, p4 = projetos_gama
    assert np.allclose(p1.fluxo().valores, [-200_000] + [70_000] * 5)
    assert np.allclose(p2.fluxo().valores, [-200_000, 90_000, 85_000, 75_000, 50_000, 50_000])
    f3 = p3.fluxo().valores
    assert np.allclose(f3, [-225_000, 29_040, 50_352, 54_787.2, 67_332.48, 86_115.552, 121_956.72, 158_556.72], rtol=1e-6)
    f4 = p4.fluxo().valores
    assert np.allclose(f4, [-300_000, 114_000, 108_720, 103_545.6, 73_581.12, 73_581.12, 93_194.2598], rtol=1e-6)
    tab = tabela_indicadores(projetos_gama, taxa)
    assert tab.loc["P1", "vpl"] == pytest.approx(18_901.97, abs=0.01)
    assert tab.loc["P2", "vpl"] == pytest.approx(30_609.08, abs=0.01)
    assert tab.loc["P3", "vpl"] == pytest.approx(36_440.16, abs=0.01)
    assert tab.loc["P4", "vpl"] == pytest.approx(42_349.63, abs=0.01)
    assert tab.loc["P4", "rau"] == pytest.approx(12_108.19, abs=0.01)
    assert tab.loc["P3", "rau"] == pytest.approx(9_560.51, abs=0.01)
    assert tab.loc["P2", "tir"] == pytest.approx(0.25456, abs=1e-5)
    assert tab.loc["P4", "tir"] == pytest.approx(0.23816, abs=1e-5)
    assert tab.loc["P1", "payback_descontado"] == pytest.approx(4.3822, abs=1e-3)
    assert tab.loc["P3", "payback_descontado"] == pytest.approx(6.2679, abs=1e-3)
    assert tab.loc["P3", "ibc"] == pytest.approx(1.16196, abs=1e-4)


def test_valor_residual_variantes():
    base = dict(vida_util=4, investimento=100_000, receitas=60_000, custos=20_000, aliquota_ir=0.25)
    d_merc = Projeto("m", valor_residual=ValorResidual.mercado(30_000), **base).calcular()
    assert d_merc.linhas["valor_residual"][-1] == pytest.approx(30_000 - 0.25 * 30_000)  # contábil = 0
    d_sem = Projeto("s", valor_residual=ValorResidual.mercado(30_000, tributar_ganho=False), **base).calcular()
    assert d_sem.linhas["valor_residual"][-1] == pytest.approx(30_000)
    d_cont = Projeto("c", valor_residual=ValorResidual.contabil(), depreciacao=10_000, **base).calcular()
    assert d_cont.linhas["valor_residual"][-1] == pytest.approx(60_000)
    d_perp = Projeto("p", valor_residual=ValorResidual.perpetuidade(0.10, crescimento=0.02), **base).calcular()
    nopat = d_perp.linhas["nopat"][-1]
    assert d_perp.linhas["valor_residual"][-1] == pytest.approx(nopat / (0.10 - 0.02))


def test_capital_de_giro_por_nivel():
    p = Projeto("g", 3, investimento=10_000, receitas=[100, 120, 150], aliquota_ir=0.0, capital_de_giro=[10, 12, 15])
    ncg = p.calcular().linhas["capital_de_giro"]
    assert np.allclose(ncg, [-10, -2, -3, 15])


def test_ir_sobre_prejuizo():
    lucro = np.array([-100.0, 50.0, 80.0])
    assert np.allclose(calcular_ir(lucro, 0.2, "compensar"), [-20, 10, 16])
    assert np.allclose(calcular_ir(lucro, 0.2, "zero"), [0, 10, 16])
    assert np.allclose(calcular_ir(lucro, 0.2, "acumular"), [0, 0, 6])


def test_vetorizacao_broadcast():
    S = 3
    un = np.array([[10_000.0], [12_000.0], [8_000.0]])
    p = Projeto("v", 5, investimento=np.full((S, 1), 100_000.0), receitas=un * 12.0, custos=un * 5.0, aliquota_ir=0.2)
    f = p.fluxo()
    assert f.vetorizado and f.valores.shape == (S, 6)
    esperado = Projeto("e", 5, investimento=100_000, receitas=12_000 * 12.0, custos=12_000 * 5.0, aliquota_ir=0.2).fluxo().valores
    assert np.allclose(f.valores[1], esperado)
    ind = indicadores(f, 0.18)
    assert ind["vpl"].shape == (S,)


def test_demonstrativo_tabela(projetos_gama):
    df = projetos_gama[2].demonstrativo()
    assert "(=) Fluxo de caixa global" in df.index
    assert df.loc["(=) Fluxo de caixa global", 7] == pytest.approx(158_556.72)
    assert df.loc["Memo: valor contábil do ativo", 7] == pytest.approx(15_000)
