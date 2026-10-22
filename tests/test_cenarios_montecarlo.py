import numpy as np
import pytest

from greenseer import (AnaliseDeCenarios, Cenario, Discreta, MonteCarlo, Normal, Pert, PorPeriodo, Produto, Projeto, Triangular, Uniforme,
                       ponto_de_equilibrio, sensibilidade, taxa_ajustada_ao_risco, vpl)
from greenseer.cenarios import coeficiente_de_variacao, desvio_padrao, valor_esperado


def modelo(p):
    return Projeto("A", 15, investimento=10_000, receitas=p["fluxo"], custos=0.0, aliquota_ir=0.0, depreciacao=0.0)


def test_cenarios_notas_de_risco():
    """Notas de Análise de Risco: fluxos 1.500/2.000/2.500 a 10% por 15 anos, probabilidades 0,25/0,5/0,25."""
    cen = [Cenario("Pessimista", {"fluxo": 1_500}, 0.25), Cenario("Mais provável", {"fluxo": 2_000}, 0.50), Cenario("Otimista", {"fluxo": 2_500}, 0.25)]
    ac = AnaliseDeCenarios({"A": modelo}, {"fluxo": 2_000}, cen, taxa=0.10)
    t = ac.tabela("vpl")
    assert t.loc["A", "Pessimista"] == pytest.approx(1_409, abs=2)
    assert t.loc["A", "Mais provável"] == pytest.approx(5_212, abs=2)
    assert t.loc["A", "Otimista"] == pytest.approx(9_015, abs=2)
    r = ac.resumo("vpl").loc["A"]
    assert r["Valor esperado"] == pytest.approx(5_212, abs=2)
    assert r["Desvio padrão"] == pytest.approx(2_689, abs=3)
    assert r["CV"] == pytest.approx(0.52, abs=0.01)
    assert r["P(< 0)"] == 0.0


def test_estatisticas_e_taxa_ajustada():
    v = [-10_000, 5_212, 20_424]
    p = [0.25, 0.5, 0.25]
    assert valor_esperado(v, p) == pytest.approx(5_212)
    assert desvio_padrao(v, p) == pytest.approx(10_756.5, abs=1)
    assert coeficiente_de_variacao(v, p) == pytest.approx(2.06, abs=0.01)
    curva = [(0.3, 0.07), (0.8, 0.10)]
    assert taxa_ajustada_ao_risco(0.3, curva) == pytest.approx(0.07)
    assert taxa_ajustada_ao_risco(0.8, curva) == pytest.approx(0.10)
    assert taxa_ajustada_ao_risco(0.55, curva) == pytest.approx(0.085)


def test_sensibilidade_e_equilibrio():
    def m(p):
        return Projeto("B", 5, investimento=200_000, receitas=Produto(p["unidades"], p["preco"]), custos=Produto(p["unidades"], 5.8), aliquota_ir=0.2)

    base = {"unidades": 12_500, "preco": 12.0}
    df = sensibilidade(m, base, {"unidades": 0.1, "preco": (11.0, 13.0)}, taxa=0.18)
    assert set(df["chave"]) == {"unidades", "preco"}
    assert df.iloc[0]["Amplitude"] >= df.iloc[1]["Amplitude"]
    preco_eq = ponto_de_equilibrio(m, base, "preco", taxa=0.18)
    assert vpl(m({**base, "preco": preco_eq}), 0.18) == pytest.approx(0.0, abs=1e-4)


def test_distribuicoes_ppf():
    u = np.array([0.0, 0.5, 1.0])
    assert np.allclose(Uniforme(1, 3).ppf(u), [1, 2, 3])
    assert np.allclose(Triangular(0, 1, 2).ppf(u), [0, 1, 2])
    assert Pert(0, 1, 2).media() == pytest.approx(1.0, abs=1e-3)
    assert Normal(10, 2).ppf(0.5) == pytest.approx(10)
    assert np.allclose(Discreta([1, 2, 3], [0.2, 0.5, 0.3]).ppf(np.array([0.1, 0.5, 0.9])), [1, 2, 3])


def test_monte_carlo_reprodutivel_e_coerente():
    def m(p):
        return Projeto("B", 5, investimento=200_000, receitas=Produto(p["unidades"], p["preco"]), custos=Produto(p["unidades"], p["cv"]), aliquota_ir=0.2)

    inc = {"unidades": Triangular(10_000, 12_500, 14_000), "preco": Normal(12, 0.5), "cv": Pert(5.5, 5.8, 6.3)}
    mc = MonteCarlo({"B": m}, inc, n_simulacoes=2_000, seed=1, taxa=0.18)
    r1 = mc.executar()
    r2 = MonteCarlo({"B": m}, inc, n_simulacoes=2_000, seed=1, taxa=0.18).executar()
    assert np.allclose(r1.metrica("vpl")["B"], r2.metrica("vpl")["B"])
    # VPL de cada amostra coincide com o cálculo determinístico
    i = 7
    p_i = {k: float(r1.entradas.loc[i, k]) for k in inc}
    assert r1.metrica("vpl")["B"][i] == pytest.approx(vpl(m(p_i), 0.18), rel=1e-9)
    res = r1.resumo("vpl")
    assert 0 <= res.loc["B", "P(< 0)"] <= 1
    assert res.loc["B", "P5"] <= res.loc["B", "Mediana"] <= res.loc["B", "P95"]
    assert len(r1.sensibilidade("B")) == 3


def test_monte_carlo_correlacao_e_carteiras():
    def m1(p):
        return Projeto("X", 5, investimento=100_000, receitas=p["a"] * 1.0, aliquota_ir=0.0, depreciacao=0.0)

    def m2(p):
        return Projeto("Y", 5, investimento=100_000, receitas=p["b"] * 1.0, aliquota_ir=0.0, depreciacao=0.0)

    inc = {"a": Normal(30_000, 3_000), "b": Normal(30_000, 3_000)}
    r = MonteCarlo({"X": m1, "Y": m2}, inc, n_simulacoes=4_000, seed=3, taxa=0.10, correlacoes={("a", "b"): 0.7}).executar()
    rho = r.entradas[["a", "b"]].corr(method="spearman").loc["a", "b"]
    assert 0.6 < rho < 0.8
    tab, dist = r.carteiras(k=2)
    assert list(tab.index) == ["X + Y"]
    assert np.allclose(dist["X + Y"], r.metrica("vpl")["X"] + r.metrica("vpl")["Y"])
    pm = r.prob_melhor()
    assert pm.sum() == pytest.approx(1.0)


def test_por_periodo():
    def m(p):
        return Projeto("Z", 3, investimento=1_000, receitas=p["serie"], aliquota_ir=0.0, depreciacao=0.0)

    r = MonteCarlo({"Z": m}, {"serie": PorPeriodo([Uniforme(400, 500)] * 3)}, n_simulacoes=500, seed=0, taxa=0.1).executar()
    assert r.fluxos["Z"].valores.shape == (500, 4)
    assert {"serie[1]", "serie[2]", "serie[3]"} <= set(r.entradas.columns)
