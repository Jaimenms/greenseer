"""Árvore de cenários, variações e avaliação ajustada ao risco (caso Gama, Parte 2)."""
import numpy as np
import pytest

from greenseer import AnaliseDeCenarios, ArvoreDeCenarios, Cenario, Hipotese, Produto, Projeto, aplicar_cenario, taxa_ajustada_ao_risco, viz, vpl

CURVA = [(0.1, 0.18), (0.3, 0.19), (0.5, 0.20), (0.7, 0.22), (0.9, 0.25)]


def arvore(param, sinal=1.0):
    alt = lambda n, v, p: Cenario(n, probabilidade=p, variacoes={param: sinal * v})
    return ArvoreDeCenarios([
        Hipotese("Pessimista", 0.10, [alt("+2%", 0.02, 0.70), alt("+5%", 0.05, 0.15), alt("+7%", 0.07, 0.10), alt("+10%", 0.10, 0.05)]),
        Hipotese("Mais provável", 0.85),
        Hipotese("Otimista", 0.05, [alt("−2%", -0.02, 0.60), alt("−5%", -0.05, 0.20), alt("−8%", -0.08, 0.15), alt("−10%", -0.10, 0.05)]),
    ])


def modelo(p):
    return Projeto("P", 5, investimento=200_000, receitas=[570_000, 550_000, 510_000, 423_000, 423_000],
                   custos={"CPV": Produto(p["cpv"], 1.0) if False else __import__("greenseer").Percentual(p["cpv"]), "Outros": __import__("greenseer").Percentual(0.19)}, aliquota_ir=0.2)


def test_expansao_probabilidades_conjuntas():
    a = arvore("cpv")
    folhas = a.expandir()
    assert len(folhas) == 9
    assert [round(c.probabilidade, 6) for c in folhas] == [0.07, 0.015, 0.01, 0.005, 0.85, 0.03, 0.01, 0.0075, 0.0025]
    assert abs(sum(c.probabilidade for c in folhas) - 1) < 1e-12
    assert folhas[0].hipotese == "Pessimista" and folhas[0].probabilidade_condicional == 0.70
    assert folhas[4].nome == "Mais provável" and folhas[4].variacoes == {}
    t = a.tabela()
    assert t["P conjunta"].sum() == pytest.approx(1.0)
    valores, probs = a.como_distribuicao({"cpv": 0.60}, "cpv")
    assert valores[0] == pytest.approx(0.612) and probs.sum() == pytest.approx(1.0)


def test_validacao():
    with pytest.raises(ValueError):
        ArvoreDeCenarios([Hipotese("a", 0.6), Hipotese("b", 0.5)]).expandir()
    with pytest.raises(ValueError):
        ArvoreDeCenarios([Hipotese("a", 1.0, [Cenario("x", probabilidade=0.7), Cenario("y", probabilidade=0.2)])]).expandir()


def test_aplicar_cenario():
    base = {"cpv": 0.60, "preco": 20.0}
    assert aplicar_cenario(base, Cenario("r", variacoes={"cpv": 0.02}))["cpv"] == pytest.approx(0.612)
    assert aplicar_cenario(base, Cenario("pp", variacoes_pp={"cpv": 0.02}))["cpv"] == pytest.approx(0.62)
    assert aplicar_cenario(base, Cenario("s", parametros={"preco": 19.0}))["preco"] == 19.0
    with pytest.raises(KeyError):
        aplicar_cenario(base, Cenario("x", variacoes={"inexistente": 0.1}))


def test_analise_com_arvore_e_avaliacao_ajustada():
    base = {"cpv": 0.60}
    ac = AnaliseDeCenarios({"P": modelo}, base, arvore("cpv"), taxa=0.18)
    res = ac.executar()
    assert len(res) == 9 and res["Probabilidade"].sum() == pytest.approx(1.0)
    # valor esperado = Σ p · VPL
    e_manual = float((res["Probabilidade"] * res["vpl"]).sum())
    r = ac.resumo("vpl").loc["P"]
    assert r["Valor esperado"] == pytest.approx(e_manual)
    dp_manual = float(np.sqrt((res["Probabilidade"] * (res["vpl"] - e_manual) ** 2).sum()))
    assert r["Desvio padrão"] == pytest.approx(dp_manual)
    assert r["CV"] == pytest.approx(dp_manual / e_manual)
    # tabela de dispersão no formato das notas
    td = ac.tabela_dispersao("P")
    assert td.iloc[-1]["pi · (Ei − E)²"] == pytest.approx(dp_manual ** 2)
    assert td.attrs["desvio_padrao"] == pytest.approx(dp_manual)
    # por hipótese
    ph = ac.resumo_por_hipotese("vpl").set_index("Hipótese")
    assert ph.loc["Pessimista", "P(hipótese)"] == pytest.approx(0.10) and ph.loc["Mais provável", "P(hipótese)"] == pytest.approx(0.85)
    assert ph["Contribuição P × E"].sum() == pytest.approx(e_manual)
    # taxa ajustada e reavaliação do caso-base
    av = ac.avaliacao_ajustada(CURVA)
    tx = taxa_ajustada_ao_risco(r["CV"], CURVA)
    assert av.loc["P", "Taxa ajustada"] == pytest.approx(tx)
    assert av.loc["P", "VPL ajustado"] == pytest.approx(vpl(modelo(base), tx))
    assert av.loc["P", "VPL à TMA"] == pytest.approx(vpl(modelo(base), 0.18))
    assert av.loc["P", "Decisão"] in ("aceitar", "rejeitar")


def test_arvores_por_modelo():
    base = {"cpv": 0.60}
    ac = AnaliseDeCenarios({"A": modelo, "B": modelo}, base, {"A": arvore("cpv"), "B": [Cenario("único", probabilidade=1.0)]}, taxa=0.18)
    res = ac.executar()
    assert (res["Projeto"] == "A").sum() == 9 and (res["Projeto"] == "B").sum() == 1
    assert ac.resumo().loc["B", "Desvio padrão"] == 0


def test_taxa_ajustada_extrapolacao():
    assert taxa_ajustada_ao_risco(0.0, CURVA) == pytest.approx(0.18)
    assert taxa_ajustada_ao_risco(0.4, CURVA) == pytest.approx(0.195)
    assert taxa_ajustada_ao_risco(1.1, CURVA) == pytest.approx(0.25)
    assert taxa_ajustada_ao_risco(1.1, CURVA, extrapolar=True) == pytest.approx(0.28)
    assert taxa_ajustada_ao_risco(0.0, CURVA, extrapolar=True) == pytest.approx(0.175)


def test_graficos_arvore():
    ac = AnaliseDeCenarios({"P": modelo}, {"cpv": 0.60}, arvore("cpv"), taxa=0.18)
    ac.executar()
    f1 = viz.grafico_arvore(ac, "P")
    f2 = viz.grafico_distribuicao_discreta(ac, "P")
    f3 = viz.grafico_curva_indiferenca(CURVA, ac.resumo()["CV"], extrapolar=True)
    assert all(len(f.data) > 0 for f in (f1, f2, f3))
