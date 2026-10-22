import numpy as np
import pytest

from greenseer import FluxoDeCaixa, analise_marginal, campeao_incremental, carteiras, equalizar, fluxo_marginal, matriz_marginal, matriz_tir_marginal, mmc, ranking, rau, somar, tir, vpl


def test_mmc_e_equalizar(projetos_gama, taxa):
    assert mmc(5, 7) == 35 and mmc(5, 6, 7) == 210
    fs = equalizar(projetos_gama, "mmc")
    assert {f.n for f in fs} == {210}
    fs = equalizar(projetos_gama, "estender")
    assert {f.n for f in fs} == {7}
    fs = equalizar(projetos_gama, "truncar", taxa=taxa)
    assert {f.n for f in fs} == {5}
    # truncar com valor econômico preserva o VPL
    for f, p in zip(fs, projetos_gama):
        assert vpl(f, taxa) == pytest.approx(vpl(p, taxa))


def test_marginal_p2_p3_grupo7(projetos_gama, taxa):
    p1, p2, p3, p4 = projetos_gama
    m = fluxo_marginal(p2, p3, "mmc")
    assert m.n == 35
    assert tir(m) == pytest.approx(0.17774, abs=1e-4)
    assert vpl(m, taxa) == pytest.approx(-1_260.56, abs=0.05)
    # RAU marginal = diferença dos RAU
    assert rau(m, taxa) == pytest.approx(rau(p3, taxa) - rau(p2, taxa), rel=1e-9)


def test_analise_marginal_tabela(projetos_gama, taxa):
    df = analise_marginal(projetos_gama, taxa)
    assert len(df) == 6
    linha = df[(df["Defensor"] == "P2") & (df["Desafiante"] == "P3")].iloc[0]
    assert linha["Preferido"] == "P2"
    linha = df[(df["Defensor"] == "P1") & (df["Desafiante"] == "P2")].iloc[0]
    assert np.isnan(linha["TIR marginal"]) and linha["Preferido"] == "P2"
    linha = df[(df["Defensor"] == "P3") & (df["Desafiante"] == "P4")].iloc[0]
    assert linha["Nº de TIRs"] == 2 and linha["Preferido"] == "P4"
    M = matriz_tir_marginal(projetos_gama, taxa)
    assert M.loc["P3", "P2"] == pytest.approx(M.loc["P2", "P3"])
    T = matriz_marginal(projetos_gama, taxa)
    assert "P2" in T.loc["P3", "P2"]


def test_campeao_e_carteiras(projetos_gama, taxa):
    campeao, trilha = campeao_incremental(projetos_gama, taxa)
    assert campeao == "P4"
    assert list(trilha["Vence"]) == ["P1", "P2", "P2", "P4"]
    df, agg = carteiras(projetos_gama, taxa, k=2)
    assert len(df) == 6
    assert df.iloc[0]["Carteira"] == "P3 + P4"
    assert df.iloc[0]["VPL"] == pytest.approx(36_440.16 + 42_349.63, abs=0.05)
    assert df.set_index("Carteira").loc["P2 + P4", "RAU (soma)"] == pytest.approx(9_788.11 + 12_108.19, abs=0.05)
    df_orc, _ = carteiras(projetos_gama, taxa, k=2, orcamento=450_000)
    assert set(df_orc["Carteira"]) == {"P1 + P2", "P1 + P3", "P2 + P3"}


def test_ranking(projetos_gama, taxa):
    r = ranking(projetos_gama, taxa)
    assert r.loc["P4", "rank_vpl"] == 1 and r.loc["P2", "rank_tir"] == 1 and r.loc["P4", "rank_rau"] == 1
    assert r.loc["P3", "rank_vpl"] == 2 and r.loc["P3", "rank_rau"] == 3  # vidas diferentes invertem P2/P3


def test_somar_alinha_vidas():
    a = FluxoDeCaixa([-100, 60, 60], "a")
    b = FluxoDeCaixa([-50, 30, 30, 30], "b")
    s = somar([a, b])
    assert np.allclose(s.valores, [-150, 90, 90, 30])
