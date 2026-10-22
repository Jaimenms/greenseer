import pandas as pd

from greenseer import AnaliseDeCenarios, Cenario, MonteCarlo, Triangular, analise_marginal, carteiras, estilizar, estilizar_demonstrativo, sensibilidade, tabela_fluxos, tabela_indicadores, viz


def test_graficos_constroem(projetos_gama, taxa):
    tab = tabela_indicadores(projetos_gama, taxa)
    figs = [
        viz.grafico_fluxo(projetos_gama[3], taxa=taxa, acumulado=True),
        viz.grafico_fluxos(projetos_gama),
        viz.grafico_fluxos_empilhados(projetos_gama[1:3]),
        viz.grafico_perfil_vpl(projetos_gama, taxa),
        viz.grafico_indicador(tab, "vpl"),
        viz.grafico_painel_indicadores(tab, referencias={"tir": taxa}),
        viz.grafico_matriz_marginal(analise_marginal(projetos_gama, taxa)),
        viz.grafico_carteiras(carteiras(projetos_gama, taxa, k=2)[0].set_index("Carteira"), coluna="VPL", faixa=None),
    ]
    assert all(len(f.data) > 0 for f in figs)


def test_graficos_risco(projetos_gama, taxa):
    def m(p):
        return projetos_gama[0].com(receitas=p["rec"])

    ac = AnaliseDeCenarios({"P1": m}, {"rec": 150_000}, [Cenario("Pess.", {"rec": 120_000}, 0.3), Cenario("Base", {}, 0.7)], taxa)
    ac.executar()
    f1 = viz.grafico_cenarios(ac.resultado, "vpl", ac.resumo("vpl"))
    s = sensibilidade(m, {"rec": 150_000}, {"rec": 0.1}, taxa)
    f2 = viz.grafico_tornado(s)
    r = MonteCarlo({"P1": m}, {"rec": Triangular(120_000, 150_000, 170_000)}, n_simulacoes=300, taxa=taxa).executar()
    figs = [f1, f2, viz.grafico_distribuicoes(r), viz.grafico_cdf(r), viz.grafico_risco_retorno(r.resumo()), viz.grafico_leque(r, "P1"), viz.grafico_prob_melhor(r.prob_melhor()), viz.grafico_carteiras(r.carteiras()[0])]
    assert all(len(f.data) > 0 for f in figs)


def test_componentes_empilhadas_e_diagrama():
    import numpy as np

    from greenseer import FluxoDeCaixa

    a = FluxoDeCaixa([-64_000] + [21_000] * 4, "A")
    ent, sai = viz.componentes_empilhadas(a, ciclos=3)
    assert len(ent) == 13
    assert ent[4] == 21_000 and sai[4] == -64_000      # ano de reinvestimento: as duas setas
    assert ent[0] == 0 and sai[0] == -64_000 and sai[12] == 0
    assert np.allclose(ent + sai, a.replicar(3).valores)  # líquido = fluxo empilhado
    fig = viz.grafico_fluxos_empilhados([a, FluxoDeCaixa([-122_000, 29_000, 28_000, 27_000, 26_000, 25_000, 44_000], "B")])
    assert len(fig.data) > 0
    fig2 = viz.grafico_fluxos_empilhados([a], estilo="barras")
    assert len(fig2.data) == 1


def test_componentes_projeto_e_grafico_fluxo(projetos_gama):
    from greenseer import Financiamento, Projeto, ValorResidual

    comps = viz.componentes_projeto(projetos_gama[2])
    nomes = [c["nome"] for c in comps]
    assert "Valor residual (líquido de IR)" in nomes and "Investimento" in nomes
    vr = next(c for c in comps if c["nome"].startswith("Valor residual"))["valores"]
    assert vr[7] == 36_600 and vr[:7].sum() == 0
    total = sum(c["valores"] for c in comps)
    assert abs(total - projetos_gama[2].fluxo().valores).max() < 1e-6   # componentes somam o FCG
    fig = viz.grafico_fluxo(projetos_gama[2])
    assert any("Valor residual" in (tr.name or "") for tr in fig.data)
    novo = Projeto("N", 3, investimento=1_000, receitas=600, aliquota_ir=0.3, capital_de_giro=100,
                   valor_residual=ValorResidual.mercado(50), financiamento=Financiamento(500, 0.1, 2))
    fig2 = viz.grafico_fluxo(novo, tipo="proprio", acumulado=True, taxa=0.1)
    assert any("financiamento" in (tr.name or "") for tr in fig2.data)
    fig3 = viz.grafico_fluxo(projetos_gama[0].fluxo(), estilo="barras")
    assert len(fig3.data) == 1


def test_exportar_html(tmp_path, projetos_gama, taxa):
    fig = viz.grafico_fluxo(projetos_gama[0])
    saida = viz.exportar(fig, str(tmp_path / "fig"), formatos=("html",))
    assert saida and (tmp_path / "fig.html").exists()


def test_relatorios(projetos_gama, taxa):
    tab = tabela_indicadores(projetos_gama, taxa)
    html = estilizar(tab, titulo="Indicadores").to_html()
    assert "$ 42.350" in html and "25,46%" in html
    html2 = estilizar_demonstrativo(projetos_gama[0].demonstrativo()).to_html()
    assert "Fluxo de caixa global" in html2
    df = tabela_fluxos(projetos_gama)
    assert df.shape == (8, 4) and df.loc[7, "P1"] == 0
