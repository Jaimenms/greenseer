"""Validação contra os exemplos das notas de Análise de Investimentos (FDC)."""
import numpy as np
import pytest

from greenseer import FluxoDeCaixa, ibc, payback, rau, taxa_fisher, tir, tirm, vp, vpl, vpl_perfil, roia
from greenseer.metricas import fator_recuperacao_capital, fator_valor_atual, rau_componentes

A = np.array([-60_000] + [20_000] * 6)
B = np.array([-72_000, 45_000, 22_000, 20_000, 13_000, 13_000, 13_000])
A2 = np.array([-64_000] + [21_000] * 4)
B2 = np.array([-122_000, 29_000, 28_000, 27_000, 26_000, 25_000, 44_000])
T = np.array([-180_000, 26_200, 61_200, 138_200, 103_200, 129_200])


def test_fatores():
    assert fator_valor_atual(0.10, 6) == pytest.approx(4.3553, abs=1e-4)
    assert fator_recuperacao_capital(0.10, 4) == pytest.approx(0.31547, abs=1e-5)
    assert fator_valor_atual(0.0, 5) == 5


def test_vpl_e_vp_slides():
    # slide: VAFLCx 87.080 e VPL +27.080 (fatores arredondados a 3 casas)
    assert vp(A, 0.10) == pytest.approx(87_105, rel=1e-3)
    assert vpl(A, 0.10) == pytest.approx(27_105, rel=1e-3)
    assert vpl(B, 0.10) == pytest.approx(26_381, rel=2e-3)


def test_payback_ajustado_slides():
    assert payback(A, 0.10) == pytest.approx(3.75, abs=0.01)
    assert payback(B, 0.10) == pytest.approx(2.86, abs=0.01)
    assert payback(A) == pytest.approx(3.0)
    assert np.isnan(payback(np.array([-100, 10, 10])))
    assert payback(np.array([100, -10])) == 0.0


def test_ibc_slides():
    assert ibc(A, 0.10) == pytest.approx(1.45, abs=0.01)
    assert ibc(B, 0.10) == pytest.approx(1.37, abs=0.01)


def test_rau_slides():
    assert rau(A2, 0.10) == pytest.approx(810, abs=1)
    assert rau(B2, 0.10) == pytest.approx(1_351, abs=6)  # 1.356 com fatores exatos
    comp = rau_componentes(B2, 0.10)
    assert comp["eau_investimento"] == pytest.approx(28_012, rel=2e-3)
    assert comp["rau"] == pytest.approx(rau(B2, 0.10))


def test_tir_slides():
    assert tir(A2) == pytest.approx(0.1184, abs=5e-4)
    assert tir(B2) == pytest.approx(0.1156, abs=5e-4)
    assert tir(T) == pytest.approx(0.3225, abs=1e-3)  # 32,25% por interpolação no slide
    assert vpl(T, tir(T)) == pytest.approx(0.0, abs=1e-4)


def test_tirm_slides():
    assert tirm(T, 0.25) == pytest.approx(0.296, abs=1e-3)
    assert tirm(T, 0.18) == pytest.approx(0.270, abs=1e-3)


def test_empilhamento_e_tir_marginal_slides():
    a = FluxoDeCaixa(A2, "A").replicar(3)
    b = FluxoDeCaixa(B2, "B").replicar(2)
    assert a.n == b.n == 12
    assert a.valores[4] == pytest.approx(21_000 - 64_000)
    m = b - a
    assert m.valores[0] == pytest.approx(-58_000)
    assert m.valores[4] == pytest.approx(69_000)
    assert m.valores[6] == pytest.approx(-99_000)
    assert tir(m) == pytest.approx(0.1133, abs=1.5e-3)  # 11,33% (interpolado) no slide
    assert vpl(m, 0.10) == pytest.approx(3_807, rel=0.03)
    # RAU é invariante ao empilhamento
    assert rau(a, 0.10) == pytest.approx(rau(A2, 0.10), rel=1e-9)
    assert rau(b, 0.10) == pytest.approx(rau(B2, 0.10), rel=1e-9)
    # a taxa de Fisher iguala os VPL
    r = taxa_fisher(a, b)
    assert vpl(a, r) == pytest.approx(vpl(b, r), abs=1e-6)


def test_vetorizado_coerente():
    M = np.array([np.pad(A, (0, 0)), B, np.pad(A2, (0, 2))], dtype=float)
    assert np.allclose(vpl(M, 0.10), [vpl(A, 0.10), vpl(B, 0.10), vpl(A2, 0.10)])
    assert np.allclose(tir(M), [tir(A), tir(B), tir(A2)], atol=1e-8)
    assert np.allclose(payback(M, 0.10), [payback(A, 0.10), payback(B, 0.10), payback(A2, 0.10)])
    assert np.allclose(ibc(M, 0.10), [ibc(A, 0.10), ibc(B, 0.10), ibc(A2, 0.10)])
    # a TIRM depende do horizonte: compara com o mesmo fluxo completado com zeros
    assert np.allclose(tirm(M, 0.10), [tirm(A, 0.10), tirm(B, 0.10), tirm(np.pad(A2, (0, 2)), 0.10)])


def test_tir_multiplas_raizes():
    f = np.array([-100.0, 230.0, -132.0])  # raízes em 10% e 20%
    raizes = tir(f, todas=True)
    assert np.allclose(raizes, [0.10, 0.20], atol=1e-6)
    with pytest.warns(UserWarning):
        assert tir(f) == pytest.approx(0.10, abs=1e-6)


def test_perfil_e_roia():
    perfil = vpl_perfil(A, [0.0, 0.10])
    assert perfil[0] == pytest.approx(60_000)
    assert perfil[1] == pytest.approx(vpl(A, 0.10))
    assert roia(A, 0.10) == pytest.approx(ibc(A, 0.10) ** (1 / 6) - 1)
