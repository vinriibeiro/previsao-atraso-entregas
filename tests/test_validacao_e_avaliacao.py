"""Testes da separação temporal, das métricas de negócio e do pipeline do modelo."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from atraso_entregas import avaliacao
from atraso_entregas.config import PremissasDeCusto
from atraso_entregas.features import ALVO
from atraso_entregas.modelo import (
    VARIAVEIS,
    criar_gradient_boosting,
    criar_mistura,
    criar_regressao_logistica,
)
from atraso_entregas.validacao import separar_periodos

RECORTES = {
    "inicio_treino": "2017-03-01",
    "fim_treino": "2018-01-01",
    "data_de_corte": "2018-05-01",
}


def test_periodos_nao_se_sobrepoem_e_respeitam_a_ordem(tabela):
    p = separar_periodos(tabela, **RECORTES)

    assert len(p.treino) and len(p.validacao) and len(p.teste)
    assert p.treino["data_compra"].max() < p.validacao["data_compra"].min()
    assert p.validacao["data_compra"].max() < p.teste["data_compra"].min()
    ids = pd.concat([p.treino, p.validacao, p.teste])["order_id"]
    assert ids.is_unique


def test_desenvolvimento_so_tem_pedidos_com_prazo_vencido_no_corte(tabela):
    p = separar_periodos(tabela, **RECORTES)
    corte = pd.Timestamp(RECORTES["data_de_corte"])

    assert (p.desenvolvimento["data_prometida"] < corte).all()
    assert p.n_imaturos > 0
    assert (p.teste["data_compra"] >= corte).all()


def test_datas_fora_de_ordem_dao_erro(tabela):
    with pytest.raises(ValueError):
        separar_periodos(tabela, "2018-01-01", "2017-01-01", "2018-05-01")


def test_custo_da_decisao_confere_com_a_conta_manual():
    premissas = PremissasDeCusto(custo_atraso=40, custo_acao=4, eficacia_acao=0.5)
    y = np.array([1, 1, 0, 0, 0])
    sinalizado = np.array([1, 0, 1, 0, 0])

    r = avaliacao.resultado_da_decisao(y, sinalizado, premissas)

    # 1 atraso perdido (40) + 2 ações (8) + 1 atraso sinalizado, evitado pela metade (20)
    assert r["custo"] == pytest.approx(68)
    assert r["custo_sem_acao"] == pytest.approx(80)
    assert r["economia"] == pytest.approx(12)
    assert r["precisao"] == pytest.approx(0.5)
    assert r["recall"] == pytest.approx(0.5)


def test_precisao_minima_e_o_ponto_de_equilibrio():
    premissas = PremissasDeCusto(custo_atraso=40, custo_acao=4, eficacia_acao=0.5)
    assert premissas.precisao_minima == pytest.approx(0.20)
    # Na precisão mínima, a ação custa exatamente o que evita.
    assert avaliacao.custo_maximo_da_acao(0.20, premissas) == pytest.approx(premissas.custo_acao)


def test_nao_sinalizar_ninguem_nao_gera_economia_nem_erro():
    r = avaliacao.resultado_da_decisao(np.array([1, 0, 0]), np.zeros(3, dtype=bool))
    assert r["economia"] == 0
    assert np.isnan(r["precisao"])


def test_topo_por_semana_sinaliza_a_fracao_pedida_em_cada_semana():
    rng = np.random.default_rng(2)
    datas = pd.Timestamp("2018-06-04") + pd.to_timedelta(rng.integers(0, 28, 400), unit="D")
    prob = rng.uniform(0, 1, 400)

    sinalizado = avaliacao.topo_por_semana(datas, prob, fracao=0.10)

    df = pd.DataFrame(
        {"semana": pd.Series(datas).dt.to_period("W"), "prob": prob, "sinalizado": sinalizado}
    )
    for _, semana in df.groupby("semana"):
        assert semana["sinalizado"].sum() == int(np.ceil(len(semana) * 0.10))
        # os sinalizados são sempre os de maior nota da semana
        assert (
            semana.loc[semana["sinalizado"], "prob"].min()
            >= semana.loc[~semana["sinalizado"], "prob"].max()
        )


def test_captura_no_topo_com_ranqueamento_perfeito():
    y = np.array([1] * 10 + [0] * 90)
    prob = np.linspace(1, 0, 100)
    assert avaliacao.captura_no_topo(y, prob, 0.10) == pytest.approx(1.0)
    assert avaliacao.captura_no_topo(y, prob, 0.05) == pytest.approx(0.5)


@pytest.mark.parametrize(
    "criar", [criar_gradient_boosting, criar_regressao_logistica, criar_mistura]
)
def test_pipeline_treina_e_aceita_categoria_nova_e_valores_ausentes(tabela, criar):
    p = separar_periodos(tabela, **RECORTES)
    modelo = criar().fit(p.treino[VARIAVEIS], p.treino[ALVO])

    novos = p.teste[VARIAVEIS].head(20).copy()
    novos.loc[novos.index[0], "categoria_principal"] = "categoria_que_nao_existia"
    novos.loc[novos.index[1], ["distancia_km", "vendedor_taxa_atraso"]] = np.nan
    prob = modelo.predict_proba(novos)[:, 1]

    assert prob.shape == (20,)
    assert np.isfinite(prob).all()
    assert ((prob >= 0) & (prob <= 1)).all()
