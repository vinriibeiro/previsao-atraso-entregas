"""Testes da tabela analítica, com foco em vazamento de dados."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from atraso_entregas.features import (
    ALVO,
    COLUNAS_DE_DESFECHO,
    _media_suavizada,
    agregado_anterior,
    centroides_por_cep,
    distancia_km,
    montar_tabela,
)
from atraso_entregas.modelo import CONTAGENS_ACUMULADAS, NUMERICAS_PRIMEIRA_VERSAO, VARIAVEIS


def test_distancia_entre_sao_paulo_e_rio():
    km = distancia_km(-23.55, -46.63, -22.91, -43.17)
    assert km == pytest.approx(360, abs=10)


def test_distancia_zero_e_ausente():
    assert distancia_km(-10.0, -50.0, -10.0, -50.0) == pytest.approx(0)
    assert np.isnan(distancia_km(np.nan, -50.0, -10.0, -50.0))


def _forca_bruta(consultas, eventos, chave, janela):
    """Implementação ingênua, linha a linha, para conferir a versão vetorizada."""
    somas, contagens = [], []
    for _, consulta in consultas.iterrows():
        passado = eventos[
            (eventos["quando"] < consulta["agora"]) & (eventos[chave] == consulta[chave])
        ]
        if janela is not None:
            passado = passado[passado["quando"] >= consulta["agora"] - janela]
        somas.append(passado["valor"].sum())
        contagens.append(len(passado))
    return np.array(somas), np.array(contagens)


@pytest.mark.parametrize("janela", [None, pd.Timedelta(days=30)])
def test_agregado_anterior_bate_com_forca_bruta(janela):
    rng = np.random.default_rng(1)
    inicio = pd.Timestamp("2018-01-01")
    eventos = pd.DataFrame(
        {
            "chave": rng.choice(["a", "b", "c"], 300),
            # dias inteiros de propósito: gera empates entre evento e consulta
            "quando": inicio + pd.to_timedelta(rng.integers(0, 120, 300), unit="D"),
            "valor": rng.integers(0, 2, 300).astype(float),
        }
    )
    consultas = pd.DataFrame(
        {
            "chave": rng.choice(["a", "b", "c", "sem_historico"], 80),
            "agora": inicio + pd.to_timedelta(rng.integers(0, 130, 80), unit="D"),
        },
        index=rng.permutation(80),  # índice embaralhado: a ordem das linhas precisa ser preservada
    )

    obtido = agregado_anterior(
        consultas,
        eventos,
        tempo_consulta="agora",
        tempo_evento="quando",
        valor="valor",
        chaves=["chave"],
        janela=janela,
    )
    soma, n = _forca_bruta(consultas, eventos, "chave", janela)

    assert obtido.index.equals(consultas.index)
    np.testing.assert_allclose(obtido["soma"], soma)
    np.testing.assert_allclose(obtido["n"], n)


def test_agregado_anterior_ignora_evento_no_mesmo_instante():
    agora = pd.Timestamp("2018-03-01 10:00")
    eventos = pd.DataFrame(
        {"quando": [agora, agora - pd.Timedelta(seconds=1)], "valor": [1.0, 1.0]}
    )
    consultas = pd.DataFrame({"agora": [agora]})

    obtido = agregado_anterior(
        consultas, eventos, tempo_consulta="agora", tempo_evento="quando", valor="valor"
    )

    assert obtido["n"].iloc[0] == 1


def test_uma_linha_por_pedido_entregue(tabelas, tabela):
    entregues = (tabelas["pedidos"]["order_status"] == "delivered").sum()
    assert len(tabela) == entregues
    assert tabela["order_id"].is_unique
    assert set(tabela[ALVO].unique()) <= {0, 1}


def test_entrega_no_dia_prometido_nao_e_atraso(tabelas):
    pedidos = tabelas["pedidos"].copy()
    prometida = pedidos.loc[1, "order_estimated_delivery_date"]
    pedidos.loc[1, "order_delivered_customer_date"] = prometida + pd.Timedelta(hours=18)
    pedidos.loc[2, "order_delivered_customer_date"] = pedidos.loc[
        2, "order_estimated_delivery_date"
    ] + pd.Timedelta(days=1, hours=1)
    tabela = montar_tabela({**tabelas, "pedidos": pedidos}).set_index("order_id")

    assert tabela.loc["p1", ALVO] == 0
    assert tabela.loc["p2", ALVO] == 1


def test_nenhuma_variavel_do_modelo_e_desfecho():
    usadas = set(VARIAVEIS) | set(NUMERICAS_PRIMEIRA_VERSAO) | set(CONTAGENS_ACUMULADAS)
    assert not usadas & set(COLUNAS_DE_DESFECHO)
    assert not {"data_compra", "data_prometida", "order_id", "id_vendedor"} & usadas


def test_variaveis_nao_enxergam_o_futuro(tabelas, tabela):
    """O teste que mais importa: mudar o futuro não pode mudar o passado.

    Toda postagem e toda entrega que aconteceu depois de uma data é empurrada
    60 dias para a frente (o que também transforma essas entregas em atrasos).
    Para quem comprou antes dessa data, nada disso tinha acontecido ainda:
    as variáveis desses pedidos precisam continuar idênticas. Isso inclui
    pedidos comprados antes e entregues depois, que é onde o vazamento se esconde.
    """
    virada = pd.Timestamp("2018-01-01")
    pedidos = tabelas["pedidos"].copy()
    for coluna in ["order_delivered_carrier_date", "order_delivered_customer_date"]:
        pedidos.loc[pedidos[coluna] >= virada, coluna] += pd.Timedelta(days=60)
    alterada = montar_tabela({**tabelas, "pedidos": pedidos})
    assert alterada[ALVO].sum() > tabela[ALVO].sum()  # o futuro realmente mudou

    colunas = sorted(set(VARIAVEIS) | set(NUMERICAS_PRIMEIRA_VERSAO) | set(CONTAGENS_ACUMULADAS))
    antes = tabela[tabela["data_compra"] < virada].set_index("order_id")[colunas]
    depois = alterada[alterada["data_compra"] < virada].set_index("order_id")[colunas]

    assert len(antes) > 100
    pd.testing.assert_frame_equal(antes, depois)


def test_centroides_descartam_pontos_fora_do_brasil(tabelas):
    centroides = centroides_por_cep(tabelas["geolocalizacao"])
    assert centroides.loc["01310", "lat"] == pytest.approx(-23.56)


def test_primeiro_pedido_do_vendedor_nao_tem_historico(tabela):
    primeiro = tabela.sort_values("data_compra").groupby("id_vendedor").head(1)
    assert (primeiro["vendedor_n_entregas"] == 0).all()
    assert primeiro["vendedor_dias_postagem"].isna().all()


def test_media_suavizada_puxa_pouco_historico_para_a_previa():
    agregado = pd.DataFrame({"soma": [0.0, 1.0, 100.0], "n": [0.0, 1.0, 1000.0]})
    previa = np.array([0.05, 0.05, 0.05])

    suavizada = _media_suavizada(agregado, previa)

    assert suavizada[0] == pytest.approx(0.05)  # sem histórico: vale a média geral
    assert 0.05 < suavizada[1] < 0.15  # 1 atraso em 1 entrega não vira 100%
    assert suavizada[2] == pytest.approx(0.10, abs=0.002)  # muito histórico: vale o próprio
