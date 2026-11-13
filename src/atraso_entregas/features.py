"""Monta a tabela analítica: uma linha por pedido entregue.

Regra central: toda variável precisa estar disponível **no instante da compra**.
Variáveis de histórico usam apenas eventos cujo resultado já era conhecido antes
daquele instante (ver `agregado_anterior`).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

ALVO = "atrasou"

# Colunas que só existem depois da compra. Servem para montar o alvo e o
# histórico, e nunca podem entrar como variável do modelo.
COLUNAS_DE_DESFECHO = [
    "data_postagem",
    "data_entrega",
    "dias_ate_entrega",
    "dias_ate_postagem",
    ALVO,
]

# Limites aproximados do território brasileiro, para descartar coordenadas erradas.
LAT_BRASIL = (-34.0, 5.5)
LNG_BRASIL = (-74.0, -34.0)

REGIOES = {
    "Norte": ["AC", "AM", "AP", "PA", "RO", "RR", "TO"],
    "Nordeste": ["AL", "BA", "CE", "MA", "PB", "PE", "PI", "RN", "SE"],
    "Centro-Oeste": ["DF", "GO", "MS", "MT"],
    "Sudeste": ["ES", "MG", "RJ", "SP"],
    "Sul": ["PR", "RS", "SC"],
}
UF_PARA_REGIAO = {uf: regiao for regiao, ufs in REGIOES.items() for uf in ufs}

_DIA = pd.Timedelta(days=1)
PESO_DA_PREVIA = 20


def distancia_km(lat1, lng1, lat2, lng2):
    """Distância em linha reta entre dois pontos (fórmula de haversine)."""
    lat1, lng1, lat2, lng2 = (
        np.radians(np.asarray(v, dtype="float64")) for v in (lat1, lng1, lat2, lng2)
    )
    a = (
        np.sin((lat2 - lat1) / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin((lng2 - lng1) / 2) ** 2
    )
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


def agregado_anterior(
    consultas: pd.DataFrame,
    eventos: pd.DataFrame,
    *,
    tempo_consulta: str,
    tempo_evento: str,
    valor: str,
    chaves: list[str] | None = None,
    janela: pd.Timedelta | None = None,
) -> pd.DataFrame:
    """Soma e contagem dos eventos ocorridos **estritamente antes** de cada consulta.

    Para cada linha de `consultas`, considera os eventos da mesma chave com
    `tempo_evento` no intervalo [tempo_consulta - janela, tempo_consulta).
    Sem `janela`, considera todo o passado.

    Retorna um DataFrame com as colunas `soma` e `n`, no mesmo índice de `consultas`.
    """
    chaves = list(chaves or [])

    ev = eventos.dropna(subset=[tempo_evento, valor, *chaves]).sort_values(
        tempo_evento, kind="stable"
    )
    ev = ev[[*chaves, tempo_evento, valor]].copy()
    if chaves:
        grupos = ev.groupby(chaves, sort=False)[valor]
        ev["_soma"] = grupos.cumsum()
        ev["_n"] = grupos.cumcount() + 1
    else:
        ev["_soma"] = ev[valor].cumsum()
        ev["_n"] = np.arange(1, len(ev) + 1)
    ev = ev.drop(columns=valor)

    def acumulado_ate(instantes: pd.Series) -> pd.DataFrame:
        base = consultas[chaves].copy()
        base["_t"] = instantes.to_numpy()
        base["_linha"] = np.arange(len(base))
        ordenado = base.sort_values("_t", kind="stable")
        junto = pd.merge_asof(
            ordenado,
            ev,
            left_on="_t",
            right_on=tempo_evento,
            by=chaves or None,
            allow_exact_matches=False,
        )
        return junto.sort_values("_linha")[["_soma", "_n"]].fillna(0.0).to_numpy()

    total = acumulado_ate(consultas[tempo_consulta])
    if janela is not None:
        # Eventos anteriores ao início da janela: (t - janela) exclusivo,
        # portanto o que sobra é [t - janela, t).
        total = total - acumulado_ate(consultas[tempo_consulta] - janela)

    return pd.DataFrame({"soma": total[:, 0], "n": total[:, 1]}, index=consultas.index)


def _media(agregado: pd.DataFrame) -> pd.Series:
    """Média = soma / n, com NaN quando não há histórico."""
    return agregado["soma"] / agregado["n"].where(agregado["n"] > 0)


def _media_suavizada(agregado: pd.DataFrame, previa: np.ndarray) -> np.ndarray:
    """Média puxada para a prévia quando há poucos eventos.

    Equivale a somar `PESO_DA_PREVIA` eventos fictícios com o valor da prévia:
    um vendedor com 1 atraso em 1 entrega não vira "100% de atraso", e quem não
    tem histórico recebe a média geral. Assim o modelo não precisa da contagem
    de eventos como variável (ela cresce com o tempo; ver README).
    """
    soma, n = agregado["soma"].to_numpy(), agregado["n"].to_numpy()
    return (soma + PESO_DA_PREVIA * previa) / (n + PESO_DA_PREVIA)


def _resumo_dos_itens(itens: pd.DataFrame, produtos: pd.DataFrame) -> pd.DataFrame:
    """Uma linha por pedido: valores, peso, volume, vendedor e categoria principais."""
    it = itens.merge(
        produtos[
            [
                "product_id",
                "product_category_name",
                "product_weight_g",
                "product_length_cm",
                "product_height_cm",
                "product_width_cm",
            ]
        ],
        on="product_id",
        how="left",
    )
    it["volume_cm3"] = it["product_length_cm"] * it["product_height_cm"] * it["product_width_cm"]

    grupos = it.groupby("order_id")
    resumo = pd.DataFrame(
        {
            "n_itens": grupos.size(),
            "n_vendedores": grupos["seller_id"].nunique(),
            "valor_produtos": grupos["price"].sum(),
            "valor_frete": grupos["freight_value"].sum(),
            "peso_total_g": grupos["product_weight_g"].sum(min_count=1),
            "volume_total_cm3": grupos["volume_cm3"].sum(min_count=1),
        }
    )
    # Pedidos com mais de um vendedor (1,3%) são representados pelo item de maior valor.
    principal = (
        it.sort_values(["order_id", "price"], ascending=[True, False], kind="stable")
        .drop_duplicates("order_id")
        .set_index("order_id")[["seller_id", "product_category_name"]]
        .rename(
            columns={"seller_id": "id_vendedor", "product_category_name": "categoria_principal"}
        )
    )
    return resumo.join(principal).reset_index()


def _resumo_dos_pagamentos(pagamentos: pd.DataFrame) -> pd.DataFrame:
    """Uma linha por pedido: forma de pagamento de maior valor e nº de parcelas."""
    principal = (
        pagamentos.sort_values(
            ["order_id", "payment_value"], ascending=[True, False], kind="stable"
        )
        .drop_duplicates("order_id")[["order_id", "payment_type"]]
        .rename(columns={"payment_type": "tipo_pagamento"})
    )
    parcelas = pagamentos.groupby("order_id")["payment_installments"].max().rename("parcelas")
    return principal.merge(parcelas.reset_index(), on="order_id")


def centroides_por_cep(geolocalizacao: pd.DataFrame) -> pd.DataFrame:
    """Coordenada mediana de cada prefixo de CEP, descartando pontos fora do Brasil."""
    geo = geolocalizacao[
        geolocalizacao["geolocation_lat"].between(*LAT_BRASIL)
        & geolocalizacao["geolocation_lng"].between(*LNG_BRASIL)
    ]
    return (
        geo.groupby("geolocation_zip_code_prefix")[["geolocation_lat", "geolocation_lng"]]
        .median()
        .rename(columns={"geolocation_lat": "lat", "geolocation_lng": "lng"})
    )


def montar_tabela(tabelas: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Constrói a tabela analítica a partir das tabelas brutas da Olist."""
    pedidos = tabelas["pedidos"].rename(
        columns={
            "order_purchase_timestamp": "data_compra",
            "order_delivered_carrier_date": "data_postagem",
            "order_delivered_customer_date": "data_entrega",
            "order_estimated_delivery_date": "data_prometida",
        }
    )

    # ---- Todos os pedidos (qualquer status): base para volume e histórico ----
    todos = pedidos[
        [
            "order_id",
            "customer_id",
            "order_status",
            "data_compra",
            "data_postagem",
            "data_entrega",
            "data_prometida",
        ]
    ].merge(_resumo_dos_itens(tabelas["itens"], tabelas["produtos"]), on="order_id", how="left")

    clientes = tabelas["clientes"].rename(
        columns={"customer_state": "uf_cliente", "customer_zip_code_prefix": "cep_cliente"}
    )
    vendedores = tabelas["vendedores"].rename(
        columns={
            "seller_id": "id_vendedor",
            "seller_state": "uf_vendedor",
            "seller_zip_code_prefix": "cep_vendedor",
        }
    )
    todos = todos.merge(
        clientes[["customer_id", "uf_cliente", "cep_cliente"]], on="customer_id", how="left"
    )
    todos = todos.merge(
        vendedores[["id_vendedor", "uf_vendedor", "cep_vendedor"]], on="id_vendedor", how="left"
    )

    todos["dias_ate_postagem"] = ((todos["data_postagem"] - todos["data_compra"]) / _DIA).clip(
        lower=0
    )
    todos["dias_ate_entrega"] = (todos["data_entrega"] - todos["data_compra"]) / _DIA
    todos["um"] = 1.0

    # ---- Pedidos entregues: universo do modelo ----
    entregue = (todos["order_status"] == "delivered") & todos["data_entrega"].notna()
    df = todos[entregue & todos["id_vendedor"].notna()].copy()

    # A data prometida vem como meia-noite do último dia do prazo, então a
    # comparação é feita por dia: entregar durante o dia prometido não é atraso.
    df[ALVO] = (df["data_entrega"].dt.normalize() > df["data_prometida"].dt.normalize()).astype(
        "int8"
    )

    # ---- Variáveis do próprio pedido ----
    df["prazo_prometido_dias"] = (df["data_prometida"] - df["data_compra"]) / _DIA
    df["dia_semana"] = df["data_compra"].dt.dayofweek
    df["hora_compra"] = df["data_compra"].dt.hour
    df["frete_sobre_valor"] = df["valor_frete"] / df["valor_produtos"].where(
        df["valor_produtos"] > 0
    )

    df = df.merge(_resumo_dos_pagamentos(tabelas["pagamentos"]), on="order_id", how="left")
    df["tipo_pagamento"] = df["tipo_pagamento"].fillna("desconhecido")
    df["categoria_principal"] = df["categoria_principal"].fillna("desconhecida")

    # ---- Geografia ----
    centroides = centroides_por_cep(tabelas["geolocalizacao"])
    cli = centroides.reindex(df["cep_cliente"]).to_numpy()
    ven = centroides.reindex(df["cep_vendedor"]).to_numpy()
    df["distancia_km"] = distancia_km(cli[:, 0], cli[:, 1], ven[:, 0], ven[:, 1])
    df["mesma_uf"] = (df["uf_cliente"] == df["uf_vendedor"]).astype("int8")
    df["regiao_cliente"] = df["uf_cliente"].map(UF_PARA_REGIAO)

    # ---- Histórico (somente o que já era conhecido no instante da compra) ----
    # O resultado de um pedido (atrasou ou não, dias até a entrega) só é
    # conhecido na data da entrega; o tempo de postagem, na data da postagem.
    df["cep3_cliente"] = df["cep_cliente"].str[:3]
    desfechos = df[
        [
            "id_vendedor",
            "uf_vendedor",
            "uf_cliente",
            "cep3_cliente",
            "data_entrega",
            ALVO,
            "dias_ate_entrega",
        ]
    ].copy()
    desfechos[ALVO] = desfechos[ALVO].astype("float64")
    rota = ["uf_vendedor", "uf_cliente"]

    def antes(eventos, tempo_evento, valor, chaves=None, dias=None):
        return agregado_anterior(
            df,
            eventos,
            tempo_consulta="data_compra",
            tempo_evento=tempo_evento,
            valor=valor,
            chaves=chaves,
            janela=None if dias is None else pd.Timedelta(days=dias),
        )

    # Prévia: a média geral conhecida até o instante da compra. Serve de ponto
    # de partida para quem tem pouco histórico (ver `_media_suavizada`).
    previa_taxa = _media(antes(desfechos, "data_entrega", ALVO)).to_numpy()
    previa_dias = _media(antes(desfechos, "data_entrega", "dias_ate_entrega")).to_numpy()

    # Vendedor
    hist_vendedor = antes(desfechos, "data_entrega", ALVO, ["id_vendedor"])
    df["vendedor_n_entregas"] = hist_vendedor["n"].to_numpy()
    df["vendedor_taxa_atraso"] = _media_suavizada(hist_vendedor, previa_taxa)
    df["vendedor_dias_postagem"] = _media(
        antes(todos, "data_postagem", "dias_ate_postagem", ["id_vendedor"])
    ).to_numpy()
    df["vendedor_pedidos_30d"] = antes(todos, "data_compra", "um", ["id_vendedor"], dias=30)[
        "n"
    ].to_numpy()

    # Rota (UF do vendedor -> UF do cliente) e destino (3 primeiros dígitos do
    # CEP do cliente: uma região bem menor que a UF). Usam todo o passado, que
    # reage devagar e oscila pouco.
    for nome, chaves in [("rota", rota), ("destino", ["cep3_cliente"])]:
        atrasos = antes(desfechos, "data_entrega", ALVO, chaves)
        dias = antes(desfechos, "data_entrega", "dias_ate_entrega", chaves)
        df[f"{nome}_n_entregas"] = atrasos["n"].to_numpy()
        df[f"{nome}_taxa_atraso"] = _media_suavizada(atrasos, previa_taxa)
        df[f"{nome}_dias_entrega"] = _media_suavizada(dias, previa_dias)
        # Folga: quantos dias o prazo prometido tem a mais que a entrega típica.
        df[f"folga_prazo_{nome}_dias"] = df["prazo_prometido_dias"] - df[f"{nome}_dias_entrega"]

    # ---- Variáveis rejeitadas no backtest (ver README) ----
    # Ficam na tabela para que a comparação seja reproduzível: as contagens
    # acumuladas acima (`*_n_entregas`) e as variáveis da 1ª versão abaixo.
    rota_90d = antes(desfechos, "data_entrega", ALVO, rota, dias=90)
    df["rota_n_entregas_90d"] = rota_90d["n"].to_numpy()
    df["rota_taxa_atraso_90d"] = _media(rota_90d).to_numpy()
    df["rota_dias_entrega_90d"] = _media(
        antes(desfechos, "data_entrega", "dias_ate_entrega", rota, dias=90)
    ).to_numpy()
    df["folga_prazo_rota_90d_dias"] = df["prazo_prometido_dias"] - df["rota_dias_entrega_90d"]
    df["geral_taxa_atraso_30d"] = _media(antes(desfechos, "data_entrega", ALVO, dias=30)).to_numpy()

    # Pressão de volume: pedidos da última semana contra a média semanal das 4 anteriores.
    # É uma razão (e não a contagem) para não depender do crescimento da empresa.
    semana = antes(todos, "data_compra", "um", dias=7)["n"].to_numpy()
    mes = antes(todos, "data_compra", "um", dias=35)["n"].to_numpy()
    anteriores = (mes - semana) / 4
    df["pressao_volume"] = semana / np.where(anteriores > 0, anteriores, np.nan)

    colunas_de_apoio = [
        "customer_id",
        "order_status",
        "cep_cliente",
        "cep_vendedor",
        "cep3_cliente",
        "um",
    ]
    return (
        df.drop(columns=colunas_de_apoio)
        .sort_values("data_compra", kind="stable")
        .reset_index(drop=True)
    )
