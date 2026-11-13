"""Tabelas sintéticas no formato da Olist, pequenas o bastante para rodar em segundos."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="session")
def tabelas() -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(0)
    n = 1500
    ufs = ["SP", "RJ", "MG", "BA", "RS"]
    ceps = {"SP": "01310", "RJ": "20040", "MG": "30130", "BA": "40020", "RS": "90010"}
    coordenadas = {
        "SP": (-23.56, -46.65),
        "RJ": (-22.90, -43.18),
        "MG": (-19.93, -43.94),
        "BA": (-12.97, -38.50),
        "RS": (-30.03, -51.23),
    }

    vendedores = pd.DataFrame({"seller_id": [f"v{i}" for i in range(12)]})
    vendedores["seller_state"] = rng.choice(ufs, len(vendedores))
    vendedores["seller_zip_code_prefix"] = vendedores["seller_state"].map(ceps)
    vendedores["seller_city"] = "cidade"

    clientes = pd.DataFrame({"customer_id": [f"c{i}" for i in range(n)]})
    clientes["customer_unique_id"] = clientes["customer_id"]
    clientes["customer_state"] = rng.choice(ufs, n)
    clientes["customer_zip_code_prefix"] = clientes["customer_state"].map(ceps)
    clientes["customer_city"] = "cidade"

    compra = pd.Timestamp("2017-01-01") + pd.to_timedelta(np.sort(rng.uniform(0, 600, n)), unit="D")
    prazo = rng.integers(10, 30, n)
    dias_entrega = rng.gamma(4.0, 3.5, n) + 1
    prometida = (compra + pd.to_timedelta(prazo, unit="D")).normalize()
    pedidos = pd.DataFrame(
        {
            "order_id": [f"p{i}" for i in range(n)],
            "customer_id": clientes["customer_id"],
            "order_status": "delivered",
            "order_purchase_timestamp": compra,
            "order_approved_at": compra + pd.Timedelta(hours=1),
            "order_delivered_carrier_date": compra + pd.to_timedelta(dias_entrega * 0.3, unit="D"),
            "order_delivered_customer_date": compra + pd.to_timedelta(dias_entrega, unit="D"),
            "order_estimated_delivery_date": prometida,
        }
    )
    # Alguns pedidos cancelados, sem datas de postagem e entrega.
    cancelados = pedidos.index[::50]
    pedidos.loc[cancelados, "order_status"] = "canceled"
    pedidos.loc[cancelados, ["order_delivered_carrier_date", "order_delivered_customer_date"]] = (
        pd.NaT
    )

    produtos = pd.DataFrame(
        {
            "product_id": [f"prod{i}" for i in range(40)],
            "product_category_name": rng.choice(["moveis", "eletronicos", "brinquedos", None], 40),
            "product_weight_g": rng.uniform(100, 5000, 40),
            "product_length_cm": rng.uniform(10, 60, 40),
            "product_height_cm": rng.uniform(5, 40, 40),
            "product_width_cm": rng.uniform(10, 50, 40),
        }
    )
    itens = pd.DataFrame(
        {
            "order_id": pedidos["order_id"],
            "order_item_id": 1,
            "product_id": rng.choice(produtos["product_id"], n),
            "seller_id": rng.choice(vendedores["seller_id"], n),
            "shipping_limit_date": compra + pd.Timedelta(days=3),
            "price": rng.uniform(20, 500, n).round(2),
            "freight_value": rng.uniform(8, 60, n).round(2),
        }
    )
    pagamentos = pd.DataFrame(
        {
            "order_id": pedidos["order_id"],
            "payment_sequential": 1,
            "payment_type": rng.choice(["credit_card", "boleto"], n),
            "payment_installments": rng.integers(1, 10, n),
            "payment_value": itens["price"] + itens["freight_value"],
        }
    )
    geolocalizacao = pd.DataFrame(
        {
            "geolocation_zip_code_prefix": [ceps[uf] for uf in ufs] + ["01310"],
            "geolocation_lat": [coordenadas[uf][0] for uf in ufs]
            + [45.0],  # último ponto: fora do Brasil
            "geolocation_lng": [coordenadas[uf][1] for uf in ufs] + [10.0],
            "geolocation_city": "cidade",
            "geolocation_state": ufs + ["SP"],
        }
    )
    return {
        "pedidos": pedidos,
        "itens": itens,
        "pagamentos": pagamentos,
        "clientes": clientes,
        "vendedores": vendedores,
        "produtos": produtos,
        "geolocalizacao": geolocalizacao,
    }


@pytest.fixture(scope="session")
def tabela(tabelas) -> pd.DataFrame:
    from atraso_entregas.features import montar_tabela

    return montar_tabela(tabelas)
