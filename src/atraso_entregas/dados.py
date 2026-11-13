"""Download e leitura das tabelas públicas da Olist.

Uso:
    python -m atraso_entregas.dados
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

import pandas as pd

from atraso_entregas.config import DIR_DADOS_BRUTOS

URL_BASE = "https://raw.githubusercontent.com/olist/work-at-olist-data/master/datasets"

ARQUIVOS = {
    "pedidos": "olist_orders_dataset.csv",
    "itens": "olist_order_items_dataset.csv",
    "pagamentos": "olist_order_payments_dataset.csv",
    "clientes": "olist_customers_dataset.csv",
    "vendedores": "olist_sellers_dataset.csv",
    "produtos": "olist_products_dataset.csv",
    "geolocalizacao": "olist_geolocation_dataset.csv",
}

COLUNAS_DE_DATA = {
    "pedidos": [
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ],
}

# Prefixos de CEP têm zeros à esquerda: precisam ser lidos como texto.
COLUNAS_DE_TEXTO = {
    "clientes": {"customer_zip_code_prefix": "string"},
    "vendedores": {"seller_zip_code_prefix": "string"},
    "geolocalizacao": {"geolocation_zip_code_prefix": "string"},
}


def baixar(destino: Path = DIR_DADOS_BRUTOS, forcar: bool = False) -> None:
    """Baixa os CSVs da Olist que ainda não existem em `destino`."""
    destino.mkdir(parents=True, exist_ok=True)
    for arquivo in ARQUIVOS.values():
        caminho = destino / arquivo
        if caminho.exists() and not forcar:
            print(f"já existe  {arquivo}")
            continue
        print(f"baixando   {arquivo}")
        parcial = caminho.with_suffix(".parcial")
        urllib.request.urlretrieve(f"{URL_BASE}/{arquivo}", parcial)
        parcial.replace(caminho)


def carregar(origem: Path = DIR_DADOS_BRUTOS) -> dict[str, pd.DataFrame]:
    """Lê todas as tabelas em um dicionário nome -> DataFrame."""
    faltando = [a for a in ARQUIVOS.values() if not (origem / a).exists()]
    if faltando:
        raise FileNotFoundError(
            f"Arquivos ausentes em {origem}: {faltando}. "
            "Rode `python -m atraso_entregas.dados` para baixá-los."
        )
    tabelas = {
        nome: pd.read_csv(
            origem / arquivo,
            parse_dates=COLUNAS_DE_DATA.get(nome),
            dtype=COLUNAS_DE_TEXTO.get(nome),
        )
        for nome, arquivo in ARQUIVOS.items()
    }
    # Clientes e vendedores vêm sem o zero à esquerda ("9790"), a geolocalização
    # vem com ele ("09790"). Sem padronizar, 1 em cada 4 CEPs não casa.
    for nome, colunas in COLUNAS_DE_TEXTO.items():
        for coluna in colunas:
            tabelas[nome][coluna] = tabelas[nome][coluna].str.zfill(5)
    return tabelas


if __name__ == "__main__":
    baixar()
