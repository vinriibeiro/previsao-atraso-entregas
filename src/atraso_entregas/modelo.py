"""Definição dos modelos: variáveis de entrada, baseline e modelo final."""

from __future__ import annotations

import numpy as np
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, QuantileTransformer

from atraso_entregas.config import SEMENTE

CATEGORICAS = ["uf_cliente", "uf_vendedor", "categoria_principal", "tipo_pagamento"]

_PEDIDO = [
    "prazo_prometido_dias",
    "distancia_km",
    "mesma_uf",
    "n_itens",
    "n_vendedores",
    "valor_produtos",
    "valor_frete",
    "frete_sobre_valor",
    "peso_total_g",
    "volume_total_cm3",
    "parcelas",
    "dia_semana",
    "hora_compra",
]
_VENDEDOR = ["vendedor_taxa_atraso", "vendedor_dias_postagem", "vendedor_pedidos_30d"]
_ROTA = ["rota_taxa_atraso", "rota_dias_entrega", "folga_prazo_rota_dias"]
_DESTINO = ["destino_taxa_atraso", "destino_dias_entrega", "folga_prazo_destino_dias"]

# Conjunto final, escolhido no backtest (`python -m atraso_entregas.backtest`).
NUMERICAS = _PEDIDO + _VENDEDOR + _ROTA + _DESTINO
VARIAVEIS = CATEGORICAS + NUMERICAS

# ---- Variáveis rejeitadas no backtest (mantidas só para reproduzir a comparação) ----
# Contagens acumuladas: crescem com o calendário e funcionam como um relógio
# escondido, que deixa o modelo decorar períodos em vez de aprender causas.
CONTAGENS_ACUMULADAS = ["vendedor_n_entregas", "rota_n_entregas", "destino_n_entregas"]
# 1ª versão: histórico da rota em janela curta e duas variáveis de "momento da operação".
NUMERICAS_PRIMEIRA_VERSAO = (
    _PEDIDO
    + _VENDEDOR
    + ["vendedor_n_entregas"]
    + [
        "rota_n_entregas_90d",
        "rota_taxa_atraso_90d",
        "rota_dias_entrega_90d",
        "folga_prazo_rota_90d_dias",
    ]
    + ["geral_taxa_atraso_30d", "pressao_volume"]
)

# O mês da compra fica de fora pelo mesmo motivo das contagens: com menos de dois
# anos de dados, o modelo decoraria meses específicos em vez de aprender a causa.

# Hiperparâmetros conservadores, usados onde não há busca (backtest).
HIPERPARAMETROS_PADRAO = {
    "learning_rate": 0.03,
    "max_iter": 300,
    "max_leaf_nodes": 15,
    "min_samples_leaf": 100,
    "l2_regularization": 1.0,
}


def criar_regressao_logistica(numericas: list[str] = NUMERICAS) -> Pipeline:
    """Modelo linear: simples, estável e fácil de explicar."""
    preparo_numericas = Pipeline(
        [
            ("imputar", SimpleImputer(strategy="median", add_indicator=True)),
            (
                "normalizar",
                QuantileTransformer(
                    n_quantiles=200, output_distribution="normal", random_state=SEMENTE
                ),
            ),
        ]
    )
    preparo = ColumnTransformer(
        [
            (
                "cat",
                OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=50),
                CATEGORICAS,
            ),
            ("num", preparo_numericas, list(numericas)),
        ]
    )
    return Pipeline([("preparo", preparo), ("modelo", LogisticRegression(max_iter=2000))])


def criar_gradient_boosting(numericas: list[str] = NUMERICAS, **hiperparametros) -> Pipeline:
    """Gradient boosting por histogramas: captura interações que o modelo linear não vê.

    Lida nativamente com valores ausentes (CEP sem coordenada, vendedor que
    nunca postou) e com variáveis categóricas. Categorias nunca vistas no
    treino viram ausentes. Não usa peso de classe, para que a saída continue
    sendo uma probabilidade.
    """
    preparo = ColumnTransformer(
        [
            (
                "cat",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan),
                CATEGORICAS,
            ),
            ("num", "passthrough", list(numericas)),
        ]
    )
    modelo = HistGradientBoostingClassifier(
        categorical_features=list(range(len(CATEGORICAS))),
        # a parada antecipada do sklearn sorteia a validação, ignorando o tempo
        early_stopping=False,
        random_state=SEMENTE,
        **{**HIPERPARAMETROS_PADRAO, **hiperparametros},
    )
    return Pipeline([("preparo", preparo), ("modelo", modelo)])


def criar_mistura(numericas: list[str] = NUMERICAS, **hiperparametros) -> VotingClassifier:
    """Modelo final: média das probabilidades da regressão logística e do boosting.

    No backtest os dois se alternam na frente, com diferenças pequenas; a média
    dos dois teve o melhor resultado médio e o melhor pior caso.
    """
    return VotingClassifier(
        [
            ("logistica", criar_regressao_logistica(numericas)),
            ("boosting", criar_gradient_boosting(numericas, **hiperparametros)),
        ],
        voting="soft",
    )


ESPACO_DE_BUSCA = {
    "modelo__learning_rate": stats.loguniform(0.01, 0.2),
    "modelo__max_iter": stats.randint(100, 400),
    "modelo__max_leaf_nodes": stats.randint(6, 32),
    "modelo__min_samples_leaf": stats.randint(20, 300),
    "modelo__l2_regularization": stats.loguniform(1e-2, 30),
    "modelo__max_features": stats.uniform(0.4, 0.6),
}


def buscar_hiperparametros(X, y, n_iter: int = 15, n_divisoes: int = 5) -> RandomizedSearchCV:
    """Busca aleatória dos hiperparâmetros do boosting, com validação cruzada temporal.

    `X` precisa estar ordenado pela data da compra: cada divisão treina no
    passado e avalia no trecho seguinte, sem embaralhar. A métrica é a ROC-AUC
    porque a taxa de atraso muda muito entre as divisões, e o PR-AUC depende dela.
    """
    busca = RandomizedSearchCV(
        criar_gradient_boosting(),
        ESPACO_DE_BUSCA,
        n_iter=n_iter,
        scoring="roc_auc",
        cv=TimeSeriesSplit(n_splits=n_divisoes),
        random_state=SEMENTE,
        refit=False,
    )
    return busca.fit(X, y)
