"""Backtest com várias datas de origem, usando só o período de desenvolvimento.

Uso:
    python -m atraso_entregas.backtest

Uma única validação mostra o desempenho em um único regime. Aqui o treino é
repetido em várias datas ("origens"): em cada uma, o modelo aprende com o que
era conhecido até ali e é avaliado nos três meses seguintes. É neste backtest
que conjuntos de variáveis e modelos são comparados; o teste fica de fora.
"""

from __future__ import annotations

import json

import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from atraso_entregas import config, graficos
from atraso_entregas.dados import carregar
from atraso_entregas.features import ALVO, montar_tabela
from atraso_entregas.modelo import (
    CATEGORICAS,
    CONTAGENS_ACUMULADAS,
    NUMERICAS,
    NUMERICAS_PRIMEIRA_VERSAO,
    criar_gradient_boosting,
    criar_mistura,
    criar_regressao_logistica,
)
from atraso_entregas.validacao import separar_periodos

MISTURA = "Mistura (modelo final)"
LOGISTICA = "Regressão logística"
BOOSTING = "Gradient boosting"
LOGISTICA_CONTAGENS = "Regressão logística + contagens acumuladas"
BOOSTING_CONTAGENS = "Gradient boosting + contagens acumuladas"
PRIMEIRA_VERSAO = "Gradient boosting, 1ª versão das variáveis"
REGRA = "Regra: atraso histórico da rota"

CANDIDATOS = {
    MISTURA: (criar_mistura, NUMERICAS),
    LOGISTICA: (criar_regressao_logistica, NUMERICAS),
    BOOSTING: (criar_gradient_boosting, NUMERICAS),
    LOGISTICA_CONTAGENS: (criar_regressao_logistica, NUMERICAS + CONTAGENS_ACUMULADAS),
    BOOSTING_CONTAGENS: (criar_gradient_boosting, NUMERICAS + CONTAGENS_ACUMULADAS),
    PRIMEIRA_VERSAO: (criar_gradient_boosting, NUMERICAS_PRIMEIRA_VERSAO),
}

ORIGENS = pd.date_range("2017-08-01", config.FIM_TREINO, freq="MS")
MESES_DE_AVALIACAO = 3


def executar(tabela: pd.DataFrame | None = None) -> pd.DataFrame:
    if tabela is None:
        tabela = montar_tabela(carregar())
    # Só o que estaria disponível na data de corte: o teste não entra aqui.
    dev = separar_periodos(tabela).desenvolvimento

    linhas = []
    for origem in ORIGENS:
        # Em cada origem vale a mesma regra do projeto: só entra no treino o
        # pedido cujo prazo prometido já tinha vencido.
        treino = dev[(dev["data_compra"] < origem) & (dev["data_prometida"] < origem)]
        fim = origem + pd.DateOffset(months=MESES_DE_AVALIACAO)
        avaliacao = dev[(dev["data_compra"] >= origem) & (dev["data_compra"] < fim)]
        y = avaliacao[ALVO]

        notas = {REGRA: avaliacao["rota_taxa_atraso"].to_numpy()}
        for nome, (criar, numericas) in CANDIDATOS.items():
            colunas = CATEGORICAS + list(numericas)
            modelo = criar(numericas).fit(treino[colunas], treino[ALVO])
            notas[nome] = modelo.predict_proba(avaliacao[colunas])[:, 1]

        for nome, nota in notas.items():
            pr_auc = float(average_precision_score(y, nota))
            linhas.append(
                {
                    "origem": origem.strftime("%Y-%m"),
                    "candidato": nome,
                    "n_treino": int(len(treino)),
                    "n_avaliacao": int(len(avaliacao)),
                    "taxa_atraso": float(y.mean()),
                    "roc_auc": float(roc_auc_score(y, nota)),
                    # Lift: quantas vezes o PR-AUC supera o de um modelo sem informação.
                    "lift": pr_auc / float(y.mean()),
                    "previsto_medio": float(nota.mean()),
                }
            )
        print(f"origem {origem:%Y-%m} concluída", flush=True)

    resultado = pd.DataFrame(linhas)
    resumo = resultado.groupby("candidato")[["roc_auc", "lift"]].agg(["mean", "min"]).round(3)
    resumo.columns = [f"{metrica}_{estatistica}" for metrica, estatistica in resumo.columns]
    resumo = resumo.sort_values("roc_auc_mean", ascending=False)

    config.DIR_RELATORIOS.mkdir(parents=True, exist_ok=True)
    (config.DIR_RELATORIOS / "backtest.json").write_text(
        json.dumps(
            {
                "por_origem": resultado.to_dict(orient="records"),
                "resumo": resumo.to_dict(orient="index"),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    graficos.backtest(
        resultado,
        {
            MISTURA: graficos.AZUL,
            LOGISTICA_CONTAGENS: graficos.LARANJA,
            BOOSTING_CONTAGENS: graficos.VERDE_AGUA,
        },
        config.DIR_FIGURAS / "backtest.png",
    )
    print(resumo.to_string())
    return resultado


if __name__ == "__main__":
    executar()
