"""Treina, avalia e salva o modelo de ponta a ponta.

Uso:
    python -m atraso_entregas.treinar            # busca completa (alguns minutos)
    python -m atraso_entregas.treinar --rapido   # busca curta, para testar o fluxo

Protocolo (simula o treino feito em `config.DATA_DE_CORTE`):
    1. hiperparâmetros do boosting: validação cruzada temporal dentro do treino;
    2. corte por custo: escolhido na validação, com o modelo ajustado só no treino;
    3. modelo final (mistura): reajustado em treino + validação;
    4. teste: pedidos comprados depois do corte.

A escolha das variáveis e do tipo de modelo é feita antes, no backtest
(`python -m atraso_entregas.backtest`), sem usar o teste.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.inspection import permutation_importance

from atraso_entregas import avaliacao, config, graficos
from atraso_entregas.dados import carregar
from atraso_entregas.features import ALVO, montar_tabela
from atraso_entregas.modelo import (
    CATEGORICAS,
    NUMERICAS,
    VARIAVEIS,
    buscar_hiperparametros,
    criar_mistura,
)
from atraso_entregas.validacao import separar_periodos

REGRA = "Regra: atraso histórico da rota"
LOGISTICA = "Regressão logística"
BOOSTING = "Gradient boosting"
MISTURA = "Mistura (modelo final)"

FRACAO_SINALIZADA = 0.10  # capacidade da política principal: 10% dos pedidos de cada semana


def _resumo_do_periodo(df: pd.DataFrame) -> dict:
    return {
        "pedidos": len(df),
        "atrasos": int(df[ALVO].sum()),
        "taxa_atraso": float(df[ALVO].mean()),
        "de": str(df["data_compra"].min().date()),
        "ate": str(df["data_compra"].max().date()),
    }


def _regra_da_rota(df: pd.DataFrame) -> np.ndarray:
    """Baseline sem ML: usar a taxa de atraso histórica da rota como nota de risco."""
    return df["rota_taxa_atraso"].to_numpy()


def _politica_de_capacidade(teste: pd.DataFrame, prob: np.ndarray, fracao: float) -> dict:
    sinalizado = avaliacao.topo_por_semana(teste["data_compra"], prob, fracao)
    decisao = avaliacao.resultado_da_decisao(teste[ALVO], sinalizado)
    decisao["custo_maximo_da_acao"] = avaliacao.custo_maximo_da_acao(decisao["precisao"])
    return decisao


def executar(rapido: bool = False) -> dict:
    premissas = config.PREMISSAS

    print("1/6 montando a tabela analítica")
    tabela = montar_tabela(carregar())
    periodos = separar_periodos(tabela)
    treino, validacao, teste, dev = (
        periodos.treino,
        periodos.validacao,
        periodos.teste,
        periodos.desenvolvimento,
    )
    y_validacao, y_teste = validacao[ALVO], teste[ALVO]

    print("2/6 buscando hiperparâmetros do boosting (validação cruzada temporal no treino)")
    busca = buscar_hiperparametros(treino[VARIAVEIS], treino[ALVO], n_iter=3 if rapido else 15)
    hiperparametros = {k.removeprefix("modelo__"): v for k, v in busca.best_params_.items()}

    print("3/6 validação: modelo ajustado só no treino, para escolher o corte por custo")
    mistura_treino = criar_mistura(**hiperparametros).fit(treino[VARIAVEIS], treino[ALVO])
    prob_validacao = mistura_treino.predict_proba(validacao[VARIAVEIS])[:, 1]
    corte = avaliacao.escolher_corte(y_validacao, prob_validacao, premissas)

    print("4/6 reajustando em treino + validação e avaliando no teste")
    mistura = criar_mistura(**hiperparametros).fit(dev[VARIAVEIS], dev[ALVO])
    prob_teste = {
        REGRA: _regra_da_rota(teste),
        # os dois componentes da mistura, já ajustados dentro dela
        LOGISTICA: mistura.named_estimators_["logistica"].predict_proba(teste[VARIAVEIS])[:, 1],
        BOOSTING: mistura.named_estimators_["boosting"].predict_proba(teste[VARIAVEIS])[:, 1],
        MISTURA: mistura.predict_proba(teste[VARIAVEIS])[:, 1],
    }
    prob = prob_teste[MISTURA]
    politica_capacidade = {
        f"{fracao:.0%}": _politica_de_capacidade(teste, prob, fracao)
        for fracao in (0.05, 0.10, 0.20)
    }
    politica_corte = avaliacao.resultado_da_decisao(y_teste, prob >= corte, premissas)

    print("5/6 análise de erros e importância das variáveis")
    analise = teste[["data_compra", "regiao_cliente", ALVO]].copy()
    analise["prob"] = prob
    analise["sinalizado"] = avaliacao.topo_por_semana(teste["data_compra"], prob, FRACAO_SINALIZADA)
    por_mes = avaliacao.por_grupo(
        analise, analise["data_compra"].dt.strftime("%Y-%m"), ALVO, "prob", "sinalizado"
    )
    por_regiao = avaliacao.por_grupo(analise, "regiao_cliente", ALVO, "prob", "sinalizado")
    faixas = avaliacao.calibracao_por_faixa(y_teste, prob)

    permutacao = permutation_importance(
        mistura,
        teste[VARIAVEIS],
        y_teste,
        scoring="roc_auc",
        n_repeats=3 if rapido else 10,
        random_state=config.SEMENTE,
    )
    importancias = pd.Series(permutacao.importances_mean, index=VARIAVEIS).sort_values(
        ascending=False
    )

    print("6/6 salvando modelo, relatório e figuras")
    resultados = {
        "gerado_em": datetime.now().isoformat(timespec="seconds"),
        "busca_rapida": rapido,
        "versoes": {
            "scikit-learn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
        "recortes": {
            "inicio_treino": config.INICIO_TREINO,
            "fim_treino": config.FIM_TREINO,
            "data_de_corte": config.DATA_DE_CORTE,
            "pedidos_imaturos_descartados": periodos.n_imaturos,
        },
        "periodos": {
            "treino": _resumo_do_periodo(treino),
            "validacao": _resumo_do_periodo(validacao),
            "teste": _resumo_do_periodo(teste),
        },
        "premissas": asdict(premissas) | {"precisao_minima": premissas.precisao_minima},
        "hiperparametros_boosting": {k: float(v) for k, v in hiperparametros.items()},
        "roc_auc_validacao_cruzada": float(busca.best_score_),
        "validacao": avaliacao.metricas(y_validacao, prob_validacao),
        "teste": {nome: avaliacao.metricas(y_teste, p) for nome, p in prob_teste.items()},
        "politica_capacidade_teste": politica_capacidade,
        "politica_corte_por_custo": {
            "corte": corte,
            "validacao": avaliacao.resultado_da_decisao(
                y_validacao, prob_validacao >= corte, premissas
            ),
            "teste": politica_corte,
        },
        "teste_por_mes": por_mes.to_dict(orient="records"),
        "teste_por_regiao": por_regiao.to_dict(orient="records"),
        "calibracao_teste": faixas.to_dict(orient="records"),
        "importancia_permutacao": importancias.round(5).to_dict(),
    }

    config.DIR_RELATORIOS.mkdir(parents=True, exist_ok=True)
    (config.DIR_RELATORIOS / "resultados.json").write_text(
        json.dumps(resultados, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    config.DIR_MODELOS.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "modelo": mistura,
            "variaveis": VARIAVEIS,
            # Notas do modelo nos pedidos de teste: o app usa para dizer em que
            # faixa de risco um pedido cai (ex.: "entre os 10% mais arriscados").
            "notas_de_referencia": np.quantile(prob, np.linspace(0, 1, 101)),
            # "Pedido típico": mediana (números) e moda (categorias) do desenvolvimento.
            # O app usa como ponto de comparação para explicar cada previsão.
            "pedido_tipico": {
                **dev[NUMERICAS].median().to_dict(),
                **dev[CATEGORICAS].mode().iloc[0].to_dict(),
            },
            "treinado_ate": config.DATA_DE_CORTE,
            "versoes": resultados["versoes"],
        },
        config.DIR_MODELOS / "modelo.joblib",
        compress=3,
    )
    # Amostra do teste para o app: pedidos reais que o modelo nunca viu.
    exemplos = teste.groupby(ALVO, group_keys=False).sample(n=150, random_state=config.SEMENTE)
    exemplos[["order_id", "data_compra", ALVO, *VARIAVEIS]].to_parquet(
        config.DIR_MODELOS / "exemplos.parquet", index=False
    )

    figuras = config.DIR_FIGURAS
    graficos.atraso_por_mes(
        tabela, config.FIM_TREINO, config.DATA_DE_CORTE, figuras / "atraso_por_mes.png"
    )
    graficos.comparacao_de_modelos(
        {nome: m["roc_auc"] for nome, m in resultados["teste"].items()},
        MISTURA,
        figuras / "comparacao_modelos.png",
    )
    graficos.precisao_no_topo(
        {
            "Validação (mar-mai/2018)": avaliacao.precisao_no_topo(y_validacao, prob_validacao),
            "Teste (jun-ago/2018)": avaliacao.precisao_no_topo(y_teste, prob),
        },
        premissas.precisao_minima,
        f"mínimo para pagar uma ação de R$ {premissas.custo_acao:.0f}: {premissas.precisao_minima:.0%}",
        figuras / "precisao_no_topo.png",
    )
    graficos.calibracao(faixas, figuras / "calibracao.png")
    graficos.importancia_das_variaveis(importancias, figuras / "importancia.png")

    principal = politica_capacidade[f"{FRACAO_SINALIZADA:.0%}"]
    print(f"\nROC-AUC no teste: {resultados['teste'][MISTURA]['roc_auc']:.3f}")
    print(
        f"Sinalizando {FRACAO_SINALIZADA:.0%} por semana: precisão {principal['precisao']:.1%} "
        f"(taxa de atraso {y_teste.mean():.1%}), recall {principal['recall']:.1%}"
    )
    return resultados


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--rapido", action="store_true", help="busca curta de hiperparâmetros")
    executar(rapido=parser.parse_args().rapido)
