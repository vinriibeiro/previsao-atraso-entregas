"""Métricas de qualidade do modelo e tradução do resultado em dinheiro."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from atraso_entregas.config import PREMISSAS, PremissasDeCusto

CORTES = np.round(np.arange(0.01, 0.951, 0.005), 3)


def captura_no_topo(y, prob, fracao: float = 0.10) -> float:
    """Fração de todos os atrasos que está nos `fracao` pedidos de maior risco."""
    y, prob = np.asarray(y), np.asarray(prob)
    n_topo = max(1, int(round(len(y) * fracao)))
    topo = np.argsort(-prob, kind="stable")[:n_topo]
    return float(y[topo].sum() / max(y.sum(), 1))


def metricas(y, prob) -> dict[str, float]:
    """Métricas que não dependem de ponto de corte."""
    y, prob = np.asarray(y), np.asarray(prob)
    return {
        "n": int(len(y)),
        "taxa_atraso": float(y.mean()),
        "pr_auc": float(average_precision_score(y, prob)),
        "roc_auc": float(roc_auc_score(y, prob)),
        "brier": float(brier_score_loss(y, prob)),
        "captura_top10": captura_no_topo(y, prob, 0.10),
    }


def resultado_da_decisao(
    y, sinalizado, premissas: PremissasDeCusto = PREMISSAS
) -> dict[str, float]:
    """Custo de agir preventivamente sobre os pedidos sinalizados.

    - atraso não sinalizado: paga o custo cheio do atraso;
    - pedido sinalizado: paga a ação, e a ação evita parte dos atrasos;
    - comparação: não fazer nada (todo atraso custa o valor cheio).
    """
    y = np.asarray(y).astype(bool)
    sinalizado = np.asarray(sinalizado).astype(bool)

    acertos = int((y & sinalizado).sum())
    alarmes_falsos = int((~y & sinalizado).sum())
    perdidos = int((y & ~sinalizado).sum())
    n_sinalizados = acertos + alarmes_falsos

    custo = (
        perdidos * premissas.custo_atraso
        + n_sinalizados * premissas.custo_acao
        + acertos * (1 - premissas.eficacia_acao) * premissas.custo_atraso
    )
    custo_sem_acao = float(y.sum() * premissas.custo_atraso)
    economia = custo_sem_acao - custo

    return {
        "n": int(len(y)),
        "sinalizados": n_sinalizados,
        "pct_sinalizados": n_sinalizados / len(y),
        "atrasos_capturados": acertos,
        "alarmes_falsos": alarmes_falsos,
        "atrasos_perdidos": perdidos,
        "precisao": acertos / n_sinalizados if n_sinalizados else float("nan"),
        "recall": acertos / int(y.sum()) if y.any() else float("nan"),
        "custo": float(custo),
        "custo_sem_acao": custo_sem_acao,
        "economia": float(economia),
        "economia_pct": float(economia / custo_sem_acao) if custo_sem_acao else float("nan"),
    }


def curva_de_custo(y, prob, premissas: PremissasDeCusto = PREMISSAS, cortes=CORTES) -> pd.DataFrame:
    """Resultado da decisão para cada ponto de corte."""
    prob = np.asarray(prob)
    linhas = [{"corte": float(c), **resultado_da_decisao(y, prob >= c, premissas)} for c in cortes]
    return pd.DataFrame(linhas)


def escolher_corte(y, prob, premissas: PremissasDeCusto = PREMISSAS) -> float:
    """Ponto de corte de maior economia. Deve ser escolhido na validação, nunca no teste."""
    curva = curva_de_custo(y, prob, premissas)
    return float(curva.loc[curva["economia"].idxmax(), "corte"])


def topo_por_semana(datas, prob, fracao: float = 0.10) -> np.ndarray:
    """Sinaliza os `fracao` pedidos de maior risco dentro de cada semana.

    Política de capacidade fixa ("a equipe trata 10% dos pedidos"). Depende só
    da ordem das notas dentro da semana, e não do valor da probabilidade: por
    isso continua válida quando a taxa de atraso muda de patamar.
    """
    df = pd.DataFrame(
        {"semana": pd.Series(np.asarray(datas)).dt.to_period("W"), "prob": np.asarray(prob)}
    )
    posicao = df.groupby("semana")["prob"].rank(method="first", ascending=False)
    tamanho = df.groupby("semana")["prob"].transform("size")
    return (posicao <= np.ceil(tamanho * fracao)).to_numpy()


def custo_maximo_da_acao(precisao: float, premissas: PremissasDeCusto = PREMISSAS) -> float:
    """Quanto a ação preventiva pode custar, por pedido, para ainda valer a pena.

    Cada pedido sinalizado evita, em média, precisão x eficácia x custo do atraso.
    """
    return precisao * premissas.eficacia_acao * premissas.custo_atraso


def calibracao_por_faixa(y, prob, n_faixas: int = 10) -> pd.DataFrame:
    """Probabilidade prevista x frequência observada, em faixas de mesmo tamanho."""
    df = pd.DataFrame({"y": np.asarray(y), "prob": np.asarray(prob)})
    df["faixa"] = pd.qcut(df["prob"].rank(method="first"), n_faixas, labels=False)
    return (
        df.groupby("faixa")
        .agg(n=("y", "size"), previsto=("prob", "mean"), observado=("y", "mean"))
        .reset_index()
    )


def por_grupo(df: pd.DataFrame, grupo, y: str, prob: str, sinalizado: str) -> pd.DataFrame:
    """Métricas por recorte (mês, região...), para ver onde o modelo vai melhor e pior."""
    linhas = []
    for nome, parte in df.groupby(grupo):
        linha = {"grupo": str(nome), "n": len(parte), "taxa_atraso": float(parte[y].mean())}
        if parte[y].nunique() == 2:
            linha["roc_auc"] = float(roc_auc_score(parte[y], parte[prob]))
            linha["lift"] = float(average_precision_score(parte[y], parte[prob]) / parte[y].mean())
        decisao = resultado_da_decisao(parte[y], parte[sinalizado])
        linha |= {k: decisao[k] for k in ("pct_sinalizados", "precisao", "recall")}
        linhas.append(linha)
    return pd.DataFrame(linhas)


def precisao_no_topo(y, prob, fracoes=None) -> pd.DataFrame:
    """Precisão e recall ao sinalizar os k% de maior risco, para vários k."""
    y, prob = np.asarray(y), np.asarray(prob)
    fracoes = np.arange(0.01, 0.501, 0.01) if fracoes is None else fracoes
    ordenado = y[np.argsort(-prob, kind="stable")]
    acumulado = np.cumsum(ordenado)
    linhas = []
    for fracao in fracoes:
        k = max(1, int(round(len(y) * fracao)))
        linhas.append(
            {
                "fracao": float(fracao),
                "precisao": acumulado[k - 1] / k,
                "recall": acumulado[k - 1] / y.sum(),
            }
        )
    return pd.DataFrame(linhas)
