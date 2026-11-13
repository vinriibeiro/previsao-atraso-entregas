"""Gráficos do relatório. Um estilo único para todas as figuras."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FuncFormatter, PercentFormatter

AZUL, LARANJA, VERDE_AGUA = "#2a78d6", "#eb6834", "#1baf7a"
CINZA_MARCA = "#c3c2b7"
SUPERFICIE = "#fcfcfb"
TINTA, TINTA_2, TINTA_SUAVE = "#0b0b0b", "#52514e", "#898781"
GRADE, EIXO = "#e1e0d9", "#c3c2b7"

ESTILO = {
    "figure.facecolor": SUPERFICIE,
    "axes.facecolor": SUPERFICIE,
    "savefig.facecolor": SUPERFICIE,
    "font.family": ["Segoe UI", "DejaVu Sans"],
    "font.size": 10,
    "text.color": TINTA,
    "axes.labelcolor": TINTA_2,
    "axes.edgecolor": EIXO,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "axes.titlepad": 14,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": GRADE,
    "grid.linewidth": 0.8,
    "xtick.color": TINTA_SUAVE,
    "ytick.color": TINTA_SUAVE,
    "xtick.labelcolor": TINTA_2,
    "ytick.labelcolor": TINTA_2,
    "lines.linewidth": 2,
    "legend.frameon": False,
}

_PCT = PercentFormatter(1.0, decimals=0)


def _virgula(valor: float, casas: int) -> str:
    """Número com vírgula decimal, como se escreve em português."""
    return f"{valor:.{casas}f}".replace(".", ",")


def _salvar(fig, caminho: Path) -> None:
    caminho.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(caminho, dpi=160, bbox_inches="tight")
    plt.close(fig)


def atraso_por_mes(
    tabela: pd.DataFrame, fim_treino: str, data_de_corte: str, caminho: Path
) -> None:
    """Taxa de atraso por mês da compra, com os três períodos marcados."""
    mensal = tabela.set_index("data_compra")["atrasou"].resample("MS").agg(["mean", "size"])
    mensal = mensal[mensal["size"] >= 500]
    # Cada ponto fica no meio do seu mês, para cair do lado certo das divisórias.
    inicio_do_mes = mensal.index
    mensal.index = mensal.index + pd.Timedelta(days=14)

    with plt.rc_context(ESTILO):
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(mensal.index, mensal["mean"], color=AZUL, marker="o", markersize=5)
        ax.set_ylim(0, mensal["mean"].max() * 1.25)
        ax.yaxis.set_major_formatter(_PCT)
        ax.grid(axis="x", visible=False)
        topo = ax.get_ylim()[1]
        limites = [
            inicio_do_mes.min(),
            pd.Timestamp(fim_treino),
            pd.Timestamp(data_de_corte),
            inicio_do_mes.max() + pd.DateOffset(months=1),
        ]
        for inicio, fim, nome in zip(limites[:-1], limites[1:], ["Treino", "Validação", "Teste"]):
            ax.text(
                inicio + (fim - inicio) / 2, topo * 0.95, nome, ha="center", va="top", color=TINTA_2
            )
        for limite in limites[1:3]:
            ax.axvline(limite, color=EIXO, linewidth=1)
        ax.set_xlim(limites[0], limites[-1])
        pico = mensal["mean"].idxmax()
        ax.annotate(
            f"{mensal.loc[pico, 'mean']:.0%}",
            (pico, mensal.loc[pico, "mean"]),
            textcoords="offset points",
            xytext=(-8, 4),
            ha="right",
            color=TINTA,
        )
        ax.set_title("A taxa de atraso muda muito de um mês para o outro")
        ax.set_ylabel("Pedidos entregues com atraso")
        ax.set_xlabel("Mês da compra")
        _salvar(fig, caminho)


def backtest(resultado: pd.DataFrame, cores: dict[str, str], caminho: Path) -> None:
    """ROC-AUC de cada candidato em cada data de origem do backtest."""
    with plt.rc_context(ESTILO):
        fig, ax = plt.subplots(figsize=(9, 4.4))
        for nome, cor in cores.items():
            serie = resultado[resultado["candidato"] == nome]
            ax.plot(
                serie["origem"], serie["roc_auc"], color=cor, marker="o", markersize=5, label=nome
            )
        ax.axhline(0.5, color=TINTA_2, linewidth=1)
        ax.text(0, 0.5, "sem informação (0,50)", va="bottom", color=TINTA_2)
        ax.set_ylim(0.48, max(0.75, resultado["roc_auc"].max() + 0.02))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _pos: _virgula(v, 2)))
        ax.grid(axis="x", visible=False)
        ax.set_title("Backtest: desempenho nos 3 meses seguintes a cada data de treino")
        ax.set_xlabel("Data de origem (o modelo só conhece o que veio antes)")
        ax.set_ylabel("ROC-AUC")
        ax.legend(loc="upper left", bbox_to_anchor=(0, -0.2), ncols=1)
        _salvar(fig, caminho)


def comparacao_de_modelos(roc_auc: dict[str, float], destaque: str, caminho: Path) -> None:
    """ROC-AUC no teste de cada candidato, contra 0,50 (modelo sem informação)."""
    nomes = list(roc_auc)[::-1]
    valores = [roc_auc[n] for n in nomes]

    with plt.rc_context(ESTILO):
        fig, ax = plt.subplots(figsize=(8, 0.62 * len(nomes) + 1.6))
        cores = [AZUL if n == destaque else CINZA_MARCA for n in nomes]
        ax.axvline(0.5, color=TINTA_2, linewidth=1, zorder=1)
        ax.text(0.5, len(nomes) - 0.42, " sem informação: 0,50", color=TINTA_2, va="bottom")
        barras = ax.barh(nomes, valores, color=cores, height=0.55, zorder=2)
        ax.bar_label(
            barras,
            labels=[_virgula(v, 3) for v in valores],
            padding=5,
            color=TINTA,
            bbox={"facecolor": SUPERFICIE, "edgecolor": "none", "pad": 1.5},
        )
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _pos: _virgula(v, 1)))
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.6, len(nomes) - 0.1)
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", length=0)
        ax.set_title("Qualidade do ranqueamento no teste (ROC-AUC)")
        ax.set_xlabel("ROC-AUC (quanto maior, melhor)")
        _salvar(fig, caminho)


def calibracao(faixas: pd.DataFrame, caminho: Path) -> None:
    """Probabilidade prevista x frequência observada por décimo de risco."""
    with plt.rc_context(ESTILO):
        fig, ax = plt.subplots(figsize=(5.4, 5))
        limite = max(faixas["previsto"].max(), faixas["observado"].max()) * 1.1
        ax.plot([0, limite], [0, limite], color=EIXO, linewidth=1)
        ax.text(
            limite * 0.98,
            limite * 0.93,
            "calibração perfeita",
            ha="right",
            va="top",
            color=TINTA_2,
            rotation=45,
            rotation_mode="anchor",
        )
        ax.plot(faixas["previsto"], faixas["observado"], color=AZUL, marker="o", markersize=6)
        ax.set_xlim(0, limite)
        ax.set_ylim(0, limite)
        ax.xaxis.set_major_formatter(_PCT)
        ax.yaxis.set_major_formatter(_PCT)
        ax.set_title("Calibração no teste (décimos de risco)")
        ax.set_xlabel("Probabilidade média prevista")
        ax.set_ylabel("Atraso observado")
        _salvar(fig, caminho)


def precisao_no_topo(
    curvas: dict[str, pd.DataFrame], precisao_minima: float, rotulo_minima: str, caminho: Path
) -> None:
    """Precisão ao sinalizar os k% de maior risco, contra a precisão que paga a ação."""
    with plt.rc_context(ESTILO):
        fig, ax = plt.subplots(figsize=(8, 4.4))
        for (nome, curva), cor in zip(curvas.items(), [AZUL, LARANJA]):
            ax.plot(curva["fracao"], curva["precisao"], color=cor, label=nome)
        ax.axhline(precisao_minima, color=TINTA_2, linewidth=1)
        ax.text(0.006, precisao_minima, rotulo_minima, ha="left", va="top", color=TINTA_2)
        ax.set_xlim(0, 0.5)
        ax.set_ylim(0, None)
        ax.xaxis.set_major_formatter(_PCT)
        ax.yaxis.set_major_formatter(_PCT)
        ax.set_title("Quantos dos pedidos sinalizados realmente atrasam")
        ax.set_xlabel("Fração dos pedidos sinalizada (os de maior risco primeiro)")
        ax.set_ylabel("Precisão")
        ax.legend(loc="upper right")
        _salvar(fig, caminho)


def importancia_das_variaveis(importancias: pd.Series, caminho: Path, n: int = 12) -> None:
    """Queda na ROC-AUC do teste ao embaralhar cada variável."""
    topo = importancias.sort_values(ascending=False).head(n)[::-1]

    with plt.rc_context(ESTILO):
        fig, ax = plt.subplots(figsize=(8, 0.42 * len(topo) + 1.4))
        ax.barh(topo.index, topo.to_numpy(), color=AZUL, height=0.6)
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", length=0)
        ax.set_title("O que mais pesa na previsão")
        ax.set_xlabel("Queda na ROC-AUC do teste ao embaralhar a variável")
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _pos: _virgula(v, 2)))
        _salvar(fig, caminho)
