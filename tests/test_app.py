"""Teste de fumaça do app: abre a tela, troca de pedido e altera um campo."""

from __future__ import annotations

from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parents[1]
ARTEFATOS = [
    RAIZ / "models" / "modelo.joblib",
    RAIZ / "models" / "exemplos.parquet",
    RAIZ / "reports" / "resultados.json",
]

pytestmark = pytest.mark.skipif(
    not all(caminho.exists() for caminho in ARTEFATOS),
    reason="modelo ainda não treinado: rode `python -m atraso_entregas.treinar`",
)


def _metricas(app) -> dict[str, str]:
    return {m.label: m.value for m in app.metric}


def test_app_abre_preve_e_reage_a_alteracoes():
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(str(RAIZ / "app" / "app.py"), default_timeout=60).run()
    assert not app.exception
    inicial = _metricas(app)
    assert inicial["Resultado real"] in {"Atrasou", "Chegou no prazo"}
    assert inicial["Probabilidade de atraso"].endswith("%")

    # Prazo de 1 dia: o risco tem que subir, e o resultado real deixa de valer.
    prazo = next(campo for campo in app.number_input if campo.label.startswith("Prazo prometido"))
    prazo.set_value(1.0).run()
    assert not app.exception
    alterado = _metricas(app)
    assert alterado["Resultado real"] == "—"

    def como_numero(texto: str) -> float:
        return float(texto.rstrip("%").replace(",", "."))

    assert como_numero(alterado["Probabilidade de atraso"]) > como_numero(
        inicial["Probabilidade de atraso"]
    )

    # Trocar de pedido recarrega os campos com os dados do novo exemplo.
    app.selectbox[0].set_value(5).run()
    assert not app.exception
    assert _metricas(app)["Resultado real"] in {"Atrasou", "Chegou no prazo"}
