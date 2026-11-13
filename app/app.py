"""App de demonstração: risco de atraso de um pedido.

Uso:
    streamlit run app/app.py

O app não depende do pacote `atraso_entregas`: lê apenas os arquivos gerados
pelo treino (`models/` e `reports/`), para poder ser publicado sozinho.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

RAIZ = Path(__file__).resolve().parents[1]
ALVO = "atrasou"

# Variáveis agrupadas por assunto; o rótulo do grupo é o que aparece na
# explicação da previsão. As variáveis derivadas (folgas de prazo, mesma UF,
# frete sobre valor) não entram aqui: são recalculadas por `completar_derivadas`.
GRUPOS = {
    "Prazo prometido": ["prazo_prometido_dias"],
    "Distância e estados": ["distancia_km", "uf_cliente", "uf_vendedor"],
    "Histórico do destino": ["destino_dias_entrega", "destino_taxa_atraso"],
    "Histórico da rota": ["rota_dias_entrega", "rota_taxa_atraso"],
    "Histórico do vendedor": [
        "vendedor_taxa_atraso",
        "vendedor_dias_postagem",
        "vendedor_pedidos_30d",
    ],
    "Produto e valores": [
        "categoria_principal",
        "valor_produtos",
        "valor_frete",
        "peso_total_g",
        "volume_total_cm3",
        "n_itens",
        "n_vendedores",
    ],
    "Pagamento e horário": ["tipo_pagamento", "parcelas", "dia_semana", "hora_compra"],
}

# Campos editáveis na tela.
# variável -> (rótulo, mínimo, máximo, passo, fator de exibição)
NUMERICOS = {
    "prazo_prometido_dias": ("Prazo prometido (dias)", 1.0, 160.0, 1.0, 1),
    "distancia_km": ("Distância vendedor → cliente (km)", 0.0, 4000.0, 50.0, 1),
    "destino_dias_entrega": ("Entrega típica na região do cliente (dias)", 0.0, 120.0, 1.0, 1),
    "destino_taxa_atraso": ("Atrasos na região do cliente (%)", 0.0, 100.0, 1.0, 100),
    "vendedor_taxa_atraso": ("Atrasos do vendedor (%)", 0.0, 100.0, 1.0, 100),
    "vendedor_dias_postagem": ("Vendedor leva para postar (dias)", 0.0, 60.0, 0.5, 1),
    "valor_produtos": ("Valor dos produtos (R$)", 0.0, 15000.0, 10.0, 1),
    "valor_frete": ("Frete (R$)", 0.0, 2000.0, 5.0, 1),
}
CATEGORICOS = {
    "uf_vendedor": "Estado do vendedor",
    "uf_cliente": "Estado do cliente",
    "categoria_principal": "Categoria",
    "tipo_pagamento": "Forma de pagamento",
}
# Como os campos se distribuem nas três colunas da tela.
COLUNAS = [
    ("Entrega", ["prazo_prometido_dias", "distancia_km", "uf_vendedor", "uf_cliente"]),
    (
        "Histórico (até a data da compra)",
        [
            "destino_dias_entrega",
            "destino_taxa_atraso",
            "vendedor_taxa_atraso",
            "vendedor_dias_postagem",
        ],
    ),
    (
        "Produto e pagamento",
        ["categoria_principal", "valor_produtos", "valor_frete", "tipo_pagamento"],
    ),
]

st.set_page_config(page_title="Risco de atraso na entrega", page_icon="📦", layout="wide")


@st.cache_resource
def carregar_modelo() -> dict:
    return joblib.load(RAIZ / "models" / "modelo.joblib")


@st.cache_data
def carregar_exemplos() -> pd.DataFrame:
    return pd.read_parquet(RAIZ / "models" / "exemplos.parquet")


@st.cache_data
def carregar_resultados() -> dict:
    return json.loads((RAIZ / "reports" / "resultados.json").read_text(encoding="utf-8"))


def completar_derivadas(pedido: dict) -> dict:
    """Recalcula as variáveis que dependem de outras, depois de uma edição na tela."""
    pedido = dict(pedido)
    pedido["mesma_uf"] = int(pedido["uf_cliente"] == pedido["uf_vendedor"])
    pedido["frete_sobre_valor"] = (
        pedido["valor_frete"] / pedido["valor_produtos"] if pedido["valor_produtos"] else np.nan
    )
    pedido["folga_prazo_rota_dias"] = pedido["prazo_prometido_dias"] - pedido["rota_dias_entrega"]
    pedido["folga_prazo_destino_dias"] = (
        pedido["prazo_prometido_dias"] - pedido["destino_dias_entrega"]
    )
    return pedido


def prever(modelo, variaveis: list[str], pedidos: list[dict]) -> np.ndarray:
    return modelo.predict_proba(pd.DataFrame(pedidos)[variaveis])[:, 1]


def explicar(modelo, variaveis: list[str], pedido: dict, tipico: dict) -> pd.DataFrame:
    """Quanto o risco muda quando um grupo de variáveis é trocado pelo de um pedido típico.

    É uma explicação por substituição: simples de entender, mas não soma
    exatamente a diferença total, porque os grupos interagem entre si.
    """
    cenarios = [
        completar_derivadas({**pedido, **{v: tipico[v] for v in grupo}})
        for grupo in GRUPOS.values()
    ]
    risco = prever(modelo, variaveis, [pedido])[0]
    sem_o_grupo = prever(modelo, variaveis, cenarios)
    return (
        pd.DataFrame({"Fator": list(GRUPOS), "efeito": (risco - sem_o_grupo) * 100})
        .sort_values("efeito", key=np.abs, ascending=False)
        .reset_index(drop=True)
    )


artefato = carregar_modelo()
exemplos = carregar_exemplos()
resultados = carregar_resultados()
modelo, variaveis, tipico = artefato["modelo"], artefato["variaveis"], artefato["pedido_tipico"]
notas_de_referencia = np.asarray(artefato["notas_de_referencia"])
codificador = modelo.named_estimators_["boosting"].named_steps["preparo"].named_transformers_["cat"]
opcoes = dict(zip(codificador.feature_names_in_, (list(c) for c in codificador.categories_)))

st.title("Risco de atraso na entrega")
st.caption(
    "Modelo treinado com pedidos reais da Olist até maio de 2018. Os exemplos abaixo são pedidos de "
    "junho a agosto de 2018, que o modelo nunca viu."
)

aba_pedido, aba_modelo = st.tabs(["Avaliar um pedido", "Sobre o modelo"])

with aba_pedido:
    rotulos = {
        i: f"Pedido {i + 1} · {linha.data_compra:%d/%m/%Y} · {linha.uf_vendedor} → {linha.uf_cliente}"
        for i, linha in enumerate(exemplos.itertuples())
    }
    indice = st.selectbox("Pedido de exemplo", list(rotulos), format_func=rotulos.get)
    original = exemplos.iloc[indice]
    real = original[variaveis].to_dict()

    def inicial(variavel: str, fator: float = 1.0, casas: int = 1):
        """Valor que o campo mostra ao abrir: o do pedido real, ou vazio se não houver."""
        valor_real = real[variavel]
        return None if pd.isna(valor_real) else round(float(valor_real) * fator, casas)

    st.subheader("Dados do pedido")
    st.caption(
        "Altere os campos para simular outro cenário. Campo vazio significa que não há histórico."
    )
    alteracoes = {}
    for coluna, (titulo, campos) in zip(st.columns(3, gap="large"), COLUNAS):
        with coluna:
            st.markdown(f"**{titulo}**")
            for variavel in campos:
                # A chave inclui o pedido: trocar de exemplo recria os campos.
                chave = f"pedido{indice}_{variavel}"
                if variavel in CATEGORICOS:
                    escolha = st.selectbox(
                        CATEGORICOS[variavel],
                        opcoes[variavel],
                        opcoes[variavel].index(real[variavel]),
                        key=chave,
                    )
                    if escolha != real[variavel]:
                        alteracoes[variavel] = escolha
                else:
                    rotulo, minimo, maximo, passo, fator = NUMERICOS[variavel]
                    valor_inicial = inicial(variavel, fator)
                    digitado = st.number_input(
                        rotulo, minimo, maximo, valor_inicial, passo, format="%.1f", key=chave
                    )
                    if digitado != valor_inicial:
                        alteracoes[variavel] = np.nan if digitado is None else digitado / fator

    # Só os campos alterados substituem os dados reais: sem alteração, a
    # previsão é exatamente a do pedido original, sem efeito de arredondamento.
    alterado = bool(alteracoes)
    pedido = completar_derivadas({**real, **alteracoes}) if alterado else real
    risco = float(prever(modelo, variaveis, [pedido])[0])
    posicao = float(np.searchsorted(notas_de_referencia, risco) / (len(notas_de_referencia) - 1))
    entre_os_10 = risco >= notas_de_referencia[90]

    st.divider()
    r1, r2, r3 = st.columns(3, gap="large")
    r1.metric("Probabilidade de atraso", f"{risco:.1%}".replace(".", ","))
    r2.metric("Mais arriscado que", f"{min(posicao, 0.99):.0%} dos pedidos")
    if alterado:
        r3.metric(
            "Resultado real",
            "—",
            help="Você alterou os dados: o cenário deixou de ser o pedido real.",
        )
    else:
        r3.metric("Resultado real", "Atrasou" if original[ALVO] == 1 else "Chegou no prazo")

    if entre_os_10:
        st.warning(
            "**Sinalizar para ação preventiva.** Este pedido está entre os 10% de maior risco.",
            icon="⚠️",
        )
    else:
        st.info(
            "**Seguir o fluxo normal.** Este pedido está fora dos 10% de maior risco.", icon="✅"
        )

    st.subheader("O que pesa nesta previsão")
    st.caption(
        "Quanto o risco muda, em pontos percentuais, quando cada grupo de informações é trocado pelo de um "
        "pedido típico. Positivo aumenta o risco deste pedido; negativo reduz."
    )
    efeitos = explicar(modelo, variaveis, pedido, tipico)
    efeitos["Direção"] = np.where(
        efeitos["efeito"] > 0.05,
        "Aumenta o risco",
        np.where(efeitos["efeito"] < -0.05, "Reduz o risco", "Neutro"),
    )
    st.dataframe(
        efeitos,
        hide_index=True,
        column_config={
            "efeito": st.column_config.NumberColumn("Efeito no risco (p.p.)", format="%+.1f")
        },
    )

with aba_modelo:
    teste = resultados["teste"]["Mistura (modelo final)"]
    capacidade = resultados["politica_capacidade_teste"]["10%"]
    periodo = resultados["periodos"]["teste"]

    st.subheader("Desempenho em pedidos que o modelo nunca viu")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Pedidos no teste", f"{periodo['pedidos']:,}".replace(",", "."))
    m2.metric("Taxa de atraso", f"{periodo['taxa_atraso']:.1%}".replace(".", ","))
    m3.metric("ROC-AUC", f"{teste['roc_auc']:.3f}".replace(".", ","))
    m4.metric("Atrasos entre os 10% sinalizados", f"{capacidade['recall']:.0%}")
    st.markdown(
        f"""
Sinalizando os 10% de maior risco de cada semana, **{capacidade["precisao"]:.1%}** dos pedidos sinalizados
realmente atrasam, contra **{periodo["taxa_atraso"]:.1%}** na média. Com as premissas do projeto, a ação
preventiva só se paga se custar até **R$ {capacidade["custo_maximo_da_acao"]:.2f}** por pedido.

**Limites que você precisa conhecer**

- O modelo ordena os pedidos por risco melhor do que acerta o valor da probabilidade: a taxa de atraso
  muda muito de um mês para o outro, e a probabilidade acompanha o passado.
- Só enxerga o que existe no momento da compra. Greve, clima e falha da transportadora não estão nos dados.
- Foi treinado com dados de 2017 e 2018 de um único marketplace. Não vale para outra operação sem novo treino.
"""
    )
    for figura, legenda in [
        ("precisao_no_topo.png", "Precisão ao sinalizar os pedidos de maior risco"),
        ("importancia.png", "Variáveis que mais pesam no modelo"),
        ("backtest.png", "Estabilidade do modelo em diferentes datas de treino"),
    ]:
        caminho = RAIZ / "reports" / "figures" / figura
        if caminho.exists():
            st.image(str(caminho), caption=legenda)
