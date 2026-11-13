"""Separação temporal dos dados: treino, validação e teste (out-of-time)."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from atraso_entregas import config


@dataclass(frozen=True)
class Periodos:
    treino: pd.DataFrame
    validacao: pd.DataFrame
    teste: pd.DataFrame
    n_imaturos: int  # pedidos descartados por não terem resultado conhecido no corte

    @property
    def desenvolvimento(self) -> pd.DataFrame:
        """Tudo o que estaria disponível para treinar na data de corte."""
        return pd.concat([self.treino, self.validacao], ignore_index=True)


def separar_periodos(
    tabela: pd.DataFrame,
    inicio_treino: str = config.INICIO_TREINO,
    fim_treino: str = config.FIM_TREINO,
    data_de_corte: str = config.DATA_DE_CORTE,
) -> Periodos:
    """Simula o treino feito em `data_de_corte` e o uso do modelo dali em diante.

    - treino:    compras em [inicio_treino, fim_treino)
    - validação: compras em [fim_treino, data_de_corte)
    - teste:     compras a partir de `data_de_corte`

    Em `data_de_corte` só dá para saber se um pedido atrasou quando o prazo
    prometido já venceu. Pedidos comprados antes do corte com prazo ainda em
    aberto ("imaturos") são descartados do treino e da validação. O filtro usa o
    prazo prometido, e não a data da entrega, para não selecionar pelo resultado:
    manter só os já entregues deixaria os pedidos recentes artificialmente pontuais.
    """
    inicio, fim, corte = (pd.Timestamp(d) for d in (inicio_treino, fim_treino, data_de_corte))
    if not inicio < fim < corte:
        raise ValueError("As datas precisam respeitar inicio_treino < fim_treino < data_de_corte.")

    compra = tabela["data_compra"]
    maduro = tabela["data_prometida"] < corte

    treino = tabela[(compra >= inicio) & (compra < fim) & maduro]
    validacao = tabela[(compra >= fim) & (compra < corte) & maduro]
    teste = tabela[compra >= corte]
    n_imaturos = int(((compra >= inicio) & (compra < corte) & ~maduro).sum())

    return Periodos(
        treino=treino.reset_index(drop=True),
        validacao=validacao.reset_index(drop=True),
        teste=teste.reset_index(drop=True),
        n_imaturos=n_imaturos,
    )
