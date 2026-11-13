"""Caminhos, recortes de tempo e premissas de negócio do projeto."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
DIR_DADOS_BRUTOS = RAIZ / "data" / "raw"
DIR_MODELOS = RAIZ / "models"
DIR_RELATORIOS = RAIZ / "reports"
DIR_FIGURAS = DIR_RELATORIOS / "figures"

SEMENTE = 42

# ---- Recortes de tempo (pela data da compra) ----
# O primeiro mês de 2017 fica de fora do treino: serve só para "aquecer" as
# variáveis de histórico, que ainda não têm passado suficiente.
INICIO_TREINO = "2017-02-01"
FIM_TREINO = "2018-03-01"  # treino: [INICIO_TREINO, FIM_TREINO)
DATA_DE_CORTE = "2018-06-01"  # "hoje" simulado: validação antes, teste depois


@dataclass(frozen=True)
class PremissasDeCusto:
    """Premissas de negócio usadas para escolher o ponto de corte.

    São hipóteses declaradas, não números da Olist. A análise de
    sensibilidade no relatório mostra como o resultado muda com elas.
    """

    custo_atraso: float = 40.0  # R$ por pedido atrasado sem tratamento
    custo_acao: float = 4.0  # R$ por pedido sinalizado (ação preventiva)
    eficacia_acao: float = 0.5  # fração dos atrasos sinalizados que a ação evita

    @property
    def precisao_minima(self) -> float:
        """Precisão a partir da qual sinalizar um pedido passa a compensar."""
        return self.custo_acao / (self.eficacia_acao * self.custo_atraso)


PREMISSAS = PremissasDeCusto()
