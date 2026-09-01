"""
Pruebas unitarias para el módulo de evaluación de protocolos SOP.
"""

import os
import pytest
from dotenv import load_dotenv

from src.data.loader import BankingDataLoader
from src.evaluation.models import SOPChecklist, SOPReport
from src.evaluation.evaluator import SOPEvaluator

load_dotenv()


def test_sop_report_model_instantiation():
    """Valida la creación y tipos del modelo Pydantic SOPReport."""
    checklist = SOPChecklist(
        greeting_and_tone=True,
        identity_verified=True,
        issue_handled=True,
        digital_derivation=False,
        farewell_professional=True
    )
    report = SOPReport(
        score=85.0,
        checklist=checklist,
        strengths=["Buena verificación de identidad", "Tono amable"],
        areas_for_improvement=["No ofreció el uso de la app móvil"],
        qualitative_feedback="El agente desempeñó una llamada correcta cumpliendo la seguridad básica."
    )

    assert report.score == 85.0
    assert report.checklist.identity_verified is True
    assert len(report.strengths) == 2


@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="Requiere OPENAI_API_KEY en el archivo .env"
)
def test_evaluation_integration_with_csv():
    """Prueba de integración leyendo una sesión real del CSV y evaluándola con el LLM."""
    loader = BankingDataLoader("data/banking_corpus.csv")
    sessions = loader.load_sessions()
    first_session_id = next(iter(sessions))
    session = sessions[first_session_id]

    evaluator = SOPEvaluator(model_name="gpt-4o-mini")
    report = evaluator.evaluate_session(session)

    assert isinstance(report, SOPReport)
    assert 0.0 <= report.score <= 100.0
    assert isinstance(report.checklist.identity_verified, bool)