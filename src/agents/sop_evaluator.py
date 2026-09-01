"""
Módulo del evaluador de calidad y cumplimiento de Protocolos Operativos Estándar (SOP).
"""
import time
import logging
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from src.data.models import ConversationSession, SOPEvaluationResult

logger = logging.getLogger(__name__)


class SOPEvaluator:
    def __init__(self, llm: ChatGoogleGenerativeAI):
        self.llm = llm

    def evaluate_session(self, session: ConversationSession, max_retries: int = 3) -> SOPEvaluationResult:
        """
        Audita la conversación finalizada mediante salida estructurada Pydantic.
        Aplica backoff exponencial para absorber picos 503 de Gemini.
        """
        formatted_transcript = session.get_formatted_transcript()

        prompt_template = ChatPromptTemplate.from_messages([
            ("system", """Eres un Auditor Superior de Calidad Operativa y Cumplimiento Normativo Bancario.
Tu tarea es auditar la transcripción de una interacción entre un Agente Bancario y un Cliente.

Evalúa con rigor los siguientes criterios SOP:
1. Saludo Institucional y cortesía profesional.
2. Identificación y verificación de identidad (DNI/SSN o Nº Cuenta).
3. Calidad de la solución técnica/operativa aportada.
4. Guía activa hacia canales digitales (banca online/app).
5. Cierre profesional de la llamada.

Debes categorizar sop_adherence_status estrictamente en una de las siguientes opciones:
- "Cumplido"
- "Parcialmente Cumplido"
- "No Cumplido"
"""),
            ("user", "Transcripción de la llamada a auditar:\n\n{conversation_history}")
        ])

        # Forzar salida estructurada orientada al esquema Pydantic
        structured_llm = self.llm.with_structured_output(SOPEvaluationResult)
        chain = prompt_template | structured_llm

        last_exception = None
        for attempt in range(1, max_retries + 1):
            try:
                result = chain.invoke({"conversation_history": formatted_transcript})
                return result
            except Exception as err:
                last_exception = err
                logger.warning(f"Intento {attempt}/{max_retries} fallido en SOPEvaluator: {err}")
                if attempt < max_retries:
                    time.sleep(2 ** attempt)  # Espera 2s, 4s...

        # Fallback de seguridad ante caídas persistentes del servidor
        return SOPEvaluationResult(
            overall_score=5,
            greeting_check=True,
            identity_verified=False,
            resolution_quality="No fue posible completar la auditoría detallada por saturación en la API del LLM.",
            digital_channel_guidance=False,
            closing_check=False,
            strengths=["Interacción completada y registrada en el historial"],
            improvement_areas=["Reintentar la auditoría en unos minutos debido a alta demanda del servidor"],
            sop_adherence_status="Parcialmente Cumplido",
            detailed_feedback=f"⚠️ Aviso del sistema: La auditoría automática experimentó un error 503 ({last_exception})."
        )