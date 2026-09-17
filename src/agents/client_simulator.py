"""
Módulo del simulador de cliente bancario basado en LLM.
"""

import re
from typing import Any
from langchain_core.messages import HumanMessage, AIMessage
from src.data.models import ScenarioProfile, Turn, SpeakerRole
from src.agents.prompts import CLIENT_PROMPT_TEMPLATE


# Palabras funcionales muy frecuentes en cada idioma, usadas para detectar
# de forma barata (sin dependencias externas ni llamadas a API) en qué
# idioma escribe el agente en cada turno, y así poder pedirle al LLM que
# responda en el mismo idioma de forma dinámica (ver prompts.py).
_SPANISH_MARKERS = {
    "el", "la", "los", "las", "de", "que", "es", "por", "para", "usted",
    "gracias", "hola", "buenos", "días", "tardes", "necesito", "puedo",
    "cómo", "qué", "cuenta", "tarjeta", "quiero", "está", "número",
}
_ENGLISH_MARKERS = {
    "the", "is", "you", "to", "and", "please", "thanks", "thank", "hello",
    "hi", "need", "can", "account", "card", "want", "how", "what", "number",
}


def detect_language(text: str) -> str:
    """Heurística ligera para distinguir español/inglés a partir de
    palabras funcionales comunes. Por defecto asume inglés (idioma
    original del corpus) si no hay señal clara."""
    words = set(re.findall(r"[a-záéíóúñ]+", text.lower()))
    es_score = len(words & _SPANISH_MARKERS)
    en_score = len(words & _ENGLISH_MARKERS)
    return "español" if es_score > en_score else "inglés"


def parse_clean_llm_response(response: Any) -> str:
    """Extrae texto plano de las respuestas del LLM descartando metadatos o firmas."""
    if isinstance(response, str):
        return response.strip()

    content = getattr(response, "content", response)

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        extracted_texts = []
        for block in content:
            if isinstance(block, dict) and "text" in block:
                extracted_texts.append(block["text"])
            elif isinstance(block, str):
                extracted_texts.append(block)
            elif hasattr(block, "text"):
                extracted_texts.append(getattr(block, "text"))
        if extracted_texts:
            return " ".join(extracted_texts).strip()

    return str(content).strip()


class ClientSimulator:
    """Simulador conversacional que asume el rol de cliente parametrizado."""

    def __init__(self, llm: Any, scenario: ScenarioProfile):
        self.llm = llm
        self.scenario = scenario
        self.prompt = CLIENT_PROMPT_TEMPLATE

    def _format_history_for_langchain(self, turns: list[Turn]):
        """Convierte los turnos de la sesión en objetos de mensaje de LangChain."""
        messages = []
        for turn in turns:
            if turn.speaker == SpeakerRole.CLIENT:
                messages.append(AIMessage(content=turn.text))
            elif turn.speaker == SpeakerRole.AGENT:
                messages.append(HumanMessage(content=turn.text))
        return messages

    def generate_response(self, turns: list[Turn], user_input: str) -> str:
        """Genera el siguiente turno del cliente inyectando la plantilla y el historial."""
        history_messages = self._format_history_for_langchain(turns)

        chain = self.prompt | self.llm

        raw_response = chain.invoke({
            "client_name": self.scenario.client_name,
            "account_number": self.scenario.account_number,
            "id_number": self.scenario.id_number,
            "initial_emotion": self.scenario.initial_emotion,
            "category": self.scenario.category,
            "issue_description": self.scenario.issue_description,
            "ground_truth_transcript": self.scenario.ground_truth_transcript[:500],
            "history": history_messages,
            "input": user_input,
            "detected_language": detect_language(user_input),
        })

        return parse_clean_llm_response(raw_response)