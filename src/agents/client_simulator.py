"""
Módulo del simulador de cliente bancario basado en LLM.
"""

from typing import Any
from langchain_core.messages import HumanMessage, AIMessage
from src.data.models import ScenarioProfile, Turn, SpeakerRole
from src.agents.prompts import CLIENT_PROMPT_TEMPLATE


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
            "input": user_input
        })

        return parse_clean_llm_response(raw_response)