"""
Módulo de modelos de datos (Pydantic y Enums) para el sistema de roleplay bancario.

"""
from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field


class SpeakerRole(str, Enum):
    AGENT = "agent"
    CLIENT = "client"


class Turn(BaseModel):
    speaker: SpeakerRole
    text: str
    date_time: Optional[str] = None


class ScenarioProfile(BaseModel):
    scenario_id: str
    category: str
    client_name: str = "Cliente"
    account_number: str = "N/A"
    id_number: str = "N/A"
    initial_emotion: str = "Neutral"
    issue_description: str = ""
    ground_truth_transcript: str = ""


class ConversationSession(BaseModel):
    session_id: str
    scenario: ScenarioProfile
    turns: List[Turn] = Field(default_factory=list)
    is_finished: bool = False

    def add_turn(self, speaker: SpeakerRole, text: str) -> None:
        self.turns.append(Turn(speaker=speaker, text=text))

    def get_formatted_transcript(self) -> str:
        lines = []
        for t in self.turns:
            role = "AGENTE" if t.speaker == SpeakerRole.AGENT else f"CLIENTE ({self.scenario.client_name})"
            lines.append(f"{role}: {t.text}")
        return "\n".join(lines)


class SOPEvaluationResult(BaseModel):
    overall_score: int = Field(ge=1, le=10, description="Puntuación global de 1 a 10")
    greeting_check: bool
    identity_verified: bool
    resolution_quality: str
    digital_channel_guidance: bool
    closing_check: bool
    strengths: List[str]
    improvement_areas: List[str]
    sop_adherence_status: str
    detailed_feedback: str