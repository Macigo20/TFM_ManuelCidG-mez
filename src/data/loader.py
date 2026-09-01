"""

Módulo de ingesta y transformación de datos del corpus bancario.

"""
import csv
import json
import os
import re
from typing import Dict, List, Any, Optional
from src.data.models import ConversationSession, ScenarioProfile, SpeakerRole, Turn
 
 # Orden de evaluación: de más específico/crítico a más genérico. Una
# conversación de fraude que también menciona una tarjeta debe clasificarse
# como fraude, no como bloqueo de tarjeta, de ahí que fraude vaya primero.
# Orden de evaluación: de más específico/crítico a más genérico. Una
# conversación de fraude que también menciona una tarjeta debe clasificarse
# como fraude, no como bloqueo de tarjeta, de ahí que fraude vaya primero.
#
# "Bloqueo de Tarjetas" NO usa frases exactas (p. ej. "lost card") porque el
# corpus real casi nunca las dice tal cual: escribe "lost my debit card",
# "card... stolen", "cardit card expired", etc. Se usa en su lugar una regla
# de co-ocurrencia: la palabra "card" en cualquier forma + un verbo de acción
# relevante, en cualquier orden y con palabras de por medio.
CARD_ACTION_WORDS = [
    "lost", "stolen", "steal", "block", "blocked", "cancel", "cancelled",
    "canceled", "replace", "expired", "compromised", "deactivate", "freeze",
]
 
CATEGORY_KEYWORDS = [
    ("Ciberseguridad y Fraude", [
        "phishing", "fraud", "fraudulent", "suspicious", "scam", "spam",
        "unauthorized transaction", "unauthorised transaction", "hacked", "hack",
    ]),
    ("Gestión de Pagos y Facturas", [
        "automatic payment", "autopay", "auto-pay", "direct debit",
        "bill", "invoice", "payment plan", "set up payment", "electricity bill",
        "gas bill", "statement",
    ]),
]
DEFAULT_CATEGORY = "Atención General"
 
EMOTION_BY_CATEGORY = {
    "Ciberseguridad y Fraude": "confundida y recelosa",
    "Bloqueo de Tarjetas": "preocupada pero colaborativa",
    "Gestión de Pagos y Facturas": "colaborativa e informativa",
    "Atención General": "neutral e informativa",
}
 
# Frases habituales para introducir el nombre propio del cliente.
# Importante: solo la parte disparadora ("my name is", etc.) es
# case-insensitive vía (?i:...) — el grupo capturado del nombre exige
# mayúscula inicial explícita. Con re.IGNORECASE global, "I'm calling"
# capturaba "Calling" como si fuera un nombre propio.
NAME_PATTERNS = [
    r"(?i:my name is)\s+([A-Z][a-zA-Z]+)",
    r"(?i:this is)\s+([A-Z][a-zA-Z]+)\s+(?i:calling)",
    r"(?i:i'?m)\s+([A-Z][a-zA-Z]+)\b",
    r"(?i:soy)\s+([A-Z][a-zA-Z]+)",
]
 
# Fallback: cuando el propio cliente no dice su nombre, a veces el agente
# lo usa para dirigirse a él/ella justo después del primer turno del cliente
# (p. ej. "Sorry to hear that, Margery."). Se usa solo si NAME_PATTERNS
# no encontró nada en el texto del cliente.
NAME_ADDRESS_PATTERNS = [
    r"(?i:sorry to hear that|thank you|of course|i understand|no problem|thanks),?\s+([A-Z][a-zA-Z]+)[.,]",
]
_NAME_STOPWORDS = {"you", "sir", "maam", "there", "for"}
 
ACCOUNT_PATTERNS = [
    r"account number is[:\s]+(\d{6,12})",
    r"acct(?:\.|\s*number)?\s*(?:is|:)?\s*(\d{6,12})",
    # Fallback: cualquier número largo aislado. Se usa (?<!\d)...(?!\d) en vez
    # de \b porque el corpus tiene corrupciones de texto donde una letra queda
    # pegada al número sin espacio (p. ej. "it's L0123456789"), lo que rompe
    # el límite de palabra \b pero no afecta a esta comprobación.
    r"(?<!\d)(\d{6,12})(?!\d)",
]
 
# Frases que anticipan que el turno inmediatamente siguiente del cliente
# contendrá un identificador corto (los 4 últimos dígitos del SSN, DNI, etc.).
# Se usa junto con extract_id_number_from_turns, que mira turno a turno en
# vez de concatenar todo el texto: en el corpus real la pregunta y la
# respuesta van en turnos distintos ("...su social security number?" /
# "Okay, it's 1234."), así que un regex sobre el texto unido no lo detecta.
ID_TRIGGER_PATTERN = re.compile(
    r"(?i:social security number|last four digits|verify your identity|id number)"
)
SHORT_DIGIT_REPLY_PATTERN = re.compile(r"^\D*(\d{4})\D*$")
 
ID_PATTERNS = [
    r"social security number is[:\s]+(\d{3}-?\d{2}-?\d{4})",
    r"\b(\d{3}-\d{2}-\d{4})\b",
]
 
 
def classify_category(full_text: str) -> str:
    """Clasifica la conversación en una de las 4 categorías de negocio.
 
    Orden de prioridad: fraude > bloqueo de tarjeta (co-ocurrencia) >
    pagos/facturas (palabra clave) > atención general (por defecto)."""
    text_lower = full_text.lower()
 
    fraude_keywords = CATEGORY_KEYWORDS[0][1]
    if any(kw in text_lower for kw in fraude_keywords):
        return "Ciberseguridad y Fraude"
 
    if "card" in text_lower and any(action in text_lower for action in CARD_ACTION_WORDS):
        return "Bloqueo de Tarjetas"
 
    pagos_keywords = CATEGORY_KEYWORDS[1][1]
    if any(kw in text_lower for kw in pagos_keywords):
        return "Gestión de Pagos y Facturas"
 
    return DEFAULT_CATEGORY
 
 
def extract_client_name(full_client_text: str, full_text: str = "") -> str:
    for pattern in NAME_PATTERNS:
        match = re.search(pattern, full_client_text)
        if match:
            return match.group(1).capitalize()
    # El cliente no dijo su nombre: probamos si el agente lo usó al dirigirse a él.
    for pattern in NAME_ADDRESS_PATTERNS:
        match = re.search(pattern, full_text)
        if match:
            candidate = match.group(1)
            if candidate.lower() not in _NAME_STOPWORDS:
                return candidate.capitalize()
    return "Cliente"
 
 
def extract_account_number(full_text: str) -> str:
    for pattern in ACCOUNT_PATTERNS:
        match = re.search(pattern, full_text, re.IGNORECASE)
        if match:
            return match.group(1)
    return "N/A"
 
 
def extract_id_number_from_turns(turns: List[Turn], exclude: Optional[str] = None) -> str:
    """Busca el identificador corto (últimos 4 dígitos de SSN/DNI, etc.)
    mirando la secuencia de turnos: si un turno del agente menciona la
    verificación, se comprueba si el turno de cliente inmediatamente
    posterior es una respuesta corta compuesta solo por 4 dígitos."""
    for i, turn in enumerate(turns):
        if turn.speaker == SpeakerRole.AGENT and ID_TRIGGER_PATTERN.search(turn.text):
            for next_turn in turns[i + 1: i + 3]:
                if next_turn.speaker != SpeakerRole.CLIENT:
                    continue
                match = SHORT_DIGIT_REPLY_PATTERN.match(next_turn.text.strip())
                if match and match.group(1) != exclude:
                    return match.group(1)
                break
 
    # Fallback: patrones explícitos tipo SSN completo en el propio texto.
    full_text = " ".join(t.text for t in turns)
    for pattern in ID_PATTERNS:
        match = re.search(pattern, full_text, re.IGNORECASE)
        if match and match.group(1) != exclude:
            return match.group(1)
    return "N/A"
 
 
def infer_emotion(category: str) -> str:
    return EMOTION_BY_CATEGORY.get(category, "neutral")
 
 
def build_scenario_profile(conv_id: str, turns: List[Turn]) -> ScenarioProfile:
    """Construye el ScenarioProfile de una conversación a partir de sus turnos,
    aplicando las reglas de clasificación/extracción anteriores."""
    client_texts = [t.text for t in turns if t.speaker == SpeakerRole.CLIENT]
    full_client_text = " ".join(client_texts)
    full_text = " ".join(t.text for t in turns)
 
    category = classify_category(full_text)
    account_num = extract_account_number(full_text)
    id_num = extract_id_number_from_turns(turns, exclude=account_num)
    client_name = extract_client_name(full_client_text, full_text)
    emotion = infer_emotion(category)
 
    issue_description = client_texts[0] if client_texts else "Consulta bancaria"
 
    ground_truth_transcript = "\n".join([
        f"{'AGENTE' if t.speaker == SpeakerRole.AGENT else 'CLIENTE'}: {t.text}" for t in turns
    ])
 
    return ScenarioProfile(
        scenario_id=conv_id,
        category=category,
        client_name=client_name,
        account_number=account_num,
        id_number=id_num,
        initial_emotion=emotion,
        issue_description=issue_description,
        ground_truth_transcript=ground_truth_transcript,
    )
 
 
class BankingDataLoader:
    def __init__(self, data_path: str = os.path.join("data", "banking_conversations.csv")):
        self.data_path = self._resolve_path(data_path)
 
    def _resolve_path(self, path: str) -> str:
        if os.path.exists(path):
            return path
        base, _ = os.path.splitext(path)
        for ext in [".csv", ".json"]:
            candidate = f"{base}{ext}"
            if os.path.exists(candidate):
                return candidate
        default_path = os.path.join("data", "banking_conversations.csv")
        if os.path.exists(default_path):
            return default_path
        raise FileNotFoundError(f"No se encontró el archivo de datos en: {path}")
 
    def _read_records(self):
        """Generador perezoso de filas del CSV/JSON, para no cargar en
        memoria el fichero completo (crítico con ~5.5M filas)."""
        with open(self.data_path, mode="r", encoding="utf-8") as f:
            first_char = f.read(1)
        if first_char == "[":
            with open(self.data_path, mode="r", encoding="utf-8") as f:
                records = json.load(f)
            for row in records:
                yield row
        else:
            with open(self.data_path, mode="r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    yield row
 
    def load_sessions(self, limit_scenarios: Optional[int] = None) -> List[ConversationSession]:
        """Agrupa las filas por conversation_id y construye una
        ConversationSession por cada una.
 
        Nota: se asume que el CSV está ordenado por conversation_id (así lo
        está el Banking Conversation Corpus real). Si `limit_scenarios` está
        definido, se detiene la lectura en cuanto se ha completado esa
        cantidad de conversaciones distintas, evitando leer el fichero
        completo en ejecuciones de desarrollo/demo.
        """
        grouped: Dict[str, List[Turn]] = {}
        order: List[str] = []
        current_id = None
 
        for row in self._read_records():
            conv_id = str(row["conversation_id"]).strip()
 
            if conv_id != current_id:
                # Conversación nueva: si ya alcanzamos el límite, terminamos.
                if limit_scenarios is not None and len(order) >= limit_scenarios:
                    break
                current_id = conv_id
 
            if conv_id not in grouped:
                grouped[conv_id] = []
                order.append(conv_id)
 
            spk = SpeakerRole.CLIENT if str(row["speaker"]).strip().lower() == "client" else SpeakerRole.AGENT
            txt = str(row["text"]).strip()
            grouped[conv_id].append(Turn(speaker=spk, text=txt, date_time=row.get("date_time") or None))
 
        sessions: List[ConversationSession] = []
        for conv_id in order:
            turns = grouped[conv_id]
            profile = build_scenario_profile(conv_id, turns)
            sessions.append(ConversationSession(
                session_id=f"session_{conv_id}",
                scenario=profile,
                turns=turns,
                is_finished=False,
            ))
 
        return sessions
 
 
class DatasetLoader:
    """Capa fina sobre BankingDataLoader para procesos batch (EDA, ingesta)
    que solo necesitan los ScenarioProfile y quieren poder limitar cuántas
    conversaciones se procesan."""
 
    def __init__(self, data_path: str = os.path.join("data", "banking_conversations.csv")):
        self._banking_loader = BankingDataLoader(data_path)
 
    def load_and_process_scenarios(self, limit_scenarios: Optional[int] = None) -> List[ScenarioProfile]:
        sessions = self._banking_loader.load_sessions(limit_scenarios=limit_scenarios)
        return [s.scenario for s in sessions]