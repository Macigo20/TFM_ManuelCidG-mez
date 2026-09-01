"""
Módulo de repositorio para la persistencia de escenarios en SQLite.
"""
import sqlite3
from typing import Dict, List, Optional
from src.data.models import ConversationSession, ScenarioProfile
 
 
class ConversationRepository:
    def __init__(self, sessions: List[ConversationSession]):
        self._sessions: Dict[str, ConversationSession] = {
            s.scenario.scenario_id: s for s in sessions
        }
 
    def get_by_id(self, session_id: str) -> Optional[ConversationSession]:
        return self._sessions.get(session_id)
 
    def get_all(self) -> List[ConversationSession]:
        return list(self._sessions.values())
 
 
class ScenarioRepository:
    """Persistencia de ScenarioProfile en una base de datos SQLite."""
 
    _CREATE_TABLE_SQL = """
        CREATE TABLE IF NOT EXISTS scenarios (
            scenario_id TEXT PRIMARY KEY,
            client_name TEXT,
            category TEXT,
            issue_description TEXT,
            account_number TEXT,
            id_number TEXT,
            initial_emotion TEXT,
            ground_truth_transcript TEXT
        )
    """
 
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._conn = sqlite3.connect(self.db_path)
        self._conn.execute(self._CREATE_TABLE_SQL)
        self._conn.commit()
 
    def save_scenario(self, scenario: ScenarioProfile) -> None:
        self._conn.execute(
            """
            INSERT INTO scenarios (
                scenario_id, client_name, category, issue_description,
                account_number, id_number, initial_emotion, ground_truth_transcript
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(scenario_id) DO UPDATE SET
                client_name=excluded.client_name,
                category=excluded.category,
                issue_description=excluded.issue_description,
                account_number=excluded.account_number,
                id_number=excluded.id_number,
                initial_emotion=excluded.initial_emotion,
                ground_truth_transcript=excluded.ground_truth_transcript
            """,
            (
                scenario.scenario_id,
                scenario.client_name,
                scenario.category,
                scenario.issue_description,
                scenario.account_number,
                scenario.id_number,
                scenario.initial_emotion,
                scenario.ground_truth_transcript,
            ),
        )
        self._conn.commit()
 
    def get_all_scenarios(self) -> List[ScenarioProfile]:
        cursor = self._conn.execute(
            "SELECT scenario_id, client_name, category, issue_description, "
            "account_number, id_number, initial_emotion, ground_truth_transcript FROM scenarios"
        )
        columns = [d[0] for d in cursor.description]
        return [ScenarioProfile(**dict(zip(columns, row))) for row in cursor.fetchall()]
 
    def get_by_id(self, scenario_id: str) -> Optional[ScenarioProfile]:
        cursor = self._conn.execute(
            "SELECT scenario_id, client_name, category, issue_description, "
            "account_number, id_number, initial_emotion, ground_truth_transcript "
            "FROM scenarios WHERE scenario_id = ?",
            (scenario_id,),
        )
        row = cursor.fetchone()
        if row is None:
            return None
        columns = [d[0] for d in cursor.description]
        return ScenarioProfile(**dict(zip(columns, row)))
 
    def close(self) -> None:
        self._conn.close()
