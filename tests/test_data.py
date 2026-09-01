import pytest
from pathlib import Path
from src.data.loader import BankingDataLoader
from src.data.models import SpeakerRole


def test_csv_loading_and_grouping(tmp_path):
    # 1. Crear un archivo CSV temporal de prueba
    csv_file = tmp_path / "test_corpus.csv"
    csv_content = (
        "conversation_id,speaker,date_time,text\n"
        'conv_001,agent,2023-09-06T14:33:33+00:00,"Buenos días, ¿en qué puedo ayudarle?"\n'
        'conv_001,client,2023-09-06T14:33:41+00:00,"Hola, necesito desbloquear mi tarjeta."\n'
    )
    csv_file.write_text(csv_content, encoding="utf-8")

    # 2. Cargar datos
    loader = BankingDataLoader(csv_file)
    sessions = loader.load_sessions()

    # 3. Assertions (Verificaciones)
    assert "conv_001" in sessions
    session = sessions["conv_001"]
    
    assert session.total_turns == 2
    assert session.turns[0].speaker == SpeakerRole.AGENT
    assert session.turns[1].speaker == SpeakerRole.CLIENT
    assert "desbloquear mi tarjeta" in session.turns[1].text


def test_transcript_formatting(tmp_path):
    csv_file = tmp_path / "test_corpus.csv"
    csv_content = (
        "conversation_id,speaker,date_time,text\n"
        'conv_002,agent,2023-09-06T14:33:33+00:00,"Hola"\n'
        'conv_002,client,2023-09-06T14:33:41+00:00,"Buenas"\n'
    )
    csv_file.write_text(csv_content, encoding="utf-8")

    loader = BankingDataLoader(csv_file)
    session = loader.get_session_by_id("conv_002")

    transcript = session.get_formatted_transcript()
    assert "Agente: Hola" in transcript
    assert "Cliente: Buenas" in transcript