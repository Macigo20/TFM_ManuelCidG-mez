import os
import pandas as pd
from src.data.loader import DatasetLoader

# Lista de rutas posibles donde puede ubicarse el dataset
CANDIDATE_PATHS = [
    os.path.join("data", "banking_conversations.csv"),
    os.path.join("src", "data", "banking_conversations.csv"),
    "banking_conversations.csv",
    "dataset_sample.json",
    os.path.join("data", "dataset_sample.json")
]


def resolve_dataset_path() -> str:
    """Busca y retorna la primera ruta existente dentro de los candidatos."""
    for path in CANDIDATE_PATHS:
        if os.path.exists(path):
            return path
    return None


def run_eda():
    """Ejecuta el análisis exploratorio de datos (EDA) sobre el corpus bancario."""
    file_path = resolve_dataset_path()

    if not file_path:
        print("❌ No se encontró el dataset en ninguna de las siguientes rutas:")
        for p in CANDIDATE_PATHS:
            print(f"   - {p}")
        return

    print("==================================================")
    print(f" 📊 ANÁLISIS EXPLORATORIO DE DATOS (EDA) - TFM")
    print(f" Archivo detectado: {file_path}")
    print("==================================================")

    # Ingesta con Pandas según la extensión del archivo
    if file_path.endswith(".csv"):
        df = pd.read_csv(file_path)
    else:
        df = pd.read_json(file_path)

    total_utterances = len(df)
    unique_conversations = df["conversation_id"].nunique()
    speaker_counts = df["speaker"].value_counts().to_dict()

    # Métrica de longitud de texto
    df["word_count"] = df["text"].astype(str).apply(lambda x: len(x.split()))
    avg_words_per_turn = df["word_count"].mean()

    print(f"• Total de turnos/interacciones (Utterances): {total_utterances}")
    print(f"• Total de conversaciones únicas: {unique_conversations}")
    print(f"• Promedio de turnos por conversación: {total_utterances / unique_conversations:.2f}")
    print(f"• Distribución por emisor (Speaker): {speaker_counts}")
    print(f"• Promedio de palabras por turno: {avg_words_per_turn:.2f}")

    # Procesamiento estructurado a través de DatasetLoader
    loader = DatasetLoader(file_path)
    scenarios = loader.load_and_process_scenarios()

    categories = {}
    for s in scenarios:
        categories[s.category] = categories.get(s.category, 0) + 1

    print("\n• Distribución de Categorías de Incidencias Extraídas:")
    for cat, count in categories.items():
        print(f"  - {cat}: {count} escenarios ({count / len(scenarios) * 100:.1f}%)")
    print("==================================================\n")


if __name__ == "__main__":
    run_eda()