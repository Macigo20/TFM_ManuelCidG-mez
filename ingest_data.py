import os
from src.data.loader import DatasetLoader
from src.data.repository import ScenarioRepository


DATASET_PATH = os.path.join("data", "muestra_3000_conversaciones.csv")
DB_PATH = "scenarios.db"
SCENARIOS_TO_INGEST = None  # None = procesar todas las conversaciones del fichero


def main():
    if not os.path.exists(DATASET_PATH):
        print(f"❌ No se encontró el dataset en: {DATASET_PATH}")
        return

    print("==================================================")
    print(" 🗄️ PROCESO DE INGESTA Y POBLADO DE BASE DE DATOS")
    print("==================================================")
    
    etiqueta_cantidad = SCENARIOS_TO_INGEST if SCENARIOS_TO_INGEST is not None else "todos los"
    print(f"1. Procesando {etiqueta_cantidad} escenarios desde {DATASET_PATH}...")
    loader = DatasetLoader(DATASET_PATH)
    scenarios = loader.load_and_process_scenarios(limit_scenarios=SCENARIOS_TO_INGEST)
    print(f"   ✓ Extraídos {len(scenarios)} escenarios estructurados.")

    print(f"2. Conectando a la base de datos relacional '{DB_PATH}'...")
    repo = ScenarioRepository(DB_PATH)

    print("3. Guardando escenarios en la tabla 'scenarios'...")
    for scenario in scenarios:
        repo.save_scenario(scenario)

    print(f"✅ Ingesta finalizada. Base de datos lista en: {os.path.abspath(DB_PATH)}")
    print("==================================================\n")


if __name__ == "__main__":
    main()