"""
Script de diagnóstico para listar los modelos disponibles en tu API Key de Gemini.
Ejecutar con: python check_models.py
"""

import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    print("❌ Error: GOOGLE_API_KEY no encontrada en el archivo .env")
    exit(1)

genai.configure(api_key=api_key)

print("🔍 Consultando modelos disponibles para tu clave de API...\n")
try:
    available_models = []
    for model in genai.list_models():
        if "generateContent" in model.supported_generation_methods:
            print(f"  • ID: {model.name}")
            available_models.append(model.name)

    if not available_models:
        print("\n⚠️ La API respondió pero no hay modelos con soporte para 'generateContent'.")
    else:
        print(f"\n✅ Total de modelos compatibles encontrados: {len(available_models)}")

except Exception as e:
    print(f"❌ Error al conectar con la API de Google: {e}")