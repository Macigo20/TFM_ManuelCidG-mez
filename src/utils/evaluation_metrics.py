"""
Métricas de PLN para evaluar la fidelidad del cliente simulado frente a la
transcripción de referencia (ground truth) del corpus original.
 
"""

import re
from typing import List, Optional
 
_MODEL_CACHE = {}
 
 
def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-záéíóúñ0-9]+", text.lower())
 
 
def lexical_overlap_score(text_a: str, text_b: str) -> float:
    """Similitud de Jaccard entre los conjuntos de palabras de dos textos.
    Devuelve un valor entre 0.0 (sin solapamiento) y 1.0 (mismas palabras)."""
    tokens_a = set(_tokenize(text_a))
    tokens_b = set(_tokenize(text_b))
    if not tokens_a or not tokens_b:
        return 0.0
    interseccion = tokens_a & tokens_b
    union = tokens_a | tokens_b
    return len(interseccion) / len(union)
 
 
def semantic_similarity(
    text_a: str,
    text_b: str,
    model_name: str = "models/gemini-embedding-001",
) -> float:
    """Similitud coseno entre los embeddings semánticos de dos textos.
 
    Calculados mediante la API de Google Generative AI (la misma
    GOOGLE_API_KEY que ya usa el modelo conversacional Gemini), en vez de
    con un modelo local de sentence-transformers/torch. Esto evita
    depender de la instalación local de PyTorch, que en algunos entornos
    Windows puede bloquearse por políticas de seguridad (Smart App
    Control) al cargar sus DLLs nativas. Devuelve un valor entre -1.0 y
    1.0 (en la práctica, casi siempre 0-1).
    """
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    import numpy as np
 
    if model_name not in _MODEL_CACHE:
        _MODEL_CACHE[model_name] = GoogleGenerativeAIEmbeddings(model=model_name)
    embedder = _MODEL_CACHE[model_name]
 
    vector_a, vector_b = embedder.embed_documents([text_a, text_b])
    a, b = np.array(vector_a), np.array(vector_b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
 
 
def evaluate_roleplay_fidelity(
    simulated_turns: List[str],
    ground_truth_transcript: str,
    use_semantic: bool = True,
) -> dict:
    """Evalúa la fidelidad de una sesión de roleplay comparando las
    intervenciones del cliente simulado contra el ground_truth_transcript
    original de ese escenario.
 
    `simulated_turns`: lista de textos generados por ClientSimulator durante
    la sesión (turno a turno).
    """
    simulated_text = " ".join(simulated_turns)
 
    result = {
        "lexical_overlap": lexical_overlap_score(simulated_text, ground_truth_transcript),
        "semantic_similarity": None,
        "semantic_similarity_error": None,
    }
 
    if use_semantic:
        try:
            result["semantic_similarity"] = semantic_similarity(simulated_text, ground_truth_transcript)
        except Exception as e:
            # No dejamos que un fallo aquí (p. ej. problemas de red, cuota
            # de la API agotada, o clave GOOGLE_API_KEY no configurada)
            # rompa el resto de la app. Guardamos el motivo para poder
            # mostrarlo en la interfaz en vez de fallar en silencio.
            result["semantic_similarity_error"] = f"{type(e).__name__}: {e}"
 
    return result