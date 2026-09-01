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
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> float:
    """Similitud coseno entre los embeddings semánticos de dos textos.
    Devuelve un valor entre -1.0 y 1.0 (en la práctica, casi siempre 0-1)."""
    from sentence_transformers import SentenceTransformer, util
 
    if model_name not in _MODEL_CACHE:
        _MODEL_CACHE[model_name] = SentenceTransformer(model_name)
    model = _MODEL_CACHE[model_name]
 
    embeddings = model.encode([text_a, text_b], convert_to_tensor=True)
    score = util.cos_sim(embeddings[0], embeddings[1]).item()
    return score
 
 
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
            # No dejamos que un fallo aquí (p. ej. DLLs de torch bloqueadas
            # por Smart App Control en Windows, o el modelo aún
            # descargándose) rompa el resto de la app. Guardamos el motivo
            # para poder mostrarlo en la interfaz en vez de fallar en
            # silencio.
            result["semantic_similarity_error"] = f"{type(e).__name__}: {e}"
 
    return result
 
 
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
    }
 
    if use_semantic:
        try:
            result["semantic_similarity"] = semantic_similarity(simulated_text, ground_truth_transcript)
        except ImportError:
            result["semantic_similarity"] = None  # sentence-transformers no instalado
 
    return result