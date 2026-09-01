"""
===============================================================================
Trabajo Fin de Máster (TFM) - Inteligencia Artificial
Autor: Manuel Cid Gómez
Módulo Interfaz: app.py
Descripción: Aplicación principal en Streamlit para el simulador interactivo
             de atención al cliente bancario y auditoría automatizada SOP,
             conectado al Banking Conversation Corpus mediante src.data.loader.
===============================================================================
"""

import os
import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

from src.data.models import ConversationSession, SpeakerRole, ScenarioProfile
from src.data.loader import BankingDataLoader
from src.data.repository import ConversationRepository
from src.agents.client_simulator import ClientSimulator
from src.agents.sop_evaluator import SOPEvaluator
from src.utils.evaluation_metrics import evaluate_roleplay_fidelity

# -----------------------------------------------------------------------------
# 1. CONFIGURACIÓN DE LA APLICACIÓN Y CARGA DE DATOS
# -----------------------------------------------------------------------------
load_dotenv()

st.set_page_config(
    page_title="Simulador Roleplay Bancario - TFM Manuel Cid",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🏦 Sistema de Formación y Simulación de Atención al Cliente Bancario")
st.caption("Entorno de Roleplay Interactivo y Auditoría Automatizada de Protocolos Operativos Estándar (SOP)")

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    st.error("⚠️ No se detectó la clave GOOGLE_API_KEY en las variables de entorno (.env).")
    st.stop()


# Nº de conversaciones del corpus que se cargan para la demo interactiva.
# El corpus completo tiene ~300k conversaciones (~5.5M turnos); cargarlo
# entero en cada arranque de Streamlit no es viable para un demo en vivo.
# El procesamiento del corpus completo se hace aparte, de forma batch,
# con ingest_data.py -> scenarios.db.
DEMO_SCENARIOS_LIMIT = 200


@st.cache_resource
def init_repository() -> ConversationRepository:
    data_path = os.path.join("data", "banking_conversations.csv")
    if not os.path.exists(data_path):
        st.error(f"⚠️ No se encontró el archivo de corpus en: {data_path}")
        st.stop()

    loader = BankingDataLoader(data_path)
    sessions = loader.load_sessions(limit_scenarios=DEMO_SCENARIOS_LIMIT)
    return ConversationRepository(sessions)


repo = init_repository()


# -----------------------------------------------------------------------------
# 2. INICIALIZACIÓN Y CONFIGURACIÓN DEL MODELO GENERATIVO
# -----------------------------------------------------------------------------
@st.cache_resource
def get_llm(key: str) -> ChatGoogleGenerativeAI:
    """
    Evalúa secuencialmente la disponibilidad de modelos Gemini para garantizar 
    la ejecución continuada del agente conversacional ante variaciones de cuota.
    """
    # Google retiró toda la línea "gemini-2.0-flash*" el 31/03/2026.
    # Lista actualizada: primero la generación 2.5 (más probada con LangChain),
    # y como respaldo la 3.x (nombres que el propio error 404 de Google
    # recomienda como sustitutos). Si check_models.py te muestra otros
    # nombres disponibles para tu API key, añádelos aquí.
    candidate_models = [
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite",
        "gemini-3.6-flash",
        "gemini-3.5-flash-lite",
    ]
    
    errors = []
    for model_name in candidate_models:
        try:
            llm_instance = ChatGoogleGenerativeAI(
                model=model_name,
                google_api_key=key,
                temperature=0.7,
                max_retries=2
            )
            llm_instance.invoke("ping")
            return llm_instance
        except Exception as err:
            errors.append(f"{model_name}: {err}")
            continue

    raise RuntimeError(
        f"No se pudo conectar a ningún modelo de Gemini disponible.\nDetalle:\n" + "\n".join(errors)
    )

llm = get_llm(api_key)


# -----------------------------------------------------------------------------
# 3. BARRA LATERAL: SELECCIÓN Y FICHA DEL ESCENARIO
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Configuración del Escenario")

    # Escenarios reales, extraídos y clasificados automáticamente por
    # BankingDataLoader a partir del corpus (ver src/data/loader.py).
    all_sessions = repo.get_all()

    if not all_sessions:
        st.error("⚠️ No se cargó ningún escenario del corpus.")
        st.stop()

    category_filter = st.selectbox(
        "Filtrar por categoría:",
        options=["Todas"] + sorted({s.scenario.category for s in all_sessions}),
    )

    filtered_sessions = [
        s for s in all_sessions
        if category_filter == "Todas" or s.scenario.category == category_filter
    ]

    scenario_mapping = {
        s.scenario.scenario_id: f"{s.scenario.client_name} — {s.scenario.category} ({s.scenario.scenario_id[:8]}...)"
        for s in filtered_sessions
    }

    selected_label = st.selectbox(
        "Selecciona el caso de uso a simular:",
        options=list(scenario_mapping.values())
    )

    selected_id = [k for k, v in scenario_mapping.items() if v == selected_label][0]
    raw_session = repo.get_by_id(selected_id)
    profile = raw_session.scenario

    st.subheader("📋 Ficha del Cliente Simulado")
    st.write(f"**Nombre:** {profile.client_name}")
    st.write(f"**Categoría:** {profile.category}")
    st.write(f"**Estado Emocional:** {profile.initial_emotion}")
    st.write(f"**Nº Cuenta:** `{profile.account_number}`")
    st.write(f"**DNI / SSN (4 dígitos):** `{profile.id_number}`")
    st.write(f"**ID Dataset:** `{profile.scenario_id[:8]}...`")
    st.info(f"**Motivo de la llamada:** {profile.issue_description}")

    st.divider()

    if st.button("🔄 Reiniciar Sesión de Roleplay", use_container_width=True):
        st.session_state.clear()
        st.rerun()


# -----------------------------------------------------------------------------
# 4. GESTIÓN DEL ESTADO Y CANAL DE CHAT
# -----------------------------------------------------------------------------
if "session" not in st.session_state or st.session_state.get("current_scenario") != profile.scenario_id:
    st.session_state.current_scenario = profile.scenario_id
    st.session_state.session = ConversationSession(
        session_id=f"sim_{profile.scenario_id}",
        scenario=profile
    )
    st.session_state.client_simulator = ClientSimulator(llm=llm, scenario=profile)
    st.session_state.evaluation = None
    st.session_state.fidelidad = None
    st.session_state.fidelidad_crash = None

session: ConversationSession = st.session_state.session
client_simulator: ClientSimulator = st.session_state.client_simulator

chat_container = st.container()

with chat_container:
    for turn in session.turns:
        if turn.speaker == SpeakerRole.AGENT:
            role_icon, role_name, avatar_type = "👨‍💼", "Agente (Tú)", "user"
        else:
            role_icon, role_name, avatar_type = "👤", f"Cliente ({profile.client_name})", "assistant"

        with st.chat_message(avatar_type, avatar=role_icon):
            st.markdown(f"**{role_name}:** {turn.text}")

if not session.is_finished:
    if prompt := st.chat_input("Escribe tu respuesta como gestor bancario..."):
        session.add_turn(speaker=SpeakerRole.AGENT, text=prompt)

        with st.spinner(f"{profile.client_name} está escribiendo..."):
            try:
                history_turns = session.turns[:-1]
                client_text = client_simulator.generate_response(
                    turns=history_turns,
                    user_input=prompt
                )
            except Exception as e:
                client_text = "Disculpe, la señal se entrecortó por un segundo. ¿Podría repetirme lo último?"
                st.error(f"Error en la generación del simulador: {e}")

        session.add_turn(speaker=SpeakerRole.CLIENT, text=client_text)
        st.rerun()


# -----------------------------------------------------------------------------
# 5. AUDITORÍA Y EVALUACIÓN DE CUMPLIMIENTO SOP
# -----------------------------------------------------------------------------
st.divider()
col_left, col_right = st.columns([3, 1])

with col_right:
    if not session.is_finished and len(session.turns) > 0:
        if st.button("🛑 Finalizar Llamada y Evaluar", type="primary", use_container_width=True):
            session.is_finished = True

            with st.spinner("Auditando conversación según los Protocolos Operativos Estándar (SOP)..."):
                try:
                    evaluator = SOPEvaluator(llm=llm)
                    st.session_state.evaluation = evaluator.evaluate_session(session)
                except Exception as e:
                    st.error(f"Error durante la evaluación: {e}")

            # Métrica de PLN: fidelidad del cliente simulado frente al
            # ground_truth_transcript real del escenario.
            # Métrica de PLN: fidelidad del cliente simulado frente al
            # ground_truth_transcript real del escenario. Se aísla en su
            # propio try/except para que un fallo aquí (p. ej. problemas de
            # carga de torch/sentence-transformers en Windows) nunca impida
            # ver el informe SOP, que es la parte principal. El error (si lo
            # hay) se guarda en session_state para que se siga viendo tras
            # recargas, en vez de solo en el instante del fallo.
            with st.spinner("Calculando métricas de PLN (similitud léxica y semántica)..."):
                try:
                    client_turns = [t.text for t in session.turns if t.speaker == SpeakerRole.CLIENT]
                    st.session_state.fidelidad = evaluate_roleplay_fidelity(
                        client_turns, profile.ground_truth_transcript
                    )
                    st.session_state.fidelidad_crash = None
                except Exception as e:
                    import traceback
                    st.session_state.fidelidad = None
                    st.session_state.fidelidad_crash = traceback.format_exc()

if session.is_finished and st.session_state.evaluation:
    report = st.session_state.evaluation

    st.header("📊 Informe de Auditoría Operativa y Calidad (SOP)")

    score_col, status_col = st.columns(2)
    with score_col:
        st.metric(label="Puntuación Global de Cumplimiento", value=f"{report.overall_score} / 10")

        # Barra de progreso visual, coloreada según el rango de la nota:
        # verde (≥8), amarillo (5-7), rojo (<5). Streamlit no permite
        # elegir el color de st.progress directamente, así que se simula
        # con un pequeño bloque HTML/CSS.
        score_pct = max(0, min(report.overall_score, 10)) / 10
        if report.overall_score >= 8:
            bar_color = "#2ecc71"  # verde
        elif report.overall_score >= 5:
            bar_color = "#f1c40f"  # amarillo
        else:
            bar_color = "#e74c3c"  # rojo

        st.markdown(
            f"""
            <div style="background-color:#e0e0e0; border-radius:8px; height:14px; width:100%; margin-top:4px;">
                <div style="background-color:{bar_color}; width:{score_pct*100:.0f}%; height:100%; border-radius:8px;"></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with status_col:
        if report.sop_adherence_status == "Cumplido":
            st.success(f"✅ ESTADO: {report.sop_adherence_status}")
        elif report.sop_adherence_status == "Parcialmente Cumplido":
            st.warning(f"⚠️ ESTADO: {report.sop_adherence_status}")
        else:
            st.error(f"❌ ESTADO: {report.sop_adherence_status}")

    st.subheader("📋 Verificación de Protocolos Clave")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Saludo Institucional", "✅ Cumplido" if report.greeting_check else "❌ Omitido")
    c2.metric("Verificación Identidad", "✅ Cumplido" if report.identity_verified else "❌ Omitido")
    c3.metric("Guía Canal Digital", "✅ Cumplido" if report.digital_channel_guidance else "❌ Omitido")
    c4.metric("Cierre Profesional", "✅ Cumplido" if report.closing_check else "❌ Omitido")

    # Métricas de PLN: fidelidad del cliente simulado frente al diálogo real
    # del escenario (ver src/utils/evaluation_metrics.py).
    if st.session_state.fidelidad:
        st.subheader("🧪 Métricas de PLN: Fidelidad del Roleplay")
        m1, m2 = st.columns(2)
        m1.metric("Similitud léxica (Jaccard)", f"{st.session_state.fidelidad['lexical_overlap']:.2f}")
        sem = st.session_state.fidelidad["semantic_similarity"]
        sem_error = st.session_state.fidelidad.get("semantic_similarity_error")
        m2.metric("Similitud semántica (embeddings)", f"{sem:.2f}" if sem is not None else "N/A")
        if sem_error:
            st.caption(f"⚠️ Similitud semántica no disponible: {sem_error}")

    st.divider()

    st.subheader("💡 Calidad de la Solución Aportada")
    st.info(report.resolution_quality)

    col_str, col_imp = st.columns(2)
    with col_str:
        st.subheader("💪 Fortalezas Identificadas")
        for strength in report.strengths:
            st.success(f"• {strength}")

    with col_imp:
        st.subheader("⚠️ Áreas de Mejora Operativa")
        for area in report.improvement_areas:
            st.warning(f"• {area}")

    st.subheader("📝 Observaciones Detalladas del Auditor Automatizado")
    st.write(report.detailed_feedback)