import sys
import os
import time
import json
from pathlib import Path
import streamlit as st
import requests

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION & HEALTH CHECKS
# ─────────────────────────────────────────────────────────────────────────────
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

def check_backend_health() -> bool:
    """Check if the FastAPI backend is running and healthy."""
    try:
        resp = requests.get(f"{BACKEND_URL}/")
        if resp.status_code == 200:
            return resp.json().get("status") == "healthy"
        return False
    except Exception:
        return False

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Legal AI Assistant — Agentic RAG",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM PREMIUM STYLING (Glassmorphism & Micro-animations)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@400;600;800&display=swap');

    /* Theme override */
    html, body, [data-testid="stAppViewContainer"] {
        font-family: 'Inter', sans-serif;
        background: linear-gradient(135deg, #0e121a 0%, #151b26 100%);
        color: #e2e8f0;
    }
    
    [data-testid="stSidebar"] {
        background-color: rgba(21, 27, 38, 0.95);
        border-right: 1px solid rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
    }

    h1, h2, h3, [data-testid="stHeader"] {
        font-family: 'Outfit', sans-serif;
        font-weight: 800;
        letter-spacing: -0.5px;
    }

    /* Gradient Title styling */
    .title-container {
        padding: 20px 0;
        background: linear-gradient(90deg, #6366f1 0%, #a855f7 50%, #ec4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 5px;
    }

    /* Message Bubbles styling */
    .chat-bubble {
        padding: 16px 20px;
        border-radius: 16px;
        margin-bottom: 12px;
        backdrop-filter: blur(8px);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        border: 1px solid rgba(255, 255, 255, 0.06);
    }
    
    .chat-user {
        background: rgba(99, 102, 241, 0.12);
        border-left: 4px solid #6366f1;
        margin-left: 20%;
    }
    
    .chat-assistant {
        background: rgba(255, 255, 255, 0.03);
        border-left: 4px solid #a855f7;
        margin-right: 20%;
    }
    
    .chat-bubble:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.15);
    }

    /* Premium card design */
    .premium-card {
        background: rgba(255, 255, 255, 0.02);
        padding: 24px;
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.05);
        margin-bottom: 20px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.12);
        transition: transform 0.2s;
    }
    .premium-card:hover {
        transform: scale(1.01);
        border-color: rgba(99, 102, 241, 0.2);
    }
    
    /* Interactive tags and highlights */
    .source-tag {
        display: inline-block;
        padding: 4px 10px;
        background: rgba(168, 85, 247, 0.15);
        color: #d8b4fe;
        border-radius: 8px;
        font-size: 0.85em;
        font-weight: 500;
        margin: 4px;
        border: 1px solid rgba(168, 85, 247, 0.3);
    }

    /* Custom buttons */
    .stButton>button {
        background: linear-gradient(90deg, #6366f1 0%, #a855f7 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: 600;
        transition: all 0.3s;
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.3);
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(99, 102, 241, 0.5);
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# INITIALIZATION
# ─────────────────────────────────────────────────────────────────────────────
backend_alive = check_backend_health()

if "messages" not in st.session_state:
    st.session_state.messages = []

# Clear cache helper to force rebuild / re-run
def rebuild_rag_system():
    st.cache_resource.clear()
    st.success("Frontend state refreshed successfully!")
    st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR / NAVIGATION & DOCUMENT UPLOADS
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("<div style='text-align: center; padding: 15px 0;'><h2 style='color:#a855f7; margin:0;'>⚖️ Legal Assistant</h2><p style='color:#94a3b8; font-size:0.9em;'>Powered by Agentic RAG</p></div>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Show Backend Connection Status
    if backend_alive:
        st.markdown("<div style='text-align: center; color: #10b981; font-weight: 600;'>🟢 Backend Server Online</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div style='text-align: center; color: #ef4444; font-weight: 600;'>🔴 Backend Server Offline</div>", unsafe_allow_html=True)
        st.info("💡 To start the backend, run this command in your terminal:\n`python src/server.py` or `uvicorn src.server:app --reload`")

    st.markdown("---")
    page = st.radio("Go to", ["💬 Chat Interface", "📊 Model Selection", "📈 Evaluation Report"])
    
    st.markdown("---")
    st.subheader("📥 Upload Legal Document")
    st.markdown("Dynamically ingest and chunk new contracts, briefs, or case law directly into the backend Vector Store.")
    
    # PDF/TXT Uploader
    uploaded_file = st.file_uploader("Choose PDF or TXT", type=["pdf", "txt"], disabled=not backend_alive)
    if uploaded_file is not None and backend_alive:
        if st.button("🚀 Ingest & Index Document"):
            with st.spinner("Uploading and indexing document on the backend..."):
                try:
                    # Make API call to backend /upload
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf" if uploaded_file.name.endswith(".pdf") else "text/plain")}
                    resp = requests.post(f"{BACKEND_URL}/upload", files=files)
                    
                    if resp.status_code == 200:
                        st.toast(f"Successfully indexed {uploaded_file.name}!", icon="✅")
                        time.sleep(1)
                        rebuild_rag_system()
                    else:
                        st.error(f"Failed to ingest document: {resp.json().get('detail', 'Unknown error')}")
                except Exception as e:
                    st.error(f"Failed to connect to backend: {e}")

    st.markdown("---")
    if st.button("♻️ Reset Index & Reload"):
        rebuild_rag_system()


# ─────────────────────────────────────────────────────────────────────────────
# PAGES
# ─────────────────────────────────────────────────────────────────────────────

if page == "💬 Chat Interface":
    st.markdown("<h1 class='title-container'>Agentic Legal Research & Contract Analyst</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#94a3b8; font-size:1.1em; margin-bottom: 24px;'>Ask complex multi-step questions. The Agentic loop will reformulate your query, perform iterative semantic retrieval, verify sufficiency, and deliver rigorous, cited analyses.</p>", unsafe_allow_html=True)

    if not backend_alive:
        st.warning("⚠️ Chat interface is disabled because the FastAPI Backend is offline. Please run the backend server first.")
    else:
        # Display chat history
        for msg in st.session_state.messages:
            role_class = "chat-user" if msg["role"] == "user" else "chat-assistant"
            role_icon = "👤" if msg["role"] == "user" else "⚖️ Agent"
            
            with st.chat_message(msg["role"]):
                st.markdown(f"<div class='chat-bubble {role_class}'><strong>{role_icon}:</strong><br><br>{msg['content']}</div>", unsafe_allow_html=True)
                if "sources" in msg and msg["sources"]:
                    # Custom chips for source citations
                    sources_html = "".join([f"<span class='source-tag'>📎 {s}</span>" for s in msg["sources"]])
                    st.markdown(f"<div style='margin-left: 15px;'>{sources_html}</div>", unsafe_allow_html=True)

        # Chat input
        if prompt := st.chat_input("Ask a legal question... (e.g. COVID-19 force majeure liability, California Labor Code 2870)"):
            with st.chat_message("user"):
                st.markdown(f"<div class='chat-bubble chat-user'><strong>👤 User:</strong><br><br>{prompt}</div>", unsafe_allow_html=True)
            
            st.session_state.messages.append({"role": "user", "content": prompt})

            # Compile chat history for backend request
            history_payload = []
            for m in st.session_state.messages[:-1]:
                history_payload.append({
                    "role": m["role"],
                    "content": m["content"]
                })

            with st.chat_message("assistant"):
                # UI placeholder for step-by-step agent tracking
                agent_status = st.empty()
                response_container = st.empty()
                
                with st.spinner("Analyzing and researching..."):
                    try:
                        agent_status.markdown("🕵️ *Running security guardrails & checking query parameters...*")
                        time.sleep(0.4)
                        
                        agent_status.markdown("🔍 *Activating LLM Agent ReAct loop and retrieving facts...*")
                        
                        # Call backend API
                        payload = {
                            "message": prompt,
                            "history": history_payload,
                            "model_provider": "gemini"
                        }
                        
                        resp = requests.post(f"{BACKEND_URL}/chat", json=payload)
                        
                        agent_status.markdown("🧐 *Evaluating context sufficiency & synthesizing analysis...*")
                        time.sleep(0.4)
                        
                        if resp.status_code == 200:
                            result_data = resp.json()
                            answer = result_data.get("answer", "")
                            sources = result_data.get("sources", [])
                        else:
                            detail = resp.json().get("detail", "Unknown server error")
                            raise Exception(detail)
                        
                        agent_status.markdown("✍️ *Rendering grounded legal citation...*")
                        
                        # STREAMING SIMULATION (Real-time token generation look & feel)
                        streamed_text = ""
                        for word in answer.split(" "):
                            streamed_text += word + " "
                            response_container.markdown(f"<div class='chat-bubble chat-assistant'><strong>⚖️ Agent:</strong><br><br>{streamed_text}▌</div>", unsafe_allow_html=True)
                            time.sleep(0.015)
                        
                        response_container.markdown(f"<div class='chat-bubble chat-assistant'><strong>⚖️ Agent:</strong><br><br>{answer}</div>", unsafe_allow_html=True)
                        
                        if sources:
                            sources_html = "".join([f"<span class='source-tag'>📎 {s}</span>" for s in sources])
                            st.markdown(f"<div style='margin-left: 15px;'>{sources_html}</div>", unsafe_allow_html=True)
                        
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": answer,
                            "sources": sources
                        })
                        
                        agent_status.empty()
                        st.rerun()

                    except Exception as e:
                        st.error(f"Error during agentic execution: {e}")
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": f"I encountered an error: {e}"
                        })

elif page == "📊 Model Selection":
    st.markdown("<h1 class='title-container'>Model Selection Report</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#94a3b8; font-size:1.1em;'>HELM LegalBench benchmarks, Artificial Analysis latency metrics, and LLM Arena Elo records driving our architected multi-provider decision matrix.</p>", unsafe_allow_html=True)
    
    try:
        with open("outputs/model_selection_report.json", "r") as f:
            model_report = json.load(f)
        
        st.markdown(f"""
        <div class='premium-card' style='border-left: 5px solid #a855f7;'>
            <h3 style='margin-top:0; color:#d8b4fe;'>🏆 Final Architected Recommendation</h3>
            <p style='font-size:1.1em; line-height:1.6; color:#e2e8f0;'>{model_report.get('final_recommendation', '')}</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.subheader("Decision Matrix by Scenario")
        cols = st.columns(len(model_report.get("decision_matrix", {})))
        for i, (task, model) in enumerate(model_report.get("decision_matrix", {}).items()):
            cols[i].markdown(f"""
            <div class='premium-card' style='text-align:center;'>
                <h4 style='color:#94a3b8; margin:0;'>{task.replace('task_', '').replace('_', ' ').title()}</h4>
                <p style='color:#6366f1; font-weight:700; margin-top:8px; font-size:1.1em;'>{model.split(' — ')[0]}</p>
                <p style='font-size:0.85em; color:#64748b;'>{model.split(' — ')[1] if ' — ' in model else ''}</p>
            </div>
            """, unsafe_allow_html=True)

        st.subheader("Raw Benchmark Metrics JSON")
        st.json(model_report)
        
    except FileNotFoundError:
        st.warning("Model selection report not found. Run `python src/model_selection.py` to generate it.")

elif page == "📈 Evaluation Report":
    st.markdown("<h1 class='title-container'>RAG Evaluation Dashboard</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#94a3b8; font-size:1.1em;'>Automated performance, token efficiency, financial metrics, and heuristic faithfulness scores from the active capstone test suite.</p>", unsafe_allow_html=True)
    
    # Add trigger evaluation button
    if backend_alive:
        if st.button("📊 Re-Run Evaluation Suite"):
            with st.spinner("Triggering evaluation benchmark on backend..."):
                try:
                    resp = requests.post(f"{BACKEND_URL}/evaluate")
                    if resp.status_code == 200:
                        st.success("Evaluation suite triggered! Running in background. Please wait a minute and refresh.")
                    else:
                        st.error(f"Failed to trigger evaluation: {resp.json().get('detail', 'Unknown error')}")
                except Exception as e:
                    st.error(f"Failed to connect to backend: {e}")

    try:
        with open("outputs/eval_report.json", "r") as f:
            eval_report = json.load(f)
        
        col1, col2, col3, col4 = st.columns(4)
        col1.markdown(f"""
        <div class='premium-card' style='text-align:center;'>
            <h2 style='color:#6366f1; margin:0;'>{eval_report.get('total_queries', 0)}</h2>
            <p style='color:#94a3b8; margin:5px 0 0 0;'>Total Queries</p>
        </div>
        """, unsafe_allow_html=True)
        
        col2.markdown(f"""
        <div class='premium-card' style='text-align:center;'>
            <h2 style='color:#a855f7; margin:0;'>{eval_report.get('avg_faithfulness', 0) * 100:.0f}%</h2>
            <p style='color:#94a3b8; margin:5px 0 0 0;'>Avg Faithfulness</p>
        </div>
        """, unsafe_allow_html=True)
        
        col3.markdown(f"""
        <div class='premium-card' style='text-align:center;'>
            <h2 style='color:#ec4899; margin:0;'>{eval_report.get('task_success_rate', 0) * 100:.0f}%</h2>
            <p style='color:#94a3b8; margin:5px 0 0 0;'>Task Success Rate</p>
        </div>
        """, unsafe_allow_html=True)
        
        col4.markdown(f"""
        <div class='premium-card' style='text-align:center;'>
            <h2 style='color:#10b981; margin:0;'>{eval_report.get('avg_latency_ms', 0):.0f}ms</h2>
            <p style='color:#94a3b8; margin:5px 0 0 0;'>Avg Latency</p>
        </div>
        """, unsafe_allow_html=True)

        st.subheader("Detailed Evaluation Results")
        st.json(eval_report)
        
    except FileNotFoundError:
        st.warning("Evaluation report not found. Run `python src/rag_pipeline.py --eval` or click the button above to generate it.")