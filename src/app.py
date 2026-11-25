"""
app.py — Streamlit Dashboard for the Drone Security Analyst Agent
-----------------------------------------------------------------
Tabs:
  1. 🎥 Live Patrol     — Frame viewer + telemetry HUD + live alerts
  2. 🔍 Frame Search    — Natural language vector search over frame archive
  3. 💬 Q&A Chat        — Conversational interface over patrol history
  4. 📋 Session Report  — Patrol summary + alert log export

Run:
    streamlit run src/app.py
"""

import json
import os
import sys
from pathlib import Path
from datetime import datetime

import streamlit as st
from PIL import Image

# Ensure src is on path
sys.path.insert(0, str(Path(__file__).parents[1]))

from src.ingestion.synthetic_frames import generate_dataset
from src.ingestion.telemetry_sync import generate_telemetry, get_telemetry_for_frame
from src.perception.vlm_captioner import get_captioner
from src.perception.embedder import FrameEmbedder
from src.indexing.vector_store import get_vector_store
from src.agent.context_manager import PatrolContextManager
from src.agent.security_rules import SecurityRuleEngine, Alert
from src.agent.summarizer import PatrolSummarizer, PatrolQA

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Drone Security Analyst",
    page_icon="🚁",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------
if "initialised" not in st.session_state:
    st.session_state.initialised = False
    st.session_state.frames_meta = []
    st.session_state.telemetry = []
    st.session_state.vector_store = None
    st.session_state.context_mgr = None
    st.session_state.alerts = []
    st.session_state.captions_cache = {}
    st.session_state.patrol_summary = ""
    st.session_state.entity_counts = {}
    st.session_state.qa = None
    st.session_state.chat_history = []

BASE_DIR = Path(__file__).resolve().parents[1]
FRAMES_DIR = BASE_DIR / "data" / "sample_frames"
TELEM_PATH = BASE_DIR / "data" / "telemetry_stream.json"


@st.cache_resource
def load_components():
    captioner = get_captioner()
    embedder = FrameEmbedder()
    store = get_vector_store()
    context = PatrolContextManager()
    rules = SecurityRuleEngine()
    summarizer = PatrolSummarizer()
    return captioner, embedder, store, context, rules, summarizer


def execute_patrol_session(num_frames: int, start_hour: int, session_id: str, show_progress: bool = True):
    Path(FRAMES_DIR).mkdir(parents=True, exist_ok=True)
    meta = generate_dataset(str(FRAMES_DIR), num_frames=num_frames, start_hour=start_hour)
    telem = generate_telemetry(num_frames=num_frames, start_time=None, output_path=str(TELEM_PATH))
    st.session_state.frames_meta = meta
    st.session_state.telemetry = telem

    captioner, embedder, store, context, rules, summarizer = load_components()
    if hasattr(store, "_records"):
        store._records.clear()
    context.reset()
    alerts = []
    entity_counts = {}
    start_ts = end_ts = None

    prog = st.progress(0, text="Processing frames …") if show_progress else None
    for i, m in enumerate(meta):
        frame_id = m["frame_id"]
        timestamp = m["timestamp"]
        zone = m["zone"]
        img_path = Path(FRAMES_DIR) / m["file"]
        pil_img = Image.open(img_path).convert("RGB")
        telem_rec = get_telemetry_for_frame(telem, frame_id)

        fc = captioner.caption(frame_id, pil_img)
        vec = embedder.embed_image(pil_img)
        store.upsert([{
            "frame_id": frame_id,
            "timestamp": timestamp,
            "drone_id": telem_rec.get("drone_id", "DRONE-01"),
            "patrol_zone": zone,
            "altitude_m": float(telem_rec.get("altitude_m", 25.0)),
            "battery_pct": int(telem_rec.get("battery_pct", 100)),
            "caption": fc.caption,
            "objects": ", ".join(fc.objects),
            "image_path": str(img_path),
            "vector": vec.tolist(),
        }])
        ctx_alerts = context.ingest(frame_id, timestamp, zone, fc.objects, fc.caption)
        frame_alerts = rules.evaluate(frame_id, timestamp, zone, fc.caption, fc.objects,
                                      telem_rec.get("altitude_m", 25.0), ctx_alerts)
        for a in frame_alerts:
            alerts.append({"level": a.level, "rule": a.rule, "message": a.message,
                           "frame_id": a.frame_id, "timestamp": a.timestamp, "zone": a.zone})
        st.session_state.captions_cache[frame_id] = fc
        for obj in fc.objects:
            entity_counts[obj] = entity_counts.get(obj, 0) + 1
        if start_ts is None:
            start_ts = timestamp
        end_ts = timestamp
        if prog:
            prog.progress((i + 1) / len(meta), text=f"Frame {frame_id}/{len(meta)-1} — {zone}")

    patrol_summary = summarizer.generate_summary(
        session_id=session_id,
        total_frames=len(meta),
        zones=[m["zone"] for m in meta],
        alerts=alerts,
        entity_counts=entity_counts,
        start_time=start_ts or "",
        end_time=end_ts or "",
    )
    st.session_state.vector_store = store
    st.session_state.context_mgr = context
    st.session_state.alerts = alerts
    st.session_state.patrol_summary = patrol_summary
    st.session_state.entity_counts = entity_counts
    st.session_state.initialised = True
    st.session_state.qa = PatrolQA(store, context, session_id)
    if prog:
        prog.empty()


# ---------------------------------------------------------------------------
# Sidebar — session control
# ---------------------------------------------------------------------------
st.sidebar.title("🚁 Drone Security Analyst")
st.sidebar.markdown("---")

session_id = st.sidebar.text_input("Session ID", value="SESSION-01")
num_frames = st.sidebar.slider("Synthetic frames to generate", 20, 120, 40, step=10)
start_hour = st.sidebar.slider("Patrol start hour (24h)", 0, 23, 22)

col_a, col_b = st.sidebar.columns(2)
run_session = col_a.button("▶ Run Patrol", type="primary", use_container_width=True)
reset_btn = col_b.button("🔄 Reset", use_container_width=True)

if reset_btn:
    for k in list(st.session_state.keys()):
        del st.session_state[k]
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown("**Backend Configuration**")
vlm = st.sidebar.selectbox("VLM Backend", ["mock", "blip2", "florence2"])
embed = st.sidebar.selectbox("Embedding Backend", ["mock", "clip"])
vec_backend = st.sidebar.selectbox("Vector Store", ["mock", "lancedb"])
os.environ["VLM_BACKEND"] = vlm
os.environ["EMBED_BACKEND"] = embed
os.environ["VECTOR_BACKEND"] = vec_backend

# ---------------------------------------------------------------------------
# Run or auto-initialize patrol session
# ---------------------------------------------------------------------------
if run_session:
    with st.spinner("Executing patrol mission simulation …"):
        execute_patrol_session(num_frames, start_hour, session_id, show_progress=True)
    st.success(f"✅ Patrol complete: {len(st.session_state.frames_meta)} frames | {len(st.session_state.alerts)} alerts")
elif not st.session_state.initialised:
    # Auto-initialize with default mission so dashboard is active immediately
    execute_patrol_session(num_frames=30, start_hour=22, session_id=session_id, show_progress=False)

# ---------------------------------------------------------------------------
# Main tabs
# ---------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs(["🎥 Live Patrol", "🔍 Frame Search", "💬 Q&A Chat", "📋 Session Report"])

# ── Tab 1: Live patrol viewer ──────────────────────────────────────────────
with tab1:
    if not st.session_state.initialised:
        st.info("👈 Configure settings in the sidebar and click **Run Patrol** to start.")
    else:
        meta = st.session_state.frames_meta
        frame_idx = st.slider("Select Frame", 0, len(meta) - 1, 0)
        m = meta[frame_idx]
        telem = get_telemetry_for_frame(st.session_state.telemetry, m["frame_id"])
        fc = st.session_state.captions_cache.get(m["frame_id"])

        col1, col2 = st.columns([2, 1])
        with col1:
            img = Image.open(Path(FRAMES_DIR) / m["file"])
            st.image(img, caption=f"Frame {m['frame_id']} — {m['zone']}", use_container_width=True)
            if fc:
                st.markdown(f"**🧠 VLM Caption:** {fc.caption}")
                if fc.objects:
                    st.markdown("**🏷️ Detected Objects:** " + " · ".join(f"`{o}`" for o in fc.objects))

        with col2:
            st.markdown("### 📡 Telemetry HUD")
            st.metric("Zone", telem.get("patrol_zone", "?"))
            st.metric("Altitude", f"{telem.get('altitude_m', 0):.1f} m")
            st.metric("Battery", f"{telem.get('battery_pct', 0)}%")
            st.metric("Speed", f"{telem.get('speed_ms', 0):.1f} m/s")
            st.metric("Heading", f"{telem.get('heading_deg', 0):.0f}°")
            st.metric("Timestamp", telem.get("timestamp", ""))

        # Alerts for this frame
        frame_alerts = [a for a in st.session_state.alerts if a["frame_id"] == m["frame_id"]]
        if frame_alerts:
            st.markdown("### 🚨 Active Alerts for this Frame")
            for a in frame_alerts:
                colour = {"CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🟡", "LOW": "⚪"}.get(a["level"], "")
                st.error(f"{colour} **{a['level']}** — {a['message']}")

# ── Tab 2: Frame search ────────────────────────────────────────────────────
with tab2:
    st.subheader("🔍 Natural Language Frame Search")
    if not st.session_state.initialised:
        st.info("Run a patrol session first.")
    else:
        query = st.text_input("Search query", placeholder="e.g. person near fence at night")
        zone_filter = st.text_input("Zone filter (optional)", "")
        obj_filter = st.text_input("Object filter (optional)", "")
        if st.button("Search") and query:
            embedder_inst = FrameEmbedder()
            qvec = embedder_inst.embed_text(query)
            results = st.session_state.vector_store.search(
                qvec, top_k=5,
                zone_filter=zone_filter or None,
                object_filter=obj_filter or None,
            )
            if not results:
                st.warning("No matching frames found.")
            else:
                for r in results:
                    with st.expander(f"Frame {r['frame_id']} | {r['timestamp']} | {r['patrol_zone']}"):
                        img_path = Path(FRAMES_DIR) / f"frame_{r['frame_id']:04d}.jpg"
                        if img_path.exists():
                            st.image(str(img_path), width=400)
                        st.markdown(f"**Caption:** {r['caption']}")
                        st.markdown(f"**Objects:** {r['objects']}")
                        st.markdown(f"**Altitude:** {r['altitude_m']} m | **Battery:** {r['battery_pct']}%")

# ── Tab 3: Q&A chat ────────────────────────────────────────────────────────
with tab3:
    st.subheader("💬 Patrol Q&A — Ask anything about this session")
    if not st.session_state.initialised or st.session_state.qa is None:
        st.info("Run a patrol session first.")
    else:
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])
        user_input = st.chat_input("Ask a question about the patrol …")
        if user_input:
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.write(user_input)
            with st.chat_message("assistant"):
                with st.spinner("Analysing …"):
                    answer = st.session_state.qa.ask(user_input)
                st.write(answer)
            st.session_state.chat_history.append({"role": "assistant", "content": answer})

# ── Tab 4: Session report ──────────────────────────────────────────────────
with tab4:
    st.subheader("📋 Patrol Mission Report")
    if not st.session_state.initialised:
        st.info("Run a patrol session first.")
    else:
        st.markdown("### Executive Summary")
        st.markdown(st.session_state.patrol_summary)

        st.markdown("### Entity Detection Summary")
        counts = st.session_state.entity_counts
        if counts:
            st.bar_chart(counts)

        st.markdown("### Security Alerts Log")
        alerts = st.session_state.alerts
        if alerts:
            for a in alerts:
                colour = {"CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🟡", "LOW": "⚪"}.get(a["level"], "")
                st.markdown(
                    f"{colour} **[{a['level']}]** Frame `{a['frame_id']}` | `{a['timestamp']}` | Zone: `{a['zone']}`  \n"
                    f"{a['message']}"
                )
        else:
            st.success("✅ No security alerts triggered during this patrol.")

        # Export
        export_data = json.dumps({
            "session_id": session_id,
            "summary": st.session_state.patrol_summary,
            "alerts": alerts,
            "entity_counts": counts,
        }, indent=2)
        st.download_button(
            "⬇️ Export Report (JSON)",
            data=export_data,
            file_name=f"patrol_report_{session_id}.json",
            mime="application/json",
        )
