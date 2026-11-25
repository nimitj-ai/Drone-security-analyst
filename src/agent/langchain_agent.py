"""
langchain_agent.py
------------------
LangChain Security Analyst Agent with tool-calling capabilities.

The agent has access to 4 tools:
  1. search_frame_archive  — vector-similarity search over indexed frames
  2. get_loiter_history    — query loitering records from context manager
  3. get_vehicle_visits    — query vehicle reconnaissance records
  4. generate_alert        — formally log a security alert

The agent uses an LLM to decide which tools to call, synthesize context,
and produce natural-language responses to operator queries.

Supported LLM backends (set LANGCHAIN_LLM env var):
  - "openai"   : GPT-4o (requires OPENAI_API_KEY)
  - "mock"     : Returns deterministic canned responses (default, no API key)
"""

import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import tool
from langchain_core.messages import HumanMessage, AIMessage


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are an autonomous Drone Security Analyst Agent for a property patrol system.

Your responsibilities:
1. Monitor incoming drone patrol frame data (captions, telemetry, detected objects)
2. Identify and classify security threats: loitering, reconnaissance, perimeter breaches, and abandoned objects
3. Answer security operator questions using the frame archive and telemetry history
4. Generate concise, actionable security alerts prioritised by severity (LOW → CRITICAL)
5. Produce a complete patrol mission summary when requested

Always be factual. If you cannot determine something from available data, say so.
Reference specific frame IDs, timestamps, and zones in your responses.
When in doubt, escalate rather than dismiss a potential threat.

Current patrol session: {session_id}
Mission start: {mission_start}
"""

# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------
# These are module-level references so they can be patched per-session
_vector_store = None
_context_mgr = None
_rule_engine = None
_alerts_log: List[Dict] = []


def _init_tools(vector_store, context_mgr, rule_engine):
    global _vector_store, _context_mgr, _rule_engine
    _vector_store = vector_store
    _context_mgr = context_mgr
    _rule_engine = rule_engine


@tool
def search_frame_archive(query: str, zone: str = "", object_type: str = "") -> str:
    """
    Search the drone patrol frame archive using a natural language query.
    Returns the top matching frames with captions, timestamps, and zones.

    Args:
        query:       Natural language search query (e.g. 'person near fence at night')
        zone:        Optional zone filter (e.g. 'North Perimeter Gate')
        object_type: Optional object filter (e.g. 'vehicle', 'backpack')
    """
    if _vector_store is None:
        return "Vector store not initialised."
    from src.perception.embedder import FrameEmbedder
    embedder = FrameEmbedder()
    qvec = embedder.embed_text(query)
    results = _vector_store.search(
        qvec, top_k=5,
        zone_filter=zone or None,
        object_filter=object_type or None,
    )
    if not results:
        return "No matching frames found for that query."
    lines = []
    for r in results:
        lines.append(
            f"Frame {r['frame_id']} | {r['timestamp']} | Zone: {r['patrol_zone']}\n"
            f"  Caption: {r['caption']}\n"
            f"  Objects: {r['objects']}"
        )
    return "\n\n".join(lines)


@tool
def get_loiter_history(zone: str = "") -> str:
    """
    Retrieve loitering records from the current patrol session.
    Returns duration, entity type, and frame IDs for each loitering event.

    Args:
        zone: Optional zone to filter by (empty = all zones)
    """
    if _context_mgr is None:
        return "Context manager not initialised."
    records = _context_mgr.get_loiter_records()
    if zone:
        records = [r for r in records if zone.lower() in r.zone.lower()]
    if not records:
        return "No loitering records found."
    lines = []
    for r in records:
        lines.append(
            f"Zone: {r.zone} | Entity: {r.entity_type} | "
            f"Duration: {r.duration_seconds:.0f}s | "
            f"First seen: {r.first_seen} | Sightings: {r.sighting_count}"
        )
    return "\n".join(lines)


@tool
def get_vehicle_visits() -> str:
    """
    Retrieve vehicle sighting counts for the current patrol.
    Useful for detecting reconnaissance (same vehicle seen multiple times).
    """
    if _context_mgr is None:
        return "Context manager not initialised."
    visits = _context_mgr.get_vehicle_visits()
    if not visits:
        return "No vehicle sightings recorded."
    return "\n".join(f"{v}: {c} sighting(s)" for v, c in sorted(visits.items(), key=lambda x: -x[1]))


@tool
def generate_alert(level: str, message: str, zone: str, frame_id: int) -> str:
    """
    Formally log a security alert into the incident registry.

    Args:
        level:    Alert severity: LOW | MEDIUM | HIGH | CRITICAL
        message:  Detailed alert description
        zone:     Patrol zone where the incident occurred
        frame_id: Video frame ID where the incident was first detected
    """
    alert = {
        "id": len(_alerts_log) + 1,
        "level": level.upper(),
        "message": message,
        "zone": zone,
        "frame_id": frame_id,
        "logged_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    _alerts_log.append(alert)
    return f"✅ Alert #{alert['id']} logged: [{level.upper()}] {message}"


TOOLS = [search_frame_archive, get_loiter_history, get_vehicle_visits, generate_alert]


# ---------------------------------------------------------------------------
# Agent factory
# ---------------------------------------------------------------------------
def build_agent(
    vector_store,
    context_mgr,
    rule_engine,
    session_id: str = "SESSION-01",
    mission_start: str | None = None,
    llm_backend: str | None = None,
) -> AgentExecutor:
    """
    Construct and return a LangChain AgentExecutor for the security analyst.
    """
    _init_tools(vector_store, context_mgr, rule_engine)
    _alerts_log.clear()

    llm_backend = llm_backend or os.getenv("LANGCHAIN_LLM", "mock")
    llm = _get_llm(llm_backend)

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder("chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ]).partial(
        session_id=session_id,
        mission_start=mission_start or datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
    )

    agent = create_tool_calling_agent(llm, TOOLS, prompt)
    return AgentExecutor(agent=agent, tools=TOOLS, verbose=True, max_iterations=6)


def _get_llm(backend: str):
    if backend == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model="gpt-4o", temperature=0)
    elif backend == "mock":
        return _MockLLM()
    else:
        raise ValueError(f"Unknown LLM backend: {backend!r}")


# ---------------------------------------------------------------------------
# Mock LLM for zero-API-key testing
# ---------------------------------------------------------------------------
class _MockLLM:
    """
    Minimal stub that satisfies the LangChain LLM interface without calling
    any external API. Returns canned security-analyst responses.
    """

    def __init__(self):
        self._responses = [
            "Based on the frame archive, I detected a person loitering at the North Perimeter Gate for 90 seconds during curfew. CRITICAL alert has been logged.",
            "Vehicle reconnaissance pattern detected: white pickup truck spotted 3 times in this patrol. MEDIUM alert filed.",
            "No anomalies detected in the requested time window. Patrol appears nominal.",
            "Abandoned backpack detected near South Gate with no person in vicinity. HIGH priority alert dispatched.",
        ]
        self._idx = 0
        # Required for LangChain compatibility
        self.model_name = "mock-llm"

    def invoke(self, messages, **kwargs):
        resp = self._responses[self._idx % len(self._responses)]
        self._idx += 1
        from langchain_core.messages import AIMessage
        return AIMessage(content=resp)

    def bind_tools(self, tools, **kwargs):
        return self  # No-op for mock

    def __call__(self, messages, **kwargs):
        return self.invoke(messages, **kwargs)
