"""
summarizer.py
-------------
Generates a concise patrol mission summary and powers conversational Q&A
over the indexed patrol session data.

The summarizer composes a structured mission digest from:
  - Total frames processed
  - Zones visited
  - Security alerts triggered
  - Notable entity sightings (persons, vehicles)
  - Overall threat assessment
"""

import os
from datetime import datetime
from typing import List, Dict


SUMMARY_PROMPT_TEMPLATE = """
You are a Drone Security Analyst summarizing a completed patrol mission.

Mission Data:
- Patrol Session: {session_id}
- Duration: {duration_str}
- Frames Processed: {total_frames}
- Zones Covered: {zones}
- Alerts Triggered: {alert_count}
- Alert Details:
{alert_details}
- Entity Sightings:
{entity_summary}

Write a concise executive patrol summary (3-5 sentences) suitable for a security manager.
Focus on: overall security posture, any incidents, and recommended follow-up actions.
"""


class PatrolSummarizer:
    def __init__(self, llm_backend: str | None = None):
        self.backend = llm_backend or os.getenv("LANGCHAIN_LLM", "mock")

    def generate_summary(
        self,
        session_id: str,
        total_frames: int,
        zones: List[str],
        alerts: List[Dict],
        entity_counts: Dict[str, int],
        start_time: str,
        end_time: str,
    ) -> str:
        """Generate a structured mission summary."""
        # Compute duration
        try:
            fmt = "%Y-%m-%dT%H:%M:%SZ"
            t0 = datetime.strptime(start_time, fmt)
            t1 = datetime.strptime(end_time, fmt)
            dur_min = (t1 - t0).total_seconds() / 60
            duration_str = f"{dur_min:.1f} minutes"
        except Exception:
            duration_str = "unknown"

        alert_details = "\n".join(
            f"  [{a.get('level','?')}] Frame {a.get('frame_id','?')} — {a.get('message','')[:120]}"
            for a in alerts
        ) or "  None"

        entity_summary = "\n".join(
            f"  {etype}: {count} sighting(s)" for etype, count in entity_counts.items()
        ) or "  No entities detected"

        prompt_text = SUMMARY_PROMPT_TEMPLATE.format(
            session_id=session_id,
            duration_str=duration_str,
            total_frames=total_frames,
            zones=", ".join(sorted(set(zones))),
            alert_count=len(alerts),
            alert_details=alert_details,
            entity_summary=entity_summary,
        )

        if self.backend == "mock" or os.getenv("LANGCHAIN_LLM", "mock") == "mock":
            return self._mock_summary(session_id, total_frames, len(alerts), entity_counts)

        # Real LLM path (OpenAI)
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(model="gpt-4o", temperature=0)
        result = llm.invoke(prompt_text)
        return result.content

    def _mock_summary(
        self,
        session_id: str,
        total_frames: int,
        alert_count: int,
        entity_counts: Dict[str, int],
    ) -> str:
        threat_level = "HIGH" if alert_count > 2 else "MEDIUM" if alert_count > 0 else "LOW"
        persons = entity_counts.get("person", 0)
        vehicles = entity_counts.get("vehicle", 0)
        return (
            f"**Patrol Mission {session_id} Complete** — {total_frames} frames analysed across all patrol zones. "
            f"Threat level assessed as **{threat_level}**. "
            f"Detected {persons} person sighting(s) and {vehicles} vehicle sighting(s). "
            f"A total of {alert_count} security alert(s) were triggered during the patrol. "
            + ("No immediate follow-up required." if alert_count == 0
               else "Security personnel should review all HIGH/CRITICAL alerts and conduct a site walkthrough.")
        )


class PatrolQA:
    """
    Interactive Q&A interface over patrol session data.
    Uses vector search + LLM synthesis for natural language answers.
    """

    def __init__(self, vector_store, context_mgr, session_id: str = "SESSION-01"):
        self.vector_store = vector_store
        self.context_mgr = context_mgr
        self.session_id = session_id
        self._chat_history: List[Dict] = []

    def ask(self, question: str) -> str:
        """Answer a natural language question about the patrol session."""
        from src.perception.embedder import FrameEmbedder
        embedder = FrameEmbedder(backend="mock")
        qvec = embedder.embed_text(question)
        results = self.vector_store.search(qvec, top_k=3)

        context_snippets = "\n".join(
            f"Frame {r['frame_id']} [{r['timestamp']}] Zone:{r['patrol_zone']} — {r['caption']}"
            for r in results
        )

        # Mock response (replace with real LLM for production)
        self._chat_history.append({"role": "user", "content": question})

        if os.getenv("LANGCHAIN_LLM", "mock") == "mock":
            if not results:
                answer = "I could not find relevant patrol data to answer that question."
            else:
                top = results[0]
                answer = (
                    f"Based on patrol data, the most relevant observation is from "
                    f"Frame {top['frame_id']} at {top['timestamp']} (Zone: {top['patrol_zone']}): "
                    f"\"{top['caption']}\". "
                    f"{'Additional frames corroborate this finding.' if len(results) > 1 else ''}"
                )
        else:
            from langchain_openai import ChatOpenAI
            from langchain.prompts import PromptTemplate
            llm = ChatOpenAI(model="gpt-4o", temperature=0)
            prompt = (
                f"You are a drone security analyst. Answer the question using ONLY the patrol data below.\n\n"
                f"PATROL DATA:\n{context_snippets}\n\n"
                f"QUESTION: {question}\n\nANSWER:"
            )
            answer = llm.invoke(prompt).content

        self._chat_history.append({"role": "assistant", "content": answer})
        return answer

    def get_history(self) -> List[Dict]:
        return list(self._chat_history)
