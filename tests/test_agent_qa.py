"""
tests/test_agent_qa.py
-----------------------
Tests for the patrol summarizer and Q&A module.
"""

import pytest
from src.agent.summarizer import PatrolSummarizer, PatrolQA
from src.indexing.vector_store import MockVectorStore
from src.agent.context_manager import PatrolContextManager
import numpy as np


@pytest.fixture
def populated_store():
    store = MockVectorStore()
    store.upsert([
        {
            "frame_id": 0,
            "timestamp": "2026-11-24T22:00:00Z",
            "drone_id": "DRONE-01",
            "patrol_zone": "North Gate",
            "altitude_m": 25.0,
            "battery_pct": 95,
            "caption": "Person loitering near the main gate at night.",
            "objects": "person",
            "image_path": "frame_0000.jpg",
            "vector": np.random.randn(512).astype(np.float32).tolist(),
        },
        {
            "frame_id": 1,
            "timestamp": "2026-11-24T22:01:00Z",
            "drone_id": "DRONE-01",
            "patrol_zone": "South Gate",
            "altitude_m": 24.0,
            "battery_pct": 94,
            "caption": "White pickup truck parked near the perimeter.",
            "objects": "vehicle",
            "image_path": "frame_0001.jpg",
            "vector": np.random.randn(512).astype(np.float32).tolist(),
        },
    ])
    return store


class TestPatrolSummarizer:
    def test_summary_no_alerts(self):
        s = PatrolSummarizer()
        summary = s.generate_summary(
            session_id="TEST-01",
            total_frames=50,
            zones=["North Gate", "South Gate"],
            alerts=[],
            entity_counts={"person": 2},
            start_time="2026-11-24T22:00:00Z",
            end_time="2026-11-24T22:50:00Z",
        )
        assert isinstance(summary, str)
        assert len(summary) > 20
        assert "LOW" in summary  # Mock summarizer sets LOW for 0 alerts

    def test_summary_with_alerts(self):
        s = PatrolSummarizer()
        alerts = [{"level": "CRITICAL", "rule": "loitering", "message": "person loitering",
                   "frame_id": 0, "timestamp": "2026-11-24T23:00:00Z", "zone": "Gate"}]
        summary = s.generate_summary(
            session_id="TEST-02",
            total_frames=100,
            zones=["Gate"],
            alerts=alerts,
            entity_counts={"person": 5},
            start_time="2026-11-24T22:00:00Z",
            end_time="2026-11-25T00:00:00Z",
        )
        assert "HIGH" in summary or "CRITICAL" in summary or "alert" in summary.lower()

    def test_summary_duration_calculated(self):
        s = PatrolSummarizer()
        # Should not raise even with malformed time strings
        summary = s.generate_summary("X", 10, [], [], {}, "bad-time", "also-bad")
        assert isinstance(summary, str)


class TestPatrolQA:
    def test_qa_returns_string(self, populated_store):
        ctx = PatrolContextManager()
        qa = PatrolQA(populated_store, ctx, "TEST-01")
        answer = qa.ask("Were there any vehicles seen during the patrol?")
        assert isinstance(answer, str)
        assert len(answer) > 10

    def test_qa_history_accumulates(self, populated_store):
        ctx = PatrolContextManager()
        qa = PatrolQA(populated_store, ctx, "TEST-01")
        qa.ask("What happened at the North Gate?")
        qa.ask("Any suspicious persons?")
        history = qa.get_history()
        assert len(history) == 4  # 2 user + 2 assistant

    def test_qa_no_results_graceful(self, populated_store):
        """Empty query vector should return graceful no-data message."""
        ctx = PatrolContextManager()
        # Use a fresh empty store
        empty_store = MockVectorStore()
        qa = PatrolQA(empty_store, ctx, "TEST-01")
        answer = qa.ask("Any anomalies?")
        assert isinstance(answer, str)
