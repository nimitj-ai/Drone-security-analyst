"""
tests/test_indexing.py
-----------------------
Unit tests for the vector store: upserting, searching, and filtering.
"""

import numpy as np
import pytest
from src.indexing.vector_store import MockVectorStore


SAMPLE_RECORDS = [
    {
        "frame_id": 0,
        "timestamp": "2026-11-24T22:00:00Z",
        "drone_id": "DRONE-01",
        "patrol_zone": "North Perimeter Gate",
        "altitude_m": 25.0,
        "battery_pct": 95,
        "caption": "Person in dark clothing near the fence at night.",
        "objects": "person",
        "image_path": "frame_0000.jpg",
        "vector": np.random.randn(512).astype(np.float32).tolist(),
    },
    {
        "frame_id": 1,
        "timestamp": "2026-11-24T22:01:00Z",
        "drone_id": "DRONE-01",
        "patrol_zone": "South Perimeter Gate",
        "altitude_m": 24.5,
        "battery_pct": 94,
        "caption": "White pickup truck parked outside the gate.",
        "objects": "vehicle",
        "image_path": "frame_0001.jpg",
        "vector": np.random.randn(512).astype(np.float32).tolist(),
    },
]


@pytest.fixture
def store():
    s = MockVectorStore()
    s.upsert(SAMPLE_RECORDS)
    return s


def test_upsert_and_count(store):
    assert store.count() == 2


def test_search_returns_results(store):
    qvec = np.random.randn(512).astype(np.float32)
    results = store.search(qvec, top_k=2)
    assert len(results) == 2


def test_zone_filter(store):
    qvec = np.random.randn(512).astype(np.float32)
    results = store.search(qvec, top_k=5, zone_filter="North Perimeter Gate")
    assert all("North Perimeter Gate" in r["patrol_zone"] for r in results)


def test_object_filter(store):
    qvec = np.random.randn(512).astype(np.float32)
    results = store.search(qvec, top_k=5, object_filter="vehicle")
    assert all("vehicle" in r["objects"].lower() for r in results)


def test_time_filter(store):
    night_records = store.filter_by_time(start_hour=22, end_hour=24)
    assert len(night_records) == 2  # Both records are at 22:xx


def test_get_all(store):
    all_r = store.get_all()
    assert len(all_r) == 2
    assert all_r[0]["frame_id"] == 0
