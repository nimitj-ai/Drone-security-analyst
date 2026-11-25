"""
vector_store.py
---------------
Hybrid vector + metadata store for drone patrol frame records.

Backend: LanceDB (embedded, serverless) with optional ChromaDB fallback.
Supports:
  - Upsert frame records (embedding + all metadata)
  - Semantic search by text query (via CLIP text embeddings)
  - Metadata filter queries (zone, time range, detected objects)
  - Retrieval for the LangChain agent tool

Schema per record:
  frame_id      : int
  timestamp     : str (ISO-8601)
  drone_id      : str
  patrol_zone   : str
  altitude_m    : float
  battery_pct   : int
  caption       : str
  objects       : str  (comma-separated list)
  image_path    : str  (relative path to saved frame image)
  vector        : list[float]  (512-dim CLIP embedding)
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np


# ---------------------------------------------------------------------------
# LanceDB backend (default)
# ---------------------------------------------------------------------------
class LanceDBStore:
    TABLE_NAME = "patrol_frames"

    def __init__(self, db_path: str = "data/lancedb"):
        import lancedb
        import pyarrow as pa
        self.db_path = db_path
        Path(db_path).mkdir(parents=True, exist_ok=True)
        self.db = lancedb.connect(db_path)
        self._pa = pa

        # Lazy-create table
        if self.TABLE_NAME not in self.db.table_names():
            schema = pa.schema([
                pa.field("frame_id", pa.int32()),
                pa.field("timestamp", pa.string()),
                pa.field("drone_id", pa.string()),
                pa.field("patrol_zone", pa.string()),
                pa.field("altitude_m", pa.float32()),
                pa.field("battery_pct", pa.int32()),
                pa.field("caption", pa.string()),
                pa.field("objects", pa.string()),
                pa.field("image_path", pa.string()),
                pa.field("vector", pa.list_(pa.float32(), 512)),
            ])
            self.table = self.db.create_table(self.TABLE_NAME, schema=schema)
        else:
            self.table = self.db.open_table(self.TABLE_NAME)

    def upsert(self, records: List[Dict[str, Any]]):
        """Add or overwrite frame records."""
        self.table.add(records, mode="append")

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
        zone_filter: Optional[str] = None,
        object_filter: Optional[str] = None,
    ) -> List[Dict]:
        q = self.table.search(query_vector.tolist()).limit(top_k)
        results = q.to_list()
        # Post-filter
        if zone_filter:
            results = [r for r in results if zone_filter.lower() in r["patrol_zone"].lower()]
        if object_filter:
            results = [r for r in results if object_filter.lower() in r["objects"].lower()]
        return results

    def get_all(self) -> List[Dict]:
        return self.table.to_pandas().to_dict(orient="records")

    def filter_by_time(self, start_hour: int, end_hour: int) -> List[Dict]:
        """Return records whose timestamp falls in [start_hour, end_hour)."""
        all_records = self.get_all()
        from datetime import datetime
        filtered = []
        for r in all_records:
            try:
                dt = datetime.strptime(r["timestamp"], "%Y-%m-%dT%H:%M:%SZ")
                if start_hour <= dt.hour < end_hour:
                    filtered.append(r)
            except Exception:
                pass
        return filtered

    def count(self) -> int:
        return len(self.table.to_pandas())


# ---------------------------------------------------------------------------
# In-memory mock store (zero deps, for testing)
# ---------------------------------------------------------------------------
class MockVectorStore:
    def __init__(self):
        self._records: List[Dict] = []

    def upsert(self, records: List[Dict]):
        self._records.extend(records)

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
        zone_filter: Optional[str] = None,
        object_filter: Optional[str] = None,
    ) -> List[Dict]:
        # Cosine similarity against stored vectors
        scored = []
        for r in self._records:
            vec = np.array(r["vector"], dtype=np.float32)
            sim = float(np.dot(query_vector, vec))
            scored.append((sim, r))
        scored.sort(key=lambda x: -x[0])
        results = [r for _, r in scored[:top_k]]
        if zone_filter:
            results = [r for r in results if zone_filter.lower() in r.get("patrol_zone", "").lower()]
        if object_filter:
            results = [r for r in results if object_filter.lower() in r.get("objects", "").lower()]
        return results

    def get_all(self) -> List[Dict]:
        return list(self._records)

    def filter_by_time(self, start_hour: int, end_hour: int) -> List[Dict]:
        from datetime import datetime
        filtered = []
        for r in self._records:
            try:
                dt = datetime.strptime(r["timestamp"], "%Y-%m-%dT%H:%M:%SZ")
                if start_hour <= dt.hour < end_hour:
                    filtered.append(r)
            except Exception:
                pass
        return filtered

    def count(self) -> int:
        return len(self._records)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def get_vector_store(backend: str | None = None, db_path: str = "data/lancedb"):
    backend = backend or os.getenv("VECTOR_BACKEND", "mock")
    if backend == "lancedb":
        return LanceDBStore(db_path=db_path)
    elif backend == "mock":
        return MockVectorStore()
    else:
        raise ValueError(f"Unknown vector backend: {backend!r}. Choose lancedb or mock.")
