"""
context_manager.py
------------------
Maintains rolling state across patrol frames:
  - Tracks entities (persons, vehicles) across time for loitering detection
  - Records zone visit history for reconnaissance pattern analysis
  - Accumulates observed objects for abandoned-item detection

This module is intentionally stateful and in-process (suitable for a single
patrol session). For multi-drone / distributed setups, back this with Redis.
"""

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class EntitySighting:
    entity_type: str         # "person" | "vehicle" | "backpack" | ...
    zone: str
    timestamp: str
    frame_id: int
    caption_snippet: str


@dataclass
class LoiterRecord:
    entity_type: str
    zone: str
    first_seen: str
    last_seen: str
    sighting_count: int
    frame_ids: List[int] = field(default_factory=list)

    @property
    def duration_seconds(self) -> float:
        fmt = "%Y-%m-%dT%H:%M:%SZ"
        try:
            t0 = datetime.strptime(self.first_seen, fmt)
            t1 = datetime.strptime(self.last_seen, fmt)
            return (t1 - t0).total_seconds()
        except Exception:
            return 0.0


# ---------------------------------------------------------------------------
# Context Manager
# ---------------------------------------------------------------------------
class PatrolContextManager:
    """
    Tracks entity sightings, loitering, and vehicle revisits across the
    duration of a single patrol mission.
    """

    def __init__(
        self,
        loiter_threshold_sec: float = 60.0,
        recon_vehicle_threshold: int = 2,
        curfew_start: int = 22,
        curfew_end: int = 6,
    ):
        self.loiter_threshold_sec = loiter_threshold_sec
        self.recon_vehicle_threshold = recon_vehicle_threshold
        self.curfew_start = curfew_start
        self.curfew_end = curfew_end

        # zone → list of entity sightings
        self._zone_sightings: Dict[str, List[EntitySighting]] = defaultdict(list)
        # entity_type → zone → LoiterRecord
        self._loiter_map: Dict[str, Dict[str, LoiterRecord]] = defaultdict(dict)
        # vehicle sighting counter per patrol
        self._vehicle_visits: Dict[str, int] = defaultdict(int)
        # all sightings in order
        self.all_sightings: List[EntitySighting] = []
        # triggered alerts (to avoid duplicate firing)
        self._alerted_loiter: set = set()
        self._alerted_recon: set = set()

    # ------------------------------------------------------------------
    # Ingest a frame result
    # ------------------------------------------------------------------
    def ingest(
        self,
        frame_id: int,
        timestamp: str,
        zone: str,
        objects: List[str],
        caption: str,
    ) -> List[str]:
        """
        Process one frame's perception output.
        Returns a list of triggered rule-violation strings (empty = no alert).
        """
        alerts: List[str] = []

        for obj in objects:
            obj_lower = obj.lower()

            # Classify
            if any(kw in obj_lower for kw in ("person", "individual", "intruder")):
                entity_type = "person"
            elif any(kw in obj_lower for kw in ("car", "truck", "van", "sedan", "pickup", "vehicle")):
                entity_type = "vehicle"
                self._vehicle_visits[obj_lower] += 1
            elif any(kw in obj_lower for kw in ("backpack", "bag", "package")):
                entity_type = "abandoned_object"
            else:
                entity_type = obj_lower

            sighting = EntitySighting(
                entity_type=entity_type,
                zone=zone,
                timestamp=timestamp,
                frame_id=frame_id,
                caption_snippet=caption[:120],
            )
            self._zone_sightings[zone].append(sighting)
            self.all_sightings.append(sighting)

            # Update loitering tracker
            self._update_loiter(entity_type, zone, timestamp, frame_id)

            # Check loitering alert (person in curfew hours)
            if entity_type == "person":
                loiter_alert = self._check_loiter_alert(zone, timestamp)
                if loiter_alert:
                    alerts.append(loiter_alert)

        # Check vehicle reconnaissance alert
        for vehicle, count in self._vehicle_visits.items():
            if count >= self.recon_vehicle_threshold:
                key = vehicle
                if key not in self._alerted_recon:
                    self._alerted_recon.add(key)
                    alerts.append(
                        f"⚠️  RECONNAISSANCE ALERT: '{vehicle}' has been spotted "
                        f"{count} times during this patrol. Possible surveillance."
                    )

        return alerts

    # ------------------------------------------------------------------
    # Loitering helpers
    # ------------------------------------------------------------------
    def _update_loiter(self, entity_type: str, zone: str, timestamp: str, frame_id: int):
        if zone not in self._loiter_map[entity_type]:
            self._loiter_map[entity_type][zone] = LoiterRecord(
                entity_type=entity_type,
                zone=zone,
                first_seen=timestamp,
                last_seen=timestamp,
                sighting_count=1,
                frame_ids=[frame_id],
            )
        else:
            rec = self._loiter_map[entity_type][zone]
            rec.last_seen = timestamp
            rec.sighting_count += 1
            rec.frame_ids.append(frame_id)

    def _is_curfew(self, timestamp: str) -> bool:
        try:
            dt = datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%SZ")
            h = dt.hour
            if self.curfew_start < self.curfew_end:
                return self.curfew_start <= h < self.curfew_end
            else:  # wraps midnight
                return h >= self.curfew_start or h < self.curfew_end
        except Exception:
            return False

    def _check_loiter_alert(self, zone: str, timestamp: str) -> Optional[str]:
        if not self._is_curfew(timestamp):
            return None
        rec = self._loiter_map["person"].get(zone)
        if not rec:
            return None
        dur = rec.duration_seconds
        if dur >= self.loiter_threshold_sec:
            key = f"{zone}_{rec.first_seen}"
            if key not in self._alerted_loiter:
                self._alerted_loiter.add(key)
                return (
                    f"🚨 CRITICAL ALERT — CURFEW LOITERING: Person detected at '{zone}' "
                    f"for {dur:.0f}s during curfew hours (since {rec.first_seen}). "
                    f"Dispatch security immediately."
                )
        return None

    # ------------------------------------------------------------------
    # Query API (for LangChain tools)
    # ------------------------------------------------------------------
    def get_loiter_records(self) -> List[LoiterRecord]:
        """Return all loitering records across all zones."""
        records = []
        for entity_zones in self._loiter_map.values():
            records.extend(entity_zones.values())
        return records

    def get_zone_summary(self, zone: str) -> Dict:
        sightings = self._zone_sightings.get(zone, [])
        return {
            "zone": zone,
            "total_sightings": len(sightings),
            "entities": list({s.entity_type for s in sightings}),
            "first_sighting": sightings[0].timestamp if sightings else None,
            "last_sighting": sightings[-1].timestamp if sightings else None,
        }

    def get_vehicle_visits(self) -> Dict[str, int]:
        return dict(self._vehicle_visits)

    def reset(self):
        self._zone_sightings.clear()
        self._loiter_map.clear()
        self._vehicle_visits.clear()
        self.all_sightings.clear()
        self._alerted_loiter.clear()
        self._alerted_recon.clear()
