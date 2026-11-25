"""
telemetry_sync.py
-----------------
Generates or loads drone telemetry and synchronizes it with video frames.

Telemetry schema (per frame):
{
  "frame_id": int,
  "timestamp": str (ISO 8601),
  "drone_id": str,
  "latitude": float,
  "longitude": float,
  "altitude_m": float,
  "heading_deg": float,
  "battery_pct": int,
  "speed_ms": float,
  "patrol_zone": str,
}
"""

import json
import math
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List


# ---------------------------------------------------------------------------
# Patrol zone definitions (zone name → bounding lat/lon box)
# ---------------------------------------------------------------------------
PATROL_ZONES = [
    "North Perimeter Gate",
    "East Fence Line",
    "South Perimeter Gate",
    "West Fence Line",
    "Main Entrance",
    "Restricted Fence Line",
    "Parking Lot",
    "Loading Bay",
]


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def generate_telemetry(
    num_frames: int = 300,
    drone_id: str = "DRONE-01",
    start_time: datetime | None = None,
    sample_interval_sec: float = 1.0,
    output_path: str | None = None,
) -> List[Dict]:
    """
    Simulate a patrol mission telemetry stream.

    The drone flies a rectangular path around a property, visiting each
    patrol zone sequentially. Returns a list of telemetry dicts and
    optionally saves to JSON.
    """
    if start_time is None:
        start_time = datetime.utcnow()

    # Base GPS coordinates (simulated property center)
    base_lat, base_lon = 37.7749, -122.4194

    # Waypoints (lat/lon offsets around the perimeter)
    waypoints = [
        (base_lat + 0.001, base_lon - 0.001),   # North Gate
        (base_lat + 0.001, base_lon + 0.001),   # East Fence
        (base_lat - 0.001, base_lon + 0.001),   # South Gate
        (base_lat - 0.001, base_lon - 0.001),   # West Fence
    ]

    records: List[Dict] = []
    zone_cycle = PATROL_ZONES.copy()
    battery = 100
    frames_per_leg = num_frames // len(waypoints)

    for frame_id in range(num_frames):
        leg = frame_id // frames_per_leg
        leg = min(leg, len(waypoints) - 1)
        t_in_leg = (frame_id % frames_per_leg) / max(frames_per_leg - 1, 1)

        wp_a = waypoints[leg]
        wp_b = waypoints[(leg + 1) % len(waypoints)]
        lat = _lerp(wp_a[0], wp_b[0], t_in_leg) + random.gauss(0, 0.00005)
        lon = _lerp(wp_a[1], wp_b[1], t_in_leg) + random.gauss(0, 0.00005)

        # Heading based on direction of travel
        dlat = wp_b[0] - wp_a[0]
        dlon = wp_b[1] - wp_a[1]
        heading = math.degrees(math.atan2(dlon, dlat)) % 360

        altitude = 25.0 + random.gauss(0, 1.5)
        speed = 5.0 + random.gauss(0, 0.5)
        battery = max(5, battery - random.uniform(0.03, 0.08))

        zone = zone_cycle[leg % len(zone_cycle)]
        ts = start_time + timedelta(seconds=frame_id * sample_interval_sec)

        records.append({
            "frame_id": frame_id,
            "timestamp": ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "drone_id": drone_id,
            "latitude": round(lat, 6),
            "longitude": round(lon, 6),
            "altitude_m": round(altitude, 1),
            "heading_deg": round(heading, 1),
            "battery_pct": int(battery),
            "speed_ms": round(speed, 2),
            "patrol_zone": zone,
        })

    if output_path:
        Path(output_path).write_text(json.dumps(records, indent=2))

    return records


def load_telemetry(json_path: str) -> List[Dict]:
    """Load telemetry from a JSON file."""
    return json.loads(Path(json_path).read_text())


def get_telemetry_for_frame(records: List[Dict], frame_id: int) -> Dict:
    """Fast O(1) lookup by frame_id (assumes frame_id == list index)."""
    if 0 <= frame_id < len(records):
        return records[frame_id]
    # Fall back to linear search
    for r in records:
        if r["frame_id"] == frame_id:
            return r
    raise KeyError(f"frame_id {frame_id} not found in telemetry")
