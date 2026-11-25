"""
synthetic_frames.py
--------------------
Generates a set of synthetic patrol frames as PIL images for testing
when no real video is available. Each frame contains:
  - A sky gradient background (simulates aerial view)
  - Coloured shapes representing objects (vehicles, people, objects)
  - Frame number, zone name, and timestamp watermark

Usage:
    python -m src.ingestion.synthetic_frames --out data/sample_frames --n 60
"""

import argparse
import os
import random
from datetime import datetime, timedelta
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from typing import List, Tuple


# Object templates: (label, colour, shape)
OBJECTS = [
    ("Person", (220, 50, 50), "ellipse"),
    ("Person (dark clothing)", (40, 40, 40), "ellipse"),
    ("Blue Sedan", (30, 100, 200), "rectangle"),
    ("White Pickup Truck", (240, 240, 240), "rectangle"),
    ("Security Guard", (0, 128, 0), "ellipse"),
    ("Backpack", (160, 82, 45), "rectangle"),
    ("Forklift", (255, 165, 0), "rectangle"),
]

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


def _sky_background(draw: ImageDraw.Draw, w: int, h: int, hour: int) -> None:
    """Draw a gradient sky to simulate night vs day."""
    if 6 <= hour < 20:
        top_col, bot_col = (135, 206, 235), (200, 230, 200)   # daytime
    else:
        top_col, bot_col = (5, 5, 30), (20, 20, 60)            # nighttime
    for y in range(h):
        t = y / h
        r = int(top_col[0] * (1 - t) + bot_col[0] * t)
        g = int(top_col[1] * (1 - t) + bot_col[1] * t)
        b = int(top_col[2] * (1 - t) + bot_col[2] * t)
        draw.line([(0, y), (w, y)], fill=(r, g, b))


def generate_frame(
    frame_id: int,
    zone: str,
    timestamp: str,
    hour: int,
    width: int = 640,
    height: int = 360,
) -> Tuple[Image.Image, List[str]]:
    """
    Returns (PIL Image, list_of_object_labels_in_scene).
    """
    img = Image.new("RGB", (width, height))
    draw = ImageDraw.Draw(img)
    _sky_background(draw, width, height, hour)

    # Draw ground plane (bottom quarter)
    draw.rectangle([0, height * 3 // 4, width, height], fill=(100, 120, 80))

    # Draw fence line
    fence_y = height * 3 // 4 - 10
    for x in range(0, width, 30):
        draw.rectangle([x, fence_y - 20, x + 5, fence_y], fill=(160, 140, 100))
    draw.line([(0, fence_y - 20), (width, fence_y - 20)], fill=(180, 160, 110), width=2)

    # Decide how many objects to place (0-3)
    num_objects = random.choices([0, 1, 2, 3], weights=[20, 40, 30, 10])[0]
    placed_labels: List[str] = []

    for _ in range(num_objects):
        obj_label, obj_colour, obj_shape = random.choice(OBJECTS)
        # Place near fence line or on ground
        x = random.randint(20, width - 60)
        y = random.randint(fence_y - 40, height - 40)
        w, h_ = random.randint(20, 50), random.randint(20, 40)
        if obj_shape == "ellipse":
            draw.ellipse([x, y, x + w, y + h_], fill=obj_colour, outline=(0, 0, 0))
        else:
            draw.rectangle([x, y, x + w, y + h_], fill=obj_colour, outline=(0, 0, 0))
        # Label
        draw.text((x, y - 12), obj_label, fill=(255, 255, 0))
        placed_labels.append(obj_label)

    # HUD overlay
    hud_lines = [
        f"Frame: {frame_id:04d}",
        f"Zone:  {zone}",
        f"Time:  {timestamp}",
    ]
    for i, line in enumerate(hud_lines):
        draw.rectangle([5, 5 + i * 18, 300, 22 + i * 18], fill=(0, 0, 0, 128))
        draw.text((8, 6 + i * 18), line, fill=(0, 255, 0))

    return img, placed_labels


def generate_dataset(
    output_dir: str,
    num_frames: int = 60,
    start_hour: int = 22,          # Start at 10 PM (curfew scenarios)
    sample_interval_sec: float = 1.0,
) -> List[dict]:
    """Generate synthetic frames and return metadata list."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    zones = PATROL_ZONES * (num_frames // len(PATROL_ZONES) + 1)
    start_dt = datetime.utcnow().replace(hour=start_hour, minute=0, second=0, microsecond=0)
    metadata = []

    for i in range(num_frames):
        ts = start_dt + timedelta(seconds=i * sample_interval_sec)
        zone = zones[i]
        hour = ts.hour
        img, labels = generate_frame(i, zone, ts.strftime("%Y-%m-%dT%H:%M:%SZ"), hour)
        fname = f"frame_{i:04d}.jpg"
        img.save(os.path.join(output_dir, fname))
        metadata.append({
            "frame_id": i,
            "timestamp": ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "zone": zone,
            "file": fname,
            "objects_in_scene": labels,
        })

    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/sample_frames")
    parser.add_argument("--n", type=int, default=60)
    parser.add_argument("--hour", type=int, default=22)
    args = parser.parse_args()
    meta = generate_dataset(args.out, args.n, args.hour)
    print(f"Generated {len(meta)} frames → {args.out}")
