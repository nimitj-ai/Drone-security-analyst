"""
pipeline.py
-----------
Main orchestration script: runs the full drone patrol analysis pipeline
end-to-end for a given video/frame directory + telemetry file.

Usage:
    python -m src.pipeline \
        --frames  data/sample_frames \
        --telem   data/telemetry_stream.json \
        --out     data/session_output.json

Environment variables:
    VLM_BACKEND      blip2 | florence2 | mock  (default: mock)
    EMBED_BACKEND    clip | mock               (default: mock)
    VECTOR_BACKEND   lancedb | mock            (default: mock)
    LANGCHAIN_LLM    openai | mock             (default: mock)
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

from src.ingestion.video_loader import extract_frames
from src.ingestion.telemetry_sync import load_telemetry, get_telemetry_for_frame, generate_telemetry
from src.perception.vlm_captioner import get_captioner
from src.perception.embedder import FrameEmbedder
from src.indexing.vector_store import get_vector_store
from src.agent.context_manager import PatrolContextManager
from src.agent.security_rules import SecurityRuleEngine
from src.agent.summarizer import PatrolSummarizer


def run_pipeline(
    frames_path: str,
    telemetry_path: str | None = None,
    output_path: str = "data/session_output.json",
    session_id: str = "SESSION-01",
    sample_fps: float = 1.0,
) -> dict:
    print(f"\n{'='*60}")
    print(f"  Drone Security Analyst — Patrol Session {session_id}")
    print(f"{'='*60}\n")

    # --- Initialise components ---
    captioner = get_captioner()
    embedder = FrameEmbedder()
    store = get_vector_store()
    context = PatrolContextManager()
    rules = SecurityRuleEngine()
    summarizer = PatrolSummarizer()

    # --- Load or generate telemetry ---
    if telemetry_path and Path(telemetry_path).exists():
        telemetry = load_telemetry(telemetry_path)
    else:
        print("[Pipeline] No telemetry file provided — generating synthetic telemetry.")
        telemetry = generate_telemetry(num_frames=300, output_path=telemetry_path)

    # --- Process frames ---
    all_alerts = []
    zone_history = []
    entity_counts: dict = {}
    start_ts = end_ts = None
    total_frames = 0

    print(f"[Pipeline] Ingesting frames from: {frames_path}")
    for frame_id, timestamp, pil_img in extract_frames(frames_path, sample_fps=sample_fps):
        total_frames += 1
        if start_ts is None:
            start_ts = timestamp
        end_ts = timestamp

        # Get telemetry for this frame
        try:
            telem = get_telemetry_for_frame(telemetry, frame_id)
        except KeyError:
            telem = {
                "frame_id": frame_id, "timestamp": timestamp,
                "drone_id": "DRONE-01", "patrol_zone": "Unknown",
                "altitude_m": 25.0, "battery_pct": 80,
            }

        zone = telem.get("patrol_zone", "Unknown")
        zone_history.append(zone)
        altitude = telem.get("altitude_m", 25.0)

        # --- VLM captioning ---
        fc = captioner.caption(frame_id, pil_img)

        # --- Embed & index ---
        vec = embedder.embed_image(pil_img)
        store.upsert([{
            "frame_id": frame_id,
            "timestamp": timestamp,
            "drone_id": telem.get("drone_id", "DRONE-01"),
            "patrol_zone": zone,
            "altitude_m": float(altitude),
            "battery_pct": int(telem.get("battery_pct", 100)),
            "caption": fc.caption,
            "objects": ", ".join(fc.objects),
            "image_path": f"frame_{frame_id:04d}.jpg",
            "vector": vec.tolist(),
        }])

        # --- Context tracking ---
        context_alerts = context.ingest(
            frame_id=frame_id,
            timestamp=timestamp,
            zone=zone,
            objects=fc.objects,
            caption=fc.caption,
        )

        # --- Rule engine ---
        frame_alerts = rules.evaluate(
            frame_id=frame_id,
            timestamp=timestamp,
            zone=zone,
            caption=fc.caption,
            objects=fc.objects,
            altitude_m=altitude,
            context_alerts=context_alerts,
        )

        for a in frame_alerts:
            all_alerts.append({
                "level": a.level,
                "rule": a.rule,
                "message": a.message,
                "frame_id": a.frame_id,
                "timestamp": a.timestamp,
                "zone": a.zone,
            })
            level_color = {
                "CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🟡", "LOW": "⚪"
            }.get(a.level, "")
            print(f"  {level_color} [{a.level}] Frame {frame_id}: {a.message[:100]}")

        # Accumulate entity counts
        for obj in fc.objects:
            entity_counts[obj] = entity_counts.get(obj, 0) + 1

        if frame_id % 10 == 0:
            print(f"  Processed frame {frame_id:04d} | Zone: {zone} | Objects: {fc.objects}")

    # --- Generate patrol summary ---
    print(f"\n[Pipeline] Generating patrol summary …")
    summary = summarizer.generate_summary(
        session_id=session_id,
        total_frames=total_frames,
        zones=zone_history,
        alerts=all_alerts,
        entity_counts=entity_counts,
        start_time=start_ts or "",
        end_time=end_ts or "",
    )

    result = {
        "session_id": session_id,
        "total_frames": total_frames,
        "alerts": all_alerts,
        "entity_counts": entity_counts,
        "summary": summary,
        "start_time": start_ts,
        "end_time": end_ts,
        "indexed_frames": store.count(),
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(result, indent=2))

    print(f"\n{'='*60}")
    print(f"  Session complete: {total_frames} frames | {len(all_alerts)} alerts")
    print(f"  Output saved to: {output_path}")
    print(f"\nPATROL SUMMARY:\n{summary}")
    print(f"{'='*60}\n")

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Drone Security Analyst Pipeline")
    parser.add_argument("--frames", default="data/sample_frames", help="Path to frames dir or video file")
    parser.add_argument("--telem", default="data/telemetry_stream.json", help="Path to telemetry JSON")
    parser.add_argument("--out", default="data/session_output.json", help="Output JSON path")
    parser.add_argument("--session", default="SESSION-01", help="Session identifier")
    parser.add_argument("--fps", type=float, default=1.0, help="Frames per second to sample")
    args = parser.parse_args()

    run_pipeline(
        frames_path=args.frames,
        telemetry_path=args.telem,
        output_path=args.out,
        session_id=args.session,
        sample_fps=args.fps,
    )
