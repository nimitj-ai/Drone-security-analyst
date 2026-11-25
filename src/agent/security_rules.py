"""
security_rules.py
-----------------
Deterministic rule-based alert evaluation.

Rules are configured via config/rules_config.yaml and can be extended
without touching this code. Each rule produces an Alert dataclass.
"""

import yaml
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "rules_config.yaml"


@dataclass
class Alert:
    rule: str                  # Rule name that triggered
    level: str                 # LOW | MEDIUM | HIGH | CRITICAL
    message: str               # Human-readable alert text
    frame_id: int
    timestamp: str
    zone: str
    evidence: Dict[str, Any]   # Supporting data for LLM context


class SecurityRuleEngine:
    """
    Evaluates a set of YAML-configured rules against per-frame observations.
    Can be called standalone or wired into the LangChain agent as a tool.
    """

    def __init__(self, config_path: str | Path | None = None):
        config_path = config_path or DEFAULT_CONFIG_PATH
        with open(config_path) as f:
            self.cfg = yaml.safe_load(f)

    # ------------------------------------------------------------------
    # Main evaluation entry point
    # ------------------------------------------------------------------
    def evaluate(
        self,
        frame_id: int,
        timestamp: str,
        zone: str,
        caption: str,
        objects: List[str],
        altitude_m: float,
        context_alerts: List[str],   # Alerts already raised by context manager
    ) -> List[Alert]:
        alerts: List[Alert] = []
        lower_caption = caption.lower()
        lower_objects = [o.lower() for o in objects]

        # Pass-through context manager alerts as formal Alert objects
        for msg in context_alerts:
            level = "CRITICAL" if "CRITICAL" in msg else "HIGH" if "⚠️" in msg else "MEDIUM"
            alerts.append(Alert(
                rule="context_manager",
                level=level,
                message=msg,
                frame_id=frame_id,
                timestamp=timestamp,
                zone=zone,
                evidence={"caption": caption[:200], "objects": objects},
            ))

        # Rule: Perimeter breach keywords
        if self.cfg.get("perimeter_breach", {}).get("enabled", False):
            thresh_alt = self.cfg["perimeter_breach"].get("altitude_threshold_m", 30)
            keywords = self.cfg["perimeter_breach"].get("cross_zone_keywords", [])
            if altitude_m < thresh_alt and any(kw in lower_caption for kw in keywords):
                alerts.append(Alert(
                    rule="perimeter_breach",
                    level="CRITICAL",
                    message=(
                        f"🚨 PERIMETER BREACH DETECTED at '{zone}' (Frame {frame_id}). "
                        f"Caption: '{caption[:120]}'. Altitude: {altitude_m}m."
                    ),
                    frame_id=frame_id,
                    timestamp=timestamp,
                    zone=zone,
                    evidence={"caption": caption, "altitude_m": altitude_m},
                ))

        # Rule: Abandoned object without person
        if self.cfg.get("abandoned_object", {}).get("enabled", False):
            obj_keywords = self.cfg["abandoned_object"].get("keywords", [])
            has_bag = any(kw in lower_caption for kw in obj_keywords)
            has_person = any(kw in lower_caption for kw in ["person", "individual", "man", "woman"])
            if has_bag and not has_person:
                alerts.append(Alert(
                    rule="abandoned_object",
                    level="HIGH",
                    message=(
                        f"⚠️  ABANDONED OBJECT at '{zone}' (Frame {frame_id}). "
                        f"Unattended item detected with no person nearby. "
                        f"Caption: '{caption[:120]}'."
                    ),
                    frame_id=frame_id,
                    timestamp=timestamp,
                    zone=zone,
                    evidence={"caption": caption, "objects": objects},
                ))

        return alerts

    def get_level_description(self, level: str) -> str:
        return self.cfg.get("alert_levels", {}).get(level, "")
