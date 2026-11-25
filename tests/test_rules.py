"""
tests/test_rules.py
-------------------
Tests for the security rule engine and context manager alert logic.
"""

import pytest
from src.agent.context_manager import PatrolContextManager
from src.agent.security_rules import SecurityRuleEngine, Alert


@pytest.fixture
def context():
    return PatrolContextManager(
        loiter_threshold_sec=5,  # low threshold for fast test
        recon_vehicle_threshold=2,
        curfew_start=22,
        curfew_end=6,
    )


@pytest.fixture
def rules():
    return SecurityRuleEngine()


# ---------------------------------------------------------------------------
# Context manager tests
# ---------------------------------------------------------------------------
class TestContextManager:
    def test_no_alert_outside_curfew(self, context):
        # 10 AM — no curfew alert expected
        for i in range(10):
            ts = f"2026-11-24T10:{i:02d}:00Z"
            alerts = context.ingest(i, ts, "North Perimeter Gate", ["person"], "person near fence")
        assert not any("CURFEW" in a for a in alerts)

    def test_loiter_alert_during_curfew(self, context):
        """Person loitering for >5s during curfew should trigger alert."""
        all_alerts = []
        # Simulate 10 seconds of sightings at 11 PM
        for i in range(10):
            ts = f"2026-11-24T23:00:{i:02d}Z"
            alerts = context.ingest(i, ts, "North Perimeter Gate", ["person"], "person seen near gate")
            all_alerts.extend(alerts)
        assert any("CRITICAL" in a for a in all_alerts), "Expected CRITICAL loitering alert"

    def test_no_alert_on_single_vehicle_sighting(self, context):
        alerts = context.ingest(0, "2026-11-24T14:00:00Z", "Parking Lot", ["blue sedan"], "car parked")
        assert not any("RECONNAISSANCE" in a for a in alerts)

    def test_recon_alert_on_repeat_vehicle(self, context):
        """Same vehicle seen twice should trigger reconnaissance alert."""
        all_alerts = []
        for i in range(2):
            ts = f"2026-11-24T14:{i:02d}:00Z"
            alerts = context.ingest(i, ts, "Parking Lot", ["white pickup truck"], "truck circling")
            all_alerts.extend(alerts)
        assert any("RECONNAISSANCE" in a for a in all_alerts)

    def test_loiter_records_populated(self, context):
        for i in range(5):
            ts = f"2026-11-24T23:00:{i:02d}Z"
            context.ingest(i, ts, "Main Entrance", ["person"], "person near entrance")
        records = context.get_loiter_records()
        assert len(records) > 0
        assert records[0].entity_type == "person"
        assert records[0].zone == "Main Entrance"
        assert records[0].sighting_count == 5

    def test_reset_clears_state(self, context):
        context.ingest(0, "2026-11-24T23:00:00Z", "Gate", ["person"], "seen")
        context.reset()
        assert len(context.all_sightings) == 0
        assert len(context.get_loiter_records()) == 0


# ---------------------------------------------------------------------------
# Rule engine tests
# ---------------------------------------------------------------------------
class TestSecurityRuleEngine:
    def test_perimeter_breach_detected(self, rules):
        alerts = rules.evaluate(
            frame_id=5,
            timestamp="2026-11-24T23:00:00Z",
            zone="Restricted Fence Line",
            caption="A person is climbing over the perimeter fence",
            objects=["person"],
            altitude_m=20.0,          # below threshold (30m)
            context_alerts=[],
        )
        assert any(a.rule == "perimeter_breach" for a in alerts)
        assert any(a.level == "CRITICAL" for a in alerts)

    def test_no_breach_above_altitude(self, rules):
        alerts = rules.evaluate(
            frame_id=6,
            timestamp="2026-11-24T23:00:00Z",
            zone="Restricted Fence Line",
            caption="A person is climbing over the perimeter fence",
            objects=["person"],
            altitude_m=50.0,          # above threshold — should not fire
            context_alerts=[],
        )
        assert not any(a.rule == "perimeter_breach" for a in alerts)

    def test_abandoned_object_alert(self, rules):
        alerts = rules.evaluate(
            frame_id=10,
            timestamp="2026-11-24T14:00:00Z",
            zone="South Perimeter Gate",
            caption="An abandoned backpack is sitting near the gate with no one around.",
            objects=["backpack"],
            altitude_m=25.0,
            context_alerts=[],
        )
        assert any(a.rule == "abandoned_object" for a in alerts)
        assert any(a.level == "HIGH" for a in alerts)

    def test_no_abandoned_alert_when_person_present(self, rules):
        alerts = rules.evaluate(
            frame_id=11,
            timestamp="2026-11-24T14:00:00Z",
            zone="South Perimeter Gate",
            caption="A person is carrying a backpack near the gate.",
            objects=["person", "backpack"],
            altitude_m=25.0,
            context_alerts=[],
        )
        assert not any(a.rule == "abandoned_object" for a in alerts)

    def test_context_alerts_pass_through(self, rules):
        ctx_msg = "🚨 CRITICAL ALERT — CURFEW LOITERING: test"
        alerts = rules.evaluate(
            frame_id=20, timestamp="2026-11-24T23:00:00Z",
            zone="Gate", caption="person near gate", objects=["person"],
            altitude_m=25.0, context_alerts=[ctx_msg],
        )
        assert any(a.rule == "context_manager" for a in alerts)
