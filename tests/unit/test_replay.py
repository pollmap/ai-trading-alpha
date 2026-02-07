"""Tests for deterministic replay system."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.simulator.replay import (
    EventType,
    ReplayEvent,
    ReplayPlayer,
    ReplayRecorder,
)


class TestReplayRecorder:
    def test_start_stop(self, tmp_path: Path) -> None:
        recorder = ReplayRecorder("test-session")
        path = recorder.start(tmp_path)
        assert path.exists()
        assert recorder.is_recording

        summary = recorder.stop()
        assert not recorder.is_recording
        assert summary.session_id == "test-session"
        assert summary.total_events >= 2  # start + end events

    def test_record_events(self, tmp_path: Path) -> None:
        recorder = ReplayRecorder("test-session")
        recorder.start(tmp_path)
        recorder.record(EventType.SNAPSHOT, {"snapshot_id": "snap-1"})
        recorder.record(EventType.SIGNAL, {"signal_id": "sig-1", "action": "BUY"})
        recorder.record(EventType.TRADE, {"trade_id": "trade-1"})
        summary = recorder.stop()
        assert summary.total_snapshots == 1
        assert summary.total_signals == 1
        assert summary.total_trades == 1

    def test_event_count(self, tmp_path: Path) -> None:
        recorder = ReplayRecorder("test-session")
        recorder.start(tmp_path)
        for i in range(10):
            recorder.record(EventType.SNAPSHOT, {"id": i})
        recorder.stop()
        assert recorder.event_count >= 12  # 10 + start + end


class TestReplayPlayer:
    def test_load_recorded_file(self, tmp_path: Path) -> None:
        # Record
        recorder = ReplayRecorder("test-load")
        path = recorder.start(tmp_path)
        recorder.record(EventType.SIGNAL, {"action": "BUY"})
        recorder.record(EventType.TRADE, {"qty": 0.5})
        recorder.stop()

        # Play back
        player = ReplayPlayer(path)
        count = player.load()
        assert count > 0
        assert player.event_count == count

    def test_filter_by_type(self, tmp_path: Path) -> None:
        recorder = ReplayRecorder("test-filter")
        path = recorder.start(tmp_path)
        recorder.record(EventType.SIGNAL, {"a": 1})
        recorder.record(EventType.SIGNAL, {"a": 2})
        recorder.record(EventType.TRADE, {"b": 1})
        recorder.stop()

        player = ReplayPlayer(path)
        player.load()
        signals = player.get_signals()
        trades = player.get_trades()
        assert len(signals) == 2
        assert len(trades) == 1

    def test_summary(self, tmp_path: Path) -> None:
        recorder = ReplayRecorder("test-summary")
        path = recorder.start(tmp_path)
        recorder.record(EventType.SNAPSHOT, {})
        recorder.record(EventType.SIGNAL, {})
        recorder.stop()

        player = ReplayPlayer(path)
        player.load()
        summary = player.summary()
        assert "snapshot" in summary
        assert "signal" in summary

    def test_missing_file(self, tmp_path: Path) -> None:
        player = ReplayPlayer(tmp_path / "nonexistent.jsonl")
        count = player.load()
        assert count == 0


class TestReplayEvent:
    def test_json_roundtrip(self) -> None:
        from datetime import datetime, timezone
        event = ReplayEvent(
            event_type=EventType.SIGNAL,
            timestamp=datetime.now(timezone.utc),
            sequence_number=1,
            data={"action": "BUY", "symbol": "BTCUSDT"},
        )
        json_str = event.to_json()
        restored = ReplayEvent.from_json(json_str)
        assert restored.event_type == EventType.SIGNAL
        assert restored.sequence_number == 1
        assert restored.data["action"] == "BUY"
