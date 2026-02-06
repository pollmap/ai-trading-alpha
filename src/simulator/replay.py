"""Deterministic replay — record and replay complete trading sessions.

Captures every input and decision during a benchmark run so that the exact
same sequence can be reproduced later for debugging, auditing, or analysis.

Storage format: JSONL (one JSON object per event line).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

from src.core.logging import get_logger

log = get_logger(__name__)


class EventType(str, Enum):
    SESSION_START = "session_start"
    SESSION_END = "session_end"
    SNAPSHOT = "snapshot"
    SIGNAL = "signal"
    RISK_CHECK = "risk_check"
    TRADE = "trade"
    PORTFOLIO_UPDATE = "portfolio_update"
    REFLECTION = "reflection"
    REGIME_CHANGE = "regime_change"
    ERROR = "error"


@dataclass
class ReplayEvent:
    """Single event in a replay session."""
    event_type: EventType
    timestamp: datetime
    sequence_number: int
    data: dict[str, object]
    metadata: dict[str, object] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps({
            "event_type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "sequence_number": self.sequence_number,
            "data": self.data,
            "metadata": self.metadata,
        }, default=str)

    @classmethod
    def from_json(cls, json_str: str) -> ReplayEvent:
        obj = json.loads(json_str)
        return cls(
            event_type=EventType(obj["event_type"]),
            timestamp=datetime.fromisoformat(obj["timestamp"]),
            sequence_number=obj["sequence_number"],
            data=obj.get("data", {}),
            metadata=obj.get("metadata", {}),
        )


@dataclass
class SessionSummary:
    """Summary statistics for a replay session."""
    session_id: str
    start_time: datetime
    end_time: datetime | None = None
    total_events: int = 0
    total_snapshots: int = 0
    total_signals: int = 0
    total_trades: int = 0
    total_errors: int = 0


class ReplayRecorder:
    """Record trading session events to a JSONL file.

    Usage::

        recorder = ReplayRecorder(session_id="benchmark-2026-01-15")
        recorder.start(output_dir=Path("data/replays"))
        recorder.record(EventType.SNAPSHOT, {"snapshot_id": "..."})
        recorder.record(EventType.SIGNAL, {"signal_id": "...", "action": "BUY"})
        recorder.stop()
    """

    def __init__(self, session_id: str) -> None:
        self._session_id: str = session_id
        self._sequence: int = 0
        self._file_handle: object | None = None
        self._output_path: Path | None = None
        self._summary: SessionSummary = SessionSummary(
            session_id=session_id,
            start_time=datetime.now(timezone.utc),
        )

    def start(self, output_dir: Path) -> Path:
        """Start recording to a JSONL file.

        Args:
            output_dir: Directory to write the replay file.

        Returns:
            Path to the created replay file.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        self._output_path = output_dir / f"replay_{self._session_id}_{ts}.jsonl"
        self._file_handle = open(self._output_path, "w")  # noqa: SIM115

        # Record session start event
        self.record(EventType.SESSION_START, {
            "session_id": self._session_id,
            "start_time": datetime.now(timezone.utc).isoformat(),
        })

        log.info(
            "replay_recording_started",
            session_id=self._session_id,
            output_path=str(self._output_path),
        )
        return self._output_path

    def record(self, event_type: EventType, data: dict[str, object]) -> None:
        """Record a single event."""
        if self._file_handle is None:
            log.warning("replay_record_called_before_start")
            return

        self._sequence += 1
        event = ReplayEvent(
            event_type=event_type,
            timestamp=datetime.now(timezone.utc),
            sequence_number=self._sequence,
            data=data,
        )

        self._file_handle.write(event.to_json() + "\n")  # type: ignore[union-attr]
        self._file_handle.flush()  # type: ignore[union-attr]

        # Update summary counters
        self._summary.total_events += 1
        if event_type == EventType.SNAPSHOT:
            self._summary.total_snapshots += 1
        elif event_type == EventType.SIGNAL:
            self._summary.total_signals += 1
        elif event_type == EventType.TRADE:
            self._summary.total_trades += 1
        elif event_type == EventType.ERROR:
            self._summary.total_errors += 1

    def stop(self) -> SessionSummary:
        """Stop recording and close the file.

        Returns:
            SessionSummary with statistics.
        """
        self.record(EventType.SESSION_END, {
            "session_id": self._session_id,
            "end_time": datetime.now(timezone.utc).isoformat(),
        })

        if self._file_handle is not None:
            self._file_handle.close()  # type: ignore[union-attr]
            self._file_handle = None

        self._summary.end_time = datetime.now(timezone.utc)
        log.info(
            "replay_recording_stopped",
            session_id=self._session_id,
            total_events=self._summary.total_events,
            output_path=str(self._output_path),
        )
        return self._summary

    @property
    def is_recording(self) -> bool:
        return self._file_handle is not None

    @property
    def event_count(self) -> int:
        return self._sequence


class ReplayPlayer:
    """Load and replay a recorded session.

    Usage::

        player = ReplayPlayer(path=Path("data/replays/replay_xxx.jsonl"))
        for event in player.events():
            print(event.event_type, event.data)
    """

    def __init__(self, path: Path) -> None:
        self._path: Path = path
        self._events: list[ReplayEvent] = []
        self._loaded: bool = False

    def load(self) -> int:
        """Load all events from the replay file.

        Returns:
            Number of events loaded.
        """
        self._events = []
        if not self._path.exists():
            log.warning("replay_file_not_found", path=str(self._path))
            return 0

        with open(self._path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    event = ReplayEvent.from_json(line)
                    self._events.append(event)
                except (json.JSONDecodeError, KeyError, ValueError) as exc:
                    log.warning(
                        "replay_event_parse_error",
                        line=line[:100],
                        error=str(exc),
                    )

        self._loaded = True
        log.info(
            "replay_loaded",
            path=str(self._path),
            event_count=len(self._events),
        )
        return len(self._events)

    def events(
        self, event_type: EventType | None = None
    ) -> list[ReplayEvent]:
        """Get all events, optionally filtered by type."""
        if not self._loaded:
            self.load()

        if event_type is None:
            return list(self._events)
        return [e for e in self._events if e.event_type == event_type]

    def get_signals(self) -> list[ReplayEvent]:
        return self.events(EventType.SIGNAL)

    def get_trades(self) -> list[ReplayEvent]:
        return self.events(EventType.TRADE)

    def get_snapshots(self) -> list[ReplayEvent]:
        return self.events(EventType.SNAPSHOT)

    @property
    def event_count(self) -> int:
        return len(self._events)

    def summary(self) -> dict[str, int]:
        """Count events by type."""
        counts: dict[str, int] = {}
        for e in self._events:
            key = e.event_type.value
            counts[key] = counts.get(key, 0) + 1
        return counts
