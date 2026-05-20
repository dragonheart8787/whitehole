"""Data provenance tracking — fail-closed by default for mock fallbacks."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


class DataLoadError(Exception):
    """Raised when real data cannot be loaded and mock fallback is not allowed."""

    def __init__(self, message: str, *, requested_source: str, reason: str) -> None:
        super().__init__(message)
        self.requested_source = requested_source
        self.reason = reason


@dataclass
class DataProvenance:
    """Audit trail for what data was actually used in inference."""

    requested_source: str
    actual_source: str
    fallback_used: bool = False
    fallback_reason: str | None = None
    inject_model: str | None = None
    event: str | None = None
    channel: str | None = None
    extra: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        if d.get("extra") is None:
            d.pop("extra", None)
        return d

    @property
    def is_mock(self) -> bool:
        return self.actual_source.startswith("MOCK")

    def summary_line(self) -> str:
        if self.fallback_used:
            return (
                f"{self.requested_source} -> {self.actual_source} "
                f"(fallback: {self.fallback_reason})"
            )
        return f"{self.actual_source}"
