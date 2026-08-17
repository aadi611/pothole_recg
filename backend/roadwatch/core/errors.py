"""Error types.

Every error carries context (NFR-OBS-05). There is no bare ``except: pass``
anywhere in this codebase — when something is genuinely safe to ignore, catch
the specific exception and log it at debug with the reason it was ignored.
"""

from __future__ import annotations


class RoadWatchError(Exception):
    """Base for every error this system raises deliberately."""


class ConfigError(RoadWatchError):
    """Configuration is missing, malformed, or invalid.

    Always fatal at startup. A system running on defaults it did not announce
    is a system whose thresholds nobody can trust.
    """


class DetectorError(RoadWatchError):
    """The detection model could not be loaded or run."""


class WeightsIntegrityError(DetectorError):
    """Weights file hash does not match the configured value (FR-DET-06).

    Fatal: it means the model on disk is not the model that was validated, so
    every mAP number and threshold recorded for it is now unverifiable.
    """


class CameraError(RoadWatchError):
    """Frame source failed in a way retry cannot fix."""


class CalibrationError(RoadWatchError):
    """Camera calibration is absent or inconsistent."""


class GpsError(RoadWatchError):
    """GPS source failure. Never fatal to the detection loop (NFR-REL-01)."""


class StorageError(RoadWatchError):
    """Evidence could not be written."""


class StorageFullError(StorageError):
    """Free space is below the configured guard threshold (FR-SYN-05).

    Capture stops, the driver is alerted, and the event is logged loudly.
    Reports are never silently dropped.
    """


class RedactionError(RoadWatchError):
    """Face/plate redaction failed (NFR-PRV-02).

    The report stays local and flagged; it is not uploaded until redaction
    succeeds or an operator overrides with a logged reason.
    """


class AuthorityRoutingError(RoadWatchError):
    """No authority matched, or the match was ambiguous.

    Callers must resolve this to PENDING_MANUAL with a stated reason. Never
    guess an authority (FR-AUTH-03).
    """


class SubmissionError(RoadWatchError):
    """An authority submission attempt failed.

    ``retryable`` distinguishes a transient 5xx or timeout from a terminal 4xx
    or validation failure (FR-AUTH-07).
    """

    def __init__(self, message: str, *, retryable: bool) -> None:
        super().__init__(message)
        self.retryable = retryable


class SyncError(RoadWatchError):
    """Report could not be synced to the server. Stays queued locally."""
