"""Distance estimation interface (FR-PRX).

Every estimator here answers one question: given a box in a frame, how far
ahead is that defect on the road?

Two rules govern everything in this package:

1. **These are estimates, never measurements.** Each result carries a
   ``confidence`` and the ``assumptions`` it rests on, and the UI is required
   to show them (ethical guardrail 2). A number presented without its
   uncertainty invites someone to trust it more than it deserves.
2. **Pure functions, no I/O.** Testable against hand-computed geometry with no
   camera (architecture rule; NFR-TST-01).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from roadwatch.core.contracts import Detection, DistanceEstimate, GpsFix


@runtime_checkable
class DistanceEstimator(Protocol):
    """Estimates ground distance to a detected hazard."""

    method: str
    """Identifier recorded on every estimate, e.g. "planar"."""

    def estimate(
        self,
        det: Detection,
        frame_shape: tuple[int, int],
        gps: GpsFix | None,
    ) -> DistanceEstimate:
        """Estimate distance to one detection.

        Args:
            det: The detection. Only ``bbox`` is used geometrically.
            frame_shape: ``(height, width)`` in pixels. Distance depends on
                where the box sits relative to the horizon, so this must be
                the shape of the frame the detection came from, not the
                configured resolution.
            gps: Latest fix, for the speed needed to derive
                time-to-pothole (FR-PRX-05). None when GPS is unavailable —
                estimators must still return a distance.

        Returns:
            An estimate that always carries confidence and assumptions.
        """
        ...
