"""RoadWatch — real-time pothole detection and civic reporting.

Two runtimes share this package:

- ``edge_app`` runs on the vehicle: camera, detection, tracking, distance,
  alerts, evidence, GPS, local queue. Must work with zero network.
- ``server_app`` runs in the cloud or on a laptop: API, dedup, geocoding,
  authority routing, analytics.

They communicate only through the API in ``spec/03_DATA_CONTRACTS.md``. For a
single-machine demo both run on the same box; nothing in the code may assume
that.
"""

__version__ = "0.1.0"
