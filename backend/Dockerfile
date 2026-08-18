FROM python:3.11-slim

# libGL/libglib: OpenCV needs them even in headless builds.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml ./
COPY roadwatch ./roadwatch
COPY alembic.ini ./

# The API container deliberately does NOT install the detection extra: torch
# is ~2.5GB and only the edge runtime performs inference (FR-DET-01 is what
# makes this separation possible).
RUN pip install --no-cache-dir -e .

ENV PYTHONUNBUFFERED=1
EXPOSE 8000

CMD ["uvicorn", "roadwatch.server_app:app", "--host", "0.0.0.0", "--port", "8000"]
