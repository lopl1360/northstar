# syntax=docker/dockerfile:1

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    POETRY_HOME="/opt/poetry" \
    PATH="$POETRY_HOME/bin:$PATH"

# Install system dependencies required during build and Poetry itself.
RUN apt-get update \
    && apt-get install --no-install-recommends -y build-essential \
    && pip install --no-cache-dir poetry \
    && apt-get purge -y build-essential \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user to run the application.
RUN useradd --create-home --home-dir /home/appuser --shell /bin/bash appuser

WORKDIR /app

# Install Python dependencies defined by Poetry.
COPY pyproject.toml poetry.lock* ./
RUN poetry config virtualenvs.create false \
    && poetry install --no-root --no-interaction --no-ansi

# Copy application source code and adjust ownership for the non-root user.
COPY --chown=appuser:appuser . .

USER appuser

CMD ["poetry", "run", "nightly"]
