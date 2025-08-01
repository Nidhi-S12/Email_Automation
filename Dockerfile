# Use multi-stage build for smaller final image
FROM python:3.11-slim AS builder

# Set build environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Set work directory
WORKDIR /build

# Install system dependencies including curl for UV
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        git && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Install UV package manager
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Copy dependency files for better layer caching
COPY pyproject.toml uv.lock ./

# Install Python dependencies using UV
RUN uv sync --frozen --no-dev

# Copy source code and project files after installing dependencies
COPY ./src ./src
COPY ./templates ./templates
COPY ./static ./static
COPY ./run_dashboard.py ./

# Start second stage with clean image
FROM python:3.11-slim

# Copy UV virtual environment from builder
COPY --from=builder /build/.venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Set environment variables
ENV PYTHONPATH=/app/src

# Set work directory
WORKDIR /app

# Copy application source code and required files
COPY --from=builder /build/src ./src
COPY --from=builder /build/templates ./templates
COPY --from=builder /build/static ./static
COPY --from=builder /build/run_dashboard.py ./
COPY --from=builder /build/pyproject.toml ./

# Create directories for output and knowledge
RUN mkdir -p output knowledge

# Install curl for health check
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    apt-get clean && rm -rf /var/lib/apt/lists/*


# Expose dashboard port (8080)
EXPOSE 8080

# Start the application directly
CMD ["python", "run_dashboard.py"]
