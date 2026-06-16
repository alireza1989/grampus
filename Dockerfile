# Stage 1: builder — install deps and build wheel
FROM python:3.12-slim AS builder

WORKDIR /app

RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock ./
COPY src/ src/

RUN uv sync --frozen --no-dev && uv build --wheel

# Stage 2: runtime — minimal image with installed wheel
FROM python:3.12-slim AS runtime

WORKDIR /app

# Install the built wheel with both provider extras
COPY --from=builder /app/dist/*.whl /tmp/
RUN pip install --no-cache-dir "/tmp/$(ls /tmp/*.whl)[anthropic,openai]" && rm /tmp/*.whl

# Dapr sidecar communicates via localhost — no special networking required
ENV DAPR_HTTP_PORT=3500
ENV DAPR_GRPC_PORT=50001
ENV GRAMPUS_ENV=production

# Non-root user for security
RUN useradd --create-home --shell /bin/bash grampus
USER grampus

# Agent code is volume-mounted at runtime, not baked into the image
WORKDIR /home/grampus/agent

ENTRYPOINT ["grampus"]
CMD ["--help"]
