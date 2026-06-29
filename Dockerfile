# Stage 1: builder — install deps and build wheel
FROM python:3.12-slim-bookworm AS builder

WORKDIR /app

RUN apt-get update && apt-get upgrade -y --no-install-recommends && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock README.md ./
COPY src/ src/

RUN uv sync --frozen --no-dev && uv build --wheel

# Stage 2: runtime — slim-bookworm; apt-get upgrade patches known CVEs
FROM python:3.12-slim-bookworm AS runtime

WORKDIR /app

RUN apt-get update && apt-get upgrade -y --no-install-recommends && rm -rf /var/lib/apt/lists/*

# Install the built wheel with both provider extras
COPY --from=builder /app/dist/*.whl /tmp/
RUN WHEEL=$(ls /tmp/*.whl) && pip install --no-cache-dir "${WHEEL}[anthropic,openai]" && rm "$WHEEL"

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
