# Deployment Guide

## What you'll learn

- Local development with `grampus dev`
- Single-machine production with Docker Compose
- Kubernetes deployment with Dapr pod injection

---

## Local development

The fastest way to get started is `grampus dev`, which starts everything and watches for file changes:

```bash
cd my-agent
grampus dev
```

This command:

1. Validates `grampus.yaml`
2. Starts the Dapr sidecar in the background
3. Starts your agent with auto-reload on file changes
4. Prints live cost and traces to the terminal

```
 Grampus dev mode
 Config: grampus.yaml
 Dapr sidecar: http://localhost:3500
 Agent port: 8000
 Watching: agent.py, grampus.yaml

[12:34:01] Agent started. Ready for input.
[12:34:05] Run started | session=dev-001 | model=claude-sonnet-4-6
[12:34:06]   tool_call: web_search(query="AI frameworks")  [312ms]
[12:34:07]   llm_call: 489 tokens  $0.0003
[12:34:07] Run complete | steps=2 | cost=$0.0005 | duration=2.1s
```

---

## Docker Compose (single-machine production)

A complete `docker-compose.yml` for running a Grampus agent in production on a single machine:

```yaml
# docker-compose.yml
version: "3.9"

services:
  # ── PostgreSQL + pgvector ──────────────────────────────────────────────────
  postgres:
    image: pgvector/pgvector:pg16
    environment:
      POSTGRES_USER: grampus
      POSTGRES_PASSWORD: nexus_secret
      POSTGRES_DB: grampus
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./scripts/init-db.sql:/docker-entrypoint-initdb.d/init.sql
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U grampus"]
      interval: 10s
      timeout: 5s
      retries: 5

  # ── Redis ──────────────────────────────────────────────────────────────────
  redis:
    image: redis:7-alpine
    command: redis-server --requirepass redis_secret
    healthcheck:
      test: ["CMD", "redis-cli", "-a", "redis_secret", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5

  # ── Dapr placement ────────────────────────────────────────────────────────
  dapr-placement:
    image: daprio/dapr:1.14
    command: ["./placement", "-port", "50006"]
    ports:
      - "50006:50006"

  # ── Jaeger (OTEL tracing) ─────────────────────────────────────────────────
  jaeger:
    image: jaegertracing/all-in-one:1.62
    ports:
      - "16686:16686"
      - "4317:4317"
      - "4318:4318"
    environment:
      COLLECTOR_OTLP_ENABLED: "true"

  # ── Your Grampus agent ──────────────────────────────────────────────────────
  agent:
    build: .
    command: ["grampus", "run", "agent.py"]
    environment:
      GRAMPUS_MODEL__ANTHROPIC_API_KEY: "${ANTHROPIC_API_KEY}"
      GRAMPUS_DAPR__HOST: localhost
      GRAMPUS_DAPR__PORT: "3500"
      GRAMPUS_OBSERVABILITY__OTEL_ENDPOINT: "http://jaeger:4317"
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
    healthcheck:
      test: ["CMD", "grampus", "--version"]
      interval: 30s
      timeout: 10s
      retries: 3

  # ── Dapr sidecar for agent ────────────────────────────────────────────────
  agent-dapr:
    image: daprio/daprd:1.14
    command:
      - "./daprd"
      - "--app-id=grampus-agent"
      - "--app-port=8000"
      - "--dapr-http-port=3500"
      - "--dapr-grpc-port=50001"
      - "--placement-host-address=dapr-placement:50006"
      - "--resources-path=/dapr/components"
      - "--config=/dapr/config.yaml"
    volumes:
      - ./dapr:/dapr
    network_mode: "service:agent"
    depends_on:
      - dapr-placement

volumes:
  postgres_data:
```

### Environment variable configuration

Create `.env` (not committed to git):

```bash
ANTHROPIC_API_KEY=sk-ant-...
```

### Start the stack

```bash
docker compose up -d

# Check all services are healthy
docker compose ps

# View agent logs
docker compose logs -f agent
```

### Health check endpoints

| Service | URL | Expected |
|---------|-----|----------|
| Dapr sidecar | `http://localhost:3500/v1.0/healthz` | `200 OK` |
| Jaeger UI | `http://localhost:16686` | Web UI |
| Agent (if HTTP) | `http://localhost:8000/health` | `{"status":"ok"}` |

---

## Kubernetes

### Prerequisites

- Kubernetes cluster (minikube, Kind, EKS, GKE, AKS)
- `dapr init --kubernetes` run
- `kubectl` configured

### Dapr pod injection

Add Dapr annotations to your Pod spec:

```yaml
# agent-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: grampus-agent
  namespace: default
spec:
  replicas: 2
  selector:
    matchLabels:
      app: grampus-agent
  template:
    metadata:
      labels:
        app: grampus-agent
      annotations:
        dapr.io/enabled: "true"
        dapr.io/app-id: "grampus-agent"
        dapr.io/app-port: "8000"
        dapr.io/config: "grampus-dapr-config"
        dapr.io/resources-path: "/dapr/components"
    spec:
      containers:
        - name: agent
          image: your-registry/grampus-agent:latest
          command: ["grampus", "run", "agent.py"]
          ports:
            - containerPort: 8000
          env:
            - name: GRAMPUS_MODEL__ANTHROPIC_API_KEY
              valueFrom:
                secretKeyRef:
                  name: grampus-secrets
                  key: anthropic-api-key
            - name: GRAMPUS_OBSERVABILITY__OTEL_ENDPOINT
              value: "http://otel-collector.monitoring:4317"
          readinessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 10
            periodSeconds: 5
          livenessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 30
            periodSeconds: 10
          resources:
            requests:
              memory: "256Mi"
              cpu: "100m"
            limits:
              memory: "1Gi"
              cpu: "500m"
```

### ConfigMap for grampus.yaml

```yaml
# grampus-config.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: grampus-config
  namespace: default
data:
  grampus.yaml: |
    model:
      default_model: claude-sonnet-4-6
      temperature: 0.0
      max_tokens: 4096
    memory:
      working_memory_token_limit: 100000
      summarization_strategy: hybrid
    safety:
      injection_detection_level: balanced
      pii_detection_enabled: true
    dapr:
      host: localhost
      port: 3500
    observability:
      otel_enabled: true
      log_level: INFO
```

### Secret for API keys

```bash
kubectl create secret generic grampus-secrets \
  --from-literal=anthropic-api-key="sk-ant-..." \
  --namespace default
```

### Horizontal pod autoscaling

```yaml
# grampus-hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: grampus-agent-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: grampus-agent
  minReplicas: 1
  maxReplicas: 10
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70
```

### Deploy

```bash
kubectl apply -f grampus-config.yaml
kubectl apply -f agent-deployment.yaml
kubectl apply -f grampus-hpa.yaml

# Check status
kubectl get pods -l app=grampus-agent
kubectl logs -l app=grampus-agent -c agent --tail=50
```

### Dapr component configuration for Kubernetes

```yaml
# dapr/components/statestore-postgres.yaml
apiVersion: dapr.io/v1alpha1
kind: Component
metadata:
  name: statestore
  namespace: default
spec:
  type: state.postgresql
  version: v1
  metadata:
    - name: connectionString
      secretKeyRef:
        name: postgres-secret
        key: connectionString
    - name: tableName
      value: dapr_state
```

---

## Dockerfile

```dockerfile
FROM python:3.12-slim

WORKDIR /app

# Install uv
RUN pip install uv

# Install dependencies
COPY pyproject.toml uv.lock ./
RUN uv sync --no-dev

# Copy source
COPY src/ src/
COPY agent.py .
COPY grampus.yaml .

# Install grampus-ai
RUN uv pip install -e .

EXPOSE 8000

CMD ["grampus", "run", "agent.py"]
```

---

## Next steps

- **[Configuration reference →](../reference/config.md)** — All environment variables
- **[Observability guide →](observability.md)** — Configure OTEL for Kubernetes
- **[CLI reference →](../reference/cli.md)** — `grampus dev` and all deployment commands
