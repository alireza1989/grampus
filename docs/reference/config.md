# Configuration Reference

Grampus is configured via a `grampus.yaml` file and/or environment variables. Environment variables always take precedence.

## Loading priority

1. **Environment variables** (highest priority) — `GRAMPUS_MODEL__ANTHROPIC_API_KEY=...`
2. **YAML file** — default: `grampus.yaml` or `nexus.yml` in the working directory; override with `GRAMPUS_CONFIG_FILE=/path/to/config.yaml`
3. **Defaults** (lowest priority)

---

## Complete annotated grampus.yaml

```yaml
# grampus.yaml

model:
  # LLM model identifier (used as default when AgentDefinition.model is not set)
  default_model: claude-sonnet-4-6

  # Sampling temperature: 0.0 = deterministic, 2.0 = very creative
  temperature: 0.0

  # Maximum tokens in LLM response
  max_tokens: 4096

  # Anthropic API key (or set GRAMPUS_MODEL__ANTHROPIC_API_KEY)
  # anthropic_api_key: sk-ant-...

  # OpenAI API key (or set GRAMPUS_MODEL__OPENAI_API_KEY)
  # openai_api_key: sk-...

memory:
  # Summarize working memory when it exceeds this token count
  working_memory_token_limit: 100000

  # Number of episodic/semantic results to retrieve per recall query
  episodic_top_k: 5

  # Temporal decay rate for episodic records (per day)
  # 0.01 = 1% reduction in score per day of age
  decay_rate: 0.01

  # Summarization strategy when working memory is full:
  # truncate  — drop oldest messages
  # summarize — LLM-generated summary of old messages
  # hybrid    — summarize old + keep recent messages verbatim
  summarization_strategy: hybrid

safety:
  # Injection detection sensitivity:
  # strict     — block on any suspicious pattern (low false negatives)
  # balanced   — block high-confidence patterns only (recommended)
  # permissive — log only, never block
  injection_detection_level: balanced

  # Enable PII detection and redaction in all I/O
  pii_detection_enabled: true

  # Maximum tool calls per agent turn (across all tools)
  action_rate_limit_per_minute: 60

dapr:
  # Dapr HTTP sidecar host
  host: localhost

  # Dapr HTTP sidecar port
  port: 3500

  # Dapr gRPC sidecar port
  grpc_port: 50001

  # Name of the Dapr state store component for primary storage
  state_store_name: statestore

  # Name of the Dapr pub/sub component
  pubsub_name: pubsub

  # Name of the Dapr state store component used as cache
  cache_store_name: cache

observability:
  # Enable OpenTelemetry tracing
  otel_enabled: true

  # OTEL Collector gRPC endpoint
  otel_endpoint: http://localhost:4317

  # Logging level: DEBUG | INFO | WARNING | ERROR
  log_level: INFO

  # Enable Prometheus metrics endpoint
  metrics_enabled: true
```

---

## ModelConfig

**Env prefix:** `GRAMPUS_MODEL__`

| Field | Type | Default | Env var override | Description |
|-------|------|---------|-----------------|-------------|
| `default_model` | `str` | `"claude-3-5-haiku-20241022"` | `GRAMPUS_MODEL__DEFAULT_MODEL` | Default model identifier |
| `temperature` | `float` | `0.0` | `GRAMPUS_MODEL__TEMPERATURE` | Sampling temperature (0.0–2.0) |
| `max_tokens` | `int` | `4096` | `GRAMPUS_MODEL__MAX_TOKENS` | Max response tokens |
| `anthropic_api_key` | `SecretStr \| None` | `None` | `GRAMPUS_MODEL__ANTHROPIC_API_KEY` | Anthropic API key (masked in logs) |
| `openai_api_key` | `SecretStr \| None` | `None` | `GRAMPUS_MODEL__OPENAI_API_KEY` | OpenAI API key (masked in logs) |

---

## MemoryConfig

**Env prefix:** `GRAMPUS_MEMORY__`

| Field | Type | Default | Env var override | Description |
|-------|------|---------|-----------------|-------------|
| `working_memory_token_limit` | `int` | `100000` | `GRAMPUS_MEMORY__WORKING_MEMORY_TOKEN_LIMIT` | Token limit before auto-summarization |
| `episodic_top_k` | `int` | `5` | `GRAMPUS_MEMORY__EPISODIC_TOP_K` | Results returned per recall query |
| `decay_rate` | `float` | `0.01` | `GRAMPUS_MEMORY__DECAY_RATE` | Per-day score decay for old memories |
| `summarization_strategy` | `str` | `"hybrid"` | `GRAMPUS_MEMORY__SUMMARIZATION_STRATEGY` | `truncate` \| `summarize` \| `hybrid` |

---

## SafetyConfig

**Env prefix:** `GRAMPUS_SAFETY__`

| Field | Type | Default | Env var override | Description |
|-------|------|---------|-----------------|-------------|
| `injection_detection_level` | `str` | `"balanced"` | `GRAMPUS_SAFETY__INJECTION_DETECTION_LEVEL` | `strict` \| `balanced` \| `permissive` |
| `pii_detection_enabled` | `bool` | `True` | `GRAMPUS_SAFETY__PII_DETECTION_ENABLED` | Enable PII detection and redaction |
| `action_rate_limit_per_minute` | `int` | `60` | `GRAMPUS_SAFETY__ACTION_RATE_LIMIT_PER_MINUTE` | Max tool calls per minute across all agents |

---

## DaprConfig

**Env prefix:** `GRAMPUS_DAPR__`

| Field | Type | Default | Env var override | Description |
|-------|------|---------|-----------------|-------------|
| `host` | `str` | `"localhost"` | `GRAMPUS_DAPR__HOST` | Dapr sidecar hostname |
| `port` | `int` | `3500` | `GRAMPUS_DAPR__PORT` | Dapr HTTP sidecar port |
| `grpc_port` | `int` | `50001` | `GRAMPUS_DAPR__GRPC_PORT` | Dapr gRPC sidecar port |
| `state_store_name` | `str` | `"statestore"` | `GRAMPUS_DAPR__STATE_STORE_NAME` | Dapr state store component name |
| `pubsub_name` | `str` | `"pubsub"` | `GRAMPUS_DAPR__PUBSUB_NAME` | Dapr pub/sub component name |
| `cache_store_name` | `str` | `"cache"` | `GRAMPUS_DAPR__CACHE_STORE_NAME` | Dapr cache component name |

**Computed property:** `base_url` → `http://{host}:{port}`

---

## ObservabilityConfig

**Env prefix:** `GRAMPUS_OBSERVABILITY__`

| Field | Type | Default | Env var override | Description |
|-------|------|---------|-----------------|-------------|
| `otel_enabled` | `bool` | `True` | `GRAMPUS_OBSERVABILITY__OTEL_ENABLED` | Enable OTEL tracing |
| `otel_endpoint` | `str` | `"http://localhost:4317"` | `GRAMPUS_OBSERVABILITY__OTEL_ENDPOINT` | OTEL Collector gRPC endpoint |
| `log_level` | `str` | `"INFO"` | `GRAMPUS_OBSERVABILITY__LOG_LEVEL` | `DEBUG` \| `INFO` \| `WARNING` \| `ERROR` |
| `metrics_enabled` | `bool` | `True` | `GRAMPUS_OBSERVABILITY__METRICS_ENABLED` | Enable Prometheus metrics |

---

## Loading config in code

```python
from grampus.core.config import GrampusConfig

# Load from environment + grampus.yaml
config = GrampusConfig()

# Load from specific YAML file
config = GrampusConfig(_env_file="production.env")

# Override in code
config = GrampusConfig(model={"default_model": "claude-opus-4-7", "max_tokens": 8192})

# Access values
print(config.model.default_model)
print(config.dapr.base_url)           # http://localhost:3500

# API keys are SecretStr — .get_secret_value() to unwrap
key = config.model.anthropic_api_key.get_secret_value()
```

::: grampus.core.config.GrampusConfig
    options:
      show_source: false
      members: []
