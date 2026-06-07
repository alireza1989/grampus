# {{project_name}}

Customer support agent using **{{model}}** with RAG over `{{knowledge_base_path}}`.

## Setup

1. Populate `{{knowledge_base_path}}/` with `.md` or `.txt` knowledge articles
2. Run the agent:

```bash
nexus run agent.py "How do I reset my password?"
```

## Features

- RAG over local knowledge base (file_read + sql_query tools)
- Episodic memory for multi-turn conversations
- Strict safety pipeline with PII redaction
- Configurable via `config.yaml`
