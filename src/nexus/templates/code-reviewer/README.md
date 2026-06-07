# {{project_name}}

Automated code review using **{{model}}** on `{{repo_path}}`.

## Usage

```bash
# Review the entire repo
nexus run agent.py

# Review a specific concern
nexus run agent.py "Focus on security vulnerabilities in the auth module"
```

## Features

- Read-only: safety pipeline blocks all write tools
- Structured review: bugs, security issues, style, performance
- Configurable via `config.yaml`

## Output format

```
## Summary
...

## Issues
- [CRITICAL] auth.py:42 — SQL injection via unsanitized input
- [HIGH] ...

## Positive Observations
...

## Recommendations
...
```
