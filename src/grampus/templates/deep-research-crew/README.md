# {{project_name}} — Deep Research Crew

A {{max_search_rounds}}-round multi-agent research pipeline.

## Agents

- **Planner** (`{{powerful_model}}`) — decomposes question into sub-queries
- **Searcher A + B** (`{{fast_model}}`, parallel) — search web for each query
- **Fact Checker** (`{{balanced_model}}`) — validates and cross-references claims
- **Synthesizer** (`{{powerful_model}}`) — builds structured synthesis
- **Critic** (`{{balanced_model}}`) — identifies gaps, triggers re-search if needed
- **Writer** (`{{powerful_model}}`) — formats final report with citations

## Usage

```bash
nexus run agent.py "What are the competitive dynamics of the vector database market in 2026?"
```

## Configuration

Edit `agents.yaml` to customize agent models, system prompts, and iteration limits.
Edit `config.yaml` to adjust research parameters.

## How it works

1. **Plan**: Planner decomposes question into {{max_sources_per_query}} sub-queries
2. **Search**: Two searchers run in parallel (up to {{max_search_rounds}} rounds)
3. **Fact-check**: Claims are verified across multiple sources
4. **Synthesize**: Verified findings are synthesized into a coherent draft
5. **Critique**: Critic checks for gaps and may trigger another search round
6. **Write**: Writer formats the final polished report with citations
