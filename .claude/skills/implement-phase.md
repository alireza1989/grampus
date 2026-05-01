---
name: implement-phase
description: Execute a specific phase from the Nexus implementation plan. Reads the plan, expands the detailed spec, implements code, writes tests, and verifies acceptance criteria.
allowed-tools: Read, Write, Bash, Grep, Glob
---

# Implement Phase Workflow

When asked to implement a phase, follow this exact sequence:

## Step 1: Read Context
1. Read `PLAN.md` and find the phase section
2. Read the corresponding `docs/phases/phase-NN.md` for detailed spec
3. Read `docs/architecture/decisions.md` for relevant ADRs
4. Identify which existing code this phase depends on (read those modules)

## Step 2: Plan (Do NOT implement yet)
1. List every file that will be created or modified
2. For each file, outline the classes/functions it will contain
3. Identify any dependency or import that needs to exist first
4. Present the plan and wait for approval

## Step 3: Implement (Test-First)
For each file in the plan:
1. Write the test file FIRST (`tests/module/test_*.py`)
2. Run the test to confirm it FAILS (expected)
3. Write the implementation
4. Run the test to confirm it PASSES
5. Run `uv run ruff check` and `uv run mypy` on the new file
6. Fix any issues before moving to the next file

## Step 4: Verify Acceptance Criteria
1. Go through EVERY acceptance criteria checkbox in PLAN.md
2. Run the specific verification command for each
3. Only mark the phase complete when ALL criteria pass

## Step 5: Post-Implementation
1. Update `__init__.py` exports for any new public APIs
2. Run full check: `uv run ruff check . && uv run mypy src/ && uv run pytest`
3. Commit with: `git add -A && git commit -m "feat(module): phase N description"`

Focus on: $ARGUMENTS
