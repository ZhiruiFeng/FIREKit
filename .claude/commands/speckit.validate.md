---
description: Run multi-layer validation pipeline with smart change detection, auto-fix capabilities, and human handoff guidance.
handoffs:
  - label: Fix Detected Issues
    agent: speckit.implement
    prompt: Fix the validation failures identified in the report. Focus on the specific files and issues mentioned.
    send: false
  - label: Analyze Root Cause
    agent: speckit.analyze
    prompt: Analyze the root cause of validation failures and check for cross-artifact consistency issues.
    send: false
---

## User Input

```text
$ARGUMENTS
```

You **MUST** consider the user input before proceeding (if not empty).

## Overview

The `/speckit.validate` command runs a comprehensive validation pipeline that:

1. **Detects what changed** since the last commit
2. **Runs only relevant validation layers** based on changes
3. **Attempts auto-fix** for fixable issues (lint, format)
4. **Produces structured reports** (JSON + Markdown)
5. **Provides human handoff guidance** when decisions are needed

## Validation Layers

| Layer | Purpose | Tools | Signal on Failure |
|-------|---------|-------|-------------------|
| 1. Environment | Verify Python, tools, venv | Shell checks | ERROR |
| 2. Build | Lint, type-check, format | ruff, mypy | AUTO-FIX or HUMAN NEEDED |
| 3. Test | Unit/integration tests | pytest | HUMAN NEEDED |
| 4. Runtime | Smoke tests, imports | Python exec | HUMAN NEEDED |

## Signal System

| Signal | Exit Code | Meaning |
|--------|-----------|---------|
| PROMOTE | 0 | All gates pass, ready for human review |
| AUTO-FIX | 2 | Fixable issues, agent attempts repair |
| HUMAN NEEDED | 2 | Requires architectural/logic decisions |
| ERROR | 3 | Infrastructure/environment errors |
| FLAKY | 4 | Flaky tests detected |
| KILL | 2 | Repeated failures (3+), consider abandoning |

## Execution Steps

1. **Parse Arguments**: Check user input for flags like `--quick`, `--full`, `--fix`, `--layer=N`.

2. **Check Configuration**:
   ```bash
   REPO_ROOT=$(git rev-parse --show-toplevel 2>/dev/null || pwd)
   VALIDATION_CONFIG="$REPO_ROOT/VALIDATION.yaml"

   if [[ ! -f "$VALIDATION_CONFIG" ]]; then
       echo "VALIDATION.yaml not found. Creating from template..."
       # Copy from .specify/templates/validation-template.yaml
   fi
   ```

3. **Smart Change Detection**:
   ```bash
   .specify/scripts/bash/detect-changes.sh --json
   ```

   Parse output to determine:
   - Changed files since last commit
   - Affected modules based on file patterns
   - Test scope (full suite vs targeted tests)
   - Whether to skip tests (docs-only changes)

4. **Run Validation Pipeline**:
   ```bash
   .specify/scripts/bash/run-validation.sh [FLAGS]
   ```

   The script will:
   - Layer 1: Verify Python >= 3.11, ruff, mypy, pytest installed
   - Layer 2: Run `ruff check`, `mypy`, `ruff format --check` (auto-fix if enabled)
   - Layer 3: Run pytest with coverage (scoped by change detection)
   - Layer 4: Test module imports and smoke tests

5. **Process Results**:
   - Read `.validation/validation_report.json` for structured data
   - Read `.validation/validation_report.md` for human-readable summary
   - Determine final signal based on all layer results

6. **Generate Human Handoff** (if HUMAN NEEDED):
   - List specific files requiring attention
   - Describe decisions made during validation
   - Provide reproduction steps
   - Include first-principle training content

## Command Arguments

| Argument | Description |
|----------|-------------|
| (none) | Run full validation pipeline |
| `--init` | Create default VALIDATION.yaml from template |
| `--quick` | Skip integration tests and coverage checks |
| `--full` | Force full test suite (ignore change detection) |
| `--fix` | Enable auto-fix for lint/format issues (default) |
| `--no-fix` | Disable auto-fix |
| `--layer=N` | Run only layer N (1=env, 2=build, 3=test, 4=runtime) |
| `--debug` | Preserve debug logs on failure |
| `--json` | Output results as JSON |

## Output Files

| File | Purpose |
|------|---------|
| `.validation/validation_report.json` | Structured validation data |
| `.validation/validation_report.md` | Human-readable summary |
| `.validation/change_manifest.json` | Detected changes |
| `.validation/debug_logs/` | Preserved debug logs (if --debug) |

## Human Handoff Report Format

When HUMAN NEEDED signal is produced, provide:

### 1. Summary
- Signal status and duration
- Layer results (pass/fail for each)
- Auto-fix attempts made

### 2. Failed Checks
For each failure, include:
- File path and line number
- Error message
- Classification (type_error, test_failure, lint_error)

### 3. Decisions Made
- What scoping decisions were made (full suite vs targeted)
- What auto-fixes were applied
- Why certain checks were skipped

### 4. Human Actions Required
Specific, actionable items:
1. File to edit
2. What to fix
3. Suggested approach

### 5. First-Principle Training
Explain why this requires human judgment:
- **Type errors**: Type system cannot infer intent
- **Test failures**: Domain knowledge required
- **Coverage gaps**: Architecture decisions needed

Common patterns and learning resources.

### 6. Reproduction Steps
Exact commands to reproduce locally:
```bash
cd /path/to/repo
.specify/scripts/bash/run-validation.sh --full
pytest vectorforge/tests/test_engine.py -v --tb=long
```

## Constitution Alignment

This skill enforces FIREKit constitution principles:

- **III. Test-First Development**: Validates test coverage and TDD discipline
- **II. Performance-First Design**: Can trigger benchmarks on perf-critical changes
- **IV. Production-Parity**: Same validation in dev and CI environments
- **V. Risk-First Execution**: Catches errors before production

## Integration with SpecKit Workflow

```
/speckit.specify -> /speckit.clarify -> /speckit.plan -> /speckit.tasks
                                                              |
                                              /speckit.validate (pre-implementation)
                                                              |
                                              /speckit.implement
                                                              |
                                              /speckit.validate (post-implementation)
                                                              |
                                              /speckit.analyze -> PR creation
```

## Example Usage

### Basic Validation
```
/speckit.validate
```
Runs full pipeline with smart change detection.

### Quick Check
```
/speckit.validate --quick
```
Skips coverage and integration tests for faster feedback.

### After Implementation
```
/speckit.validate --full
```
Forces full test suite before PR creation.

### Initialize Config
```
/speckit.validate --init
```
Creates VALIDATION.yaml from template.

### Debug Failing Tests
```
/speckit.validate --debug --layer=3
```
Runs only tests with debug logging preserved on failure.

## Notes

- Always run from repository root
- VALIDATION.yaml must exist (use --init to create)
- Reports are written to `.validation/` directory (gitignored)
- Debug logs are automatically cleaned up unless --debug flag is used
- Auto-fix is enabled by default; use --no-fix to disable
