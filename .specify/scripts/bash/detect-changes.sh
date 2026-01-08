#!/usr/bin/env bash
# detect-changes.sh - Smart change detection for validation scoping
# Analyzes git diff to determine which tests need to run

set -e

SCRIPT_DIR="$(CDPATH="" cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

JSON_MODE=false
SINCE="HEAD~1"

for arg in "$@"; do
    case "$arg" in
        --json) JSON_MODE=true ;;
        --since=*) SINCE="${arg#*=}" ;;
        --help|-h)
            echo "Usage: detect-changes.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --json        Output as JSON"
            echo "  --since=REF   Compare against REF (default: HEAD~1)"
            echo "  --help        Show this help"
            exit 0
            ;;
    esac
done

REPO_ROOT=$(get_repo_root)
cd "$REPO_ROOT"

# Get changed files
if has_git; then
    CHANGED_FILES=$(git diff --name-only "$SINCE" 2>/dev/null || echo "")
else
    CHANGED_FILES=""
fi

# Initialize arrays
declare -a TEST_FILES=()
declare -a AFFECTED_MODULES=()
FULL_SUITE=false
DOCS_ONLY=true
SKIP_TESTS=false

# Process each changed file
for file in $CHANGED_FILES; do
    # Check if any non-doc file changed
    if [[ ! "$file" =~ \.md$ ]] && [[ ! "$file" =~ ^docs/ ]]; then
        DOCS_ONLY=false
    fi

    # Determine test scope based on file patterns
    case "$file" in
        # Engine changes
        vectorforge/vectorforge/engine/*.py|vectorforge/engine/*.py)
            TEST_FILES+=("test_engine")
            AFFECTED_MODULES+=("engine")
            ;;

        # Strategy changes
        vectorforge/vectorforge/strategy/*.py|vectorforge/strategy/*.py)
            TEST_FILES+=("test_strategy")
            AFFECTED_MODULES+=("strategy")
            ;;

        # Analysis changes
        vectorforge/vectorforge/analysis/*.py|vectorforge/analysis/*.py)
            TEST_FILES+=("test_analysis")
            AFFECTED_MODULES+=("analysis")
            ;;

        # Optimization changes
        vectorforge/vectorforge/optimization/*.py|vectorforge/optimization/*.py)
            TEST_FILES+=("test_optimization")
            AFFECTED_MODULES+=("optimization")
            ;;

        # Data changes
        vectorforge/vectorforge/data/*.py|vectorforge/data/*.py)
            TEST_FILES+=("test_data")
            AFFECTED_MODULES+=("data")
            ;;

        # Execution changes
        vectorforge/vectorforge/execution/*.py|vectorforge/execution/*.py)
            TEST_FILES+=("test_execution")
            AFFECTED_MODULES+=("execution")
            ;;

        # Config changes -> full suite
        vectorforge/vectorforge/config.py|vectorforge/config.py)
            FULL_SUITE=true
            AFFECTED_MODULES+=("config")
            ;;

        # Test fixtures -> full suite
        vectorforge/tests/conftest.py|conftest.py)
            FULL_SUITE=true
            ;;

        # Project config -> full suite
        vectorforge/pyproject.toml|pyproject.toml)
            FULL_SUITE=true
            ;;

        # VALIDATION.yaml -> full suite
        VALIDATION.yaml)
            FULL_SUITE=true
            ;;

        # Init file changes -> full suite
        vectorforge/vectorforge/__init__.py)
            FULL_SUITE=true
            AFFECTED_MODULES+=("init")
            ;;

        # Documentation only
        *.md|docs/*)
            # Already handled by DOCS_ONLY check
            ;;

        # Other Python files
        *.py)
            # Unknown Python file, trigger full suite for safety
            if [[ "$file" =~ vectorforge/ ]]; then
                FULL_SUITE=true
                AFFECTED_MODULES+=("unknown")
            fi
            ;;
    esac
done

# Remove duplicates
if [[ ${#TEST_FILES[@]} -gt 0 ]]; then
    TEST_FILES=($(printf '%s\n' "${TEST_FILES[@]}" | sort -u))
fi
if [[ ${#AFFECTED_MODULES[@]} -gt 0 ]]; then
    AFFECTED_MODULES=($(printf '%s\n' "${AFFECTED_MODULES[@]}" | sort -u))
fi

# If no changes detected, mark as docs only to skip tests
if [[ -z "$CHANGED_FILES" ]]; then
    DOCS_ONLY=true
    SKIP_TESTS=true
fi

# If only docs changed, skip tests
if $DOCS_ONLY; then
    SKIP_TESTS=true
fi

# Output results
if $JSON_MODE; then
    # Build JSON arrays
    changed_json="[]"
    test_json="[]"
    modules_json="[]"

    if [[ -n "$CHANGED_FILES" ]]; then
        changed_json=$(echo "$CHANGED_FILES" | python3 -c "
import sys, json
files = [f.strip() for f in sys.stdin.read().strip().split('\n') if f.strip()]
print(json.dumps(files))
" 2>/dev/null || echo "[]")
    fi

    if [[ ${#TEST_FILES[@]} -gt 0 ]]; then
        test_json=$(printf '%s\n' "${TEST_FILES[@]}" | python3 -c "
import sys, json
items = [i.strip() for i in sys.stdin.read().strip().split('\n') if i.strip()]
print(json.dumps(items))
" 2>/dev/null || echo "[]")
    fi

    if [[ ${#AFFECTED_MODULES[@]} -gt 0 ]]; then
        modules_json=$(printf '%s\n' "${AFFECTED_MODULES[@]}" | python3 -c "
import sys, json
items = [i.strip() for i in sys.stdin.read().strip().split('\n') if i.strip()]
print(json.dumps(items))
" 2>/dev/null || echo "[]")
    fi

    cat <<EOF
{
  "changed_files": $changed_json,
  "test_scope": $test_json,
  "affected_modules": $modules_json,
  "full_suite": $FULL_SUITE,
  "docs_only": $DOCS_ONLY,
  "skip_tests": $SKIP_TESTS,
  "since": "$SINCE"
}
EOF
else
    echo "=== Change Detection Report ==="
    echo ""
    echo "Since: $SINCE"
    echo ""
    echo "Changed files:"
    if [[ -n "$CHANGED_FILES" ]]; then
        echo "$CHANGED_FILES" | while read -r file; do
            echo "  - $file"
        done
    else
        echo "  (none detected)"
    fi
    echo ""
    echo "Test scope: ${TEST_FILES[*]:-full suite}"
    echo "Affected modules: ${AFFECTED_MODULES[*]:-all}"
    echo "Full suite required: $FULL_SUITE"
    echo "Docs only: $DOCS_ONLY"
    echo "Skip tests: $SKIP_TESTS"
fi
