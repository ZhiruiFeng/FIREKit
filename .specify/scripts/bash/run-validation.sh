#!/usr/bin/env bash
# run-validation.sh - Main validation orchestrator for FIREKit
# Runs multi-layer validation pipeline with smart change detection

set -e

SCRIPT_DIR="$(CDPATH="" cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

# Exit codes
EXIT_PASS=0
EXIT_FAIL=2
EXIT_ERROR=3
EXIT_FLAKY=4

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Arguments
QUICK_MODE=false
FULL_MODE=false
AUTO_FIX=true
DEBUG_MODE=false
INIT_MODE=false
REPORT_ONLY=false
LAYER=""
JSON_MODE=false

# Tracking
SIGNAL="PROMOTE"
FINAL_EXIT=$EXIT_PASS
AUTO_FIX_ATTEMPTS=0
MAX_AUTO_FIX=2
DECISIONS=()
FAILURES=()
START_TIME=$(date +%s)

# Parse arguments
for arg in "$@"; do
    case "$arg" in
        --quick) QUICK_MODE=true ;;
        --full) FULL_MODE=true ;;
        --fix) AUTO_FIX=true ;;
        --no-fix) AUTO_FIX=false ;;
        --debug) DEBUG_MODE=true ;;
        --init) INIT_MODE=true ;;
        --report-only) REPORT_ONLY=true ;;
        --layer=*) LAYER="${arg#*=}" ;;
        --json) JSON_MODE=true ;;
        --help|-h)
            echo "Usage: run-validation.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --quick       Skip integration tests and coverage"
            echo "  --full        Force full test suite (ignore change detection)"
            echo "  --fix         Enable auto-fix for fixable issues (default)"
            echo "  --no-fix      Disable auto-fix"
            echo "  --debug       Preserve debug logs on failure"
            echo "  --init        Create default VALIDATION.yaml"
            echo "  --report-only Regenerate reports from last run"
            echo "  --layer=N     Run only layer N (1-4)"
            echo "  --json        Output results as JSON"
            echo "  --help        Show this help message"
            echo ""
            echo "Exit Codes:"
            echo "  0  PASS    - All gates pass, ready for human review"
            echo "  2  FAIL    - Fixable issues or human decision needed"
            echo "  3  ERROR   - Infrastructure/environment errors"
            echo "  4  FLAKY   - Flaky tests detected"
            exit 0
            ;;
    esac
done

# Get repo root
REPO_ROOT=$(get_repo_root)
cd "$REPO_ROOT"

# Paths
VALIDATION_CONFIG="$REPO_ROOT/VALIDATION.yaml"
VALIDATION_DIR="$REPO_ROOT/.validation"
REPORT_JSON="$VALIDATION_DIR/validation_report.json"
REPORT_MD="$VALIDATION_DIR/validation_report.md"
CHANGE_MANIFEST="$VALIDATION_DIR/change_manifest.json"

# Logging functions
log_header() {
    echo -e "\n${BLUE}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}\n"
}

log_layer() {
    local layer="$1"
    local status="$2"
    local message="$3"

    case "$status" in
        "START") echo -e "${CYAN}[$layer]${NC} ${BLUE}Starting:${NC} $message" ;;
        "PASS")  echo -e "${CYAN}[$layer]${NC} ${GREEN}PASS:${NC} $message" ;;
        "FAIL")  echo -e "${CYAN}[$layer]${NC} ${RED}FAIL:${NC} $message" ;;
        "WARN")  echo -e "${CYAN}[$layer]${NC} ${YELLOW}WARN:${NC} $message" ;;
        "FIX")   echo -e "${CYAN}[$layer]${NC} ${YELLOW}FIX:${NC} $message" ;;
        "INFO")  echo -e "${CYAN}[$layer]${NC} ${NC}INFO:${NC} $message" ;;
        "SKIP")  echo -e "${CYAN}[$layer]${NC} ${YELLOW}SKIP:${NC} $message" ;;
    esac
}

add_decision() {
    DECISIONS+=("$1")
}

add_failure() {
    FAILURES+=("$1")
}

# Init mode - create default VALIDATION.yaml
if $INIT_MODE; then
    if [[ -f "$VALIDATION_CONFIG" ]]; then
        echo "VALIDATION.yaml already exists at $VALIDATION_CONFIG"
        echo "Delete it first if you want to regenerate."
        exit 1
    fi

    TEMPLATE="$REPO_ROOT/.specify/templates/validation-template.yaml"
    if [[ -f "$TEMPLATE" ]]; then
        cp "$TEMPLATE" "$VALIDATION_CONFIG"
        echo "Created VALIDATION.yaml from template"
    else
        echo "ERROR: Template not found at $TEMPLATE"
        echo "Please create VALIDATION.yaml manually."
        exit 1
    fi
    exit 0
fi

# Check for VALIDATION.yaml
if [[ ! -f "$VALIDATION_CONFIG" ]]; then
    echo -e "${RED}ERROR: VALIDATION.yaml not found at $REPO_ROOT${NC}"
    echo "Run 'run-validation.sh --init' to create default configuration"
    exit $EXIT_ERROR
fi

# Create validation output directory
mkdir -p "$VALIDATION_DIR"

# Layer 1: Environment Validation
run_environment_validation() {
    log_layer "ENV" "START" "Checking environment..."

    local env_pass=true

    # Check Python version
    if command -v python3 &> /dev/null; then
        local py_version=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
        local py_major=$(echo "$py_version" | cut -d. -f1)
        local py_minor=$(echo "$py_version" | cut -d. -f2)

        if [[ "$py_major" -ge 3 && "$py_minor" -ge 11 ]]; then
            log_layer "ENV" "PASS" "Python $py_version (>= 3.11 required)"
        else
            log_layer "ENV" "FAIL" "Python $py_version found, >= 3.11 required"
            add_failure "Python version $py_version is below required 3.11"
            env_pass=false
        fi
    else
        log_layer "ENV" "FAIL" "Python3 not found"
        add_failure "Python3 not found in PATH"
        env_pass=false
    fi

    # Check required tools
    for tool in ruff mypy pytest; do
        if command -v $tool &> /dev/null; then
            local version=$($tool --version 2>&1 | head -1)
            log_layer "ENV" "PASS" "$tool: $version"
        else
            log_layer "ENV" "FAIL" "$tool not found"
            add_failure "$tool not found in PATH"
            env_pass=false
        fi
    done

    if $env_pass; then
        log_layer "ENV" "PASS" "Environment validation passed"
        return 0
    else
        log_layer "ENV" "FAIL" "Environment validation failed"
        SIGNAL="ERROR"
        return 1
    fi
}

# Layer 2: Build Validation
run_build_validation() {
    log_layer "BUILD" "START" "Running build validation..."

    local build_pass=true
    local vectorforge_dir="$REPO_ROOT/vectorforge"

    if [[ ! -d "$vectorforge_dir" ]]; then
        log_layer "BUILD" "FAIL" "vectorforge directory not found"
        add_failure "vectorforge directory not found at $vectorforge_dir"
        SIGNAL="ERROR"
        return 1
    fi

    cd "$vectorforge_dir"

    # Lint check
    log_layer "BUILD" "INFO" "Running ruff lint..."
    if ruff check . 2>&1; then
        log_layer "BUILD" "PASS" "Lint check passed"
    else
        if $AUTO_FIX && [[ $AUTO_FIX_ATTEMPTS -lt $MAX_AUTO_FIX ]]; then
            log_layer "BUILD" "FIX" "Attempting auto-fix for lint errors..."
            ((AUTO_FIX_ATTEMPTS++))
            add_decision "Applied ruff --fix for lint errors (attempt $AUTO_FIX_ATTEMPTS)"

            if ruff check . --fix 2>&1; then
                log_layer "BUILD" "PASS" "Lint errors auto-fixed"
            else
                # Recheck after fix
                if ! ruff check . 2>&1; then
                    log_layer "BUILD" "FAIL" "Lint errors persist after auto-fix"
                    add_failure "Lint errors could not be auto-fixed"
                    SIGNAL="AUTO-FIX"
                    build_pass=false
                fi
            fi
        else
            log_layer "BUILD" "FAIL" "Lint errors found (auto-fix disabled or max attempts reached)"
            add_failure "Lint errors found"
            SIGNAL="AUTO-FIX"
            build_pass=false
        fi
    fi

    # Type check
    log_layer "BUILD" "INFO" "Running mypy type check..."
    if mypy vectorforge/ 2>&1; then
        log_layer "BUILD" "PASS" "Type check passed"
    else
        log_layer "BUILD" "FAIL" "Type errors found"
        add_failure "Type errors found by mypy"
        SIGNAL="HUMAN NEEDED"
        build_pass=false
    fi

    # Format check
    log_layer "BUILD" "INFO" "Running ruff format check..."
    if ruff format --check . 2>&1; then
        log_layer "BUILD" "PASS" "Format check passed"
    else
        if $AUTO_FIX; then
            log_layer "BUILD" "FIX" "Applying formatting..."
            ruff format . 2>&1
            add_decision "Applied ruff format to fix formatting issues"
            log_layer "BUILD" "PASS" "Formatting applied"
        else
            log_layer "BUILD" "WARN" "Formatting issues (auto-fix disabled)"
        fi
    fi

    cd "$REPO_ROOT"

    if $build_pass; then
        log_layer "BUILD" "PASS" "Build validation passed"
        return 0
    else
        log_layer "BUILD" "FAIL" "Build validation failed"
        return 1
    fi
}

# Layer 3: Test Validation
run_test_validation() {
    log_layer "TEST" "START" "Running tests..."

    local test_pass=true
    local vectorforge_dir="$REPO_ROOT/vectorforge"

    cd "$vectorforge_dir"

    # Determine test scope
    local test_cmd="pytest tests/ -v --tb=short"
    local test_scope="full suite"

    if ! $FULL_MODE; then
        # Get change detection info
        if [[ -x "$SCRIPT_DIR/detect-changes.sh" ]]; then
            local change_info=$("$SCRIPT_DIR/detect-changes.sh" --json 2>/dev/null || echo '{}')

            # Parse test scope from change info
            local scoped_tests=$(echo "$change_info" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    scope = d.get('test_scope', [])
    if scope and not d.get('full_suite', False):
        print(' or '.join(scope))
except:
    pass
" 2>/dev/null)

            local docs_only=$(echo "$change_info" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    print('true' if d.get('docs_only', False) else 'false')
except:
    print('false')
" 2>/dev/null)

            if [[ "$docs_only" == "true" ]]; then
                log_layer "TEST" "SKIP" "Documentation-only changes, skipping tests"
                add_decision "Skipped tests: only documentation files changed"
                cd "$REPO_ROOT"
                return 0
            fi

            if [[ -n "$scoped_tests" ]]; then
                test_cmd="pytest tests/ -v --tb=short -k '$scoped_tests'"
                test_scope="scoped: $scoped_tests"
                add_decision "Scoped tests to: $scoped_tests (based on changed files)"
            fi
        fi
    else
        add_decision "Running full test suite (--full flag)"
    fi

    log_layer "TEST" "INFO" "Test scope: $test_scope"

    # Run tests
    if eval $test_cmd 2>&1; then
        log_layer "TEST" "PASS" "Tests passed"
    else
        log_layer "TEST" "FAIL" "Test failures detected"
        add_failure "Test failures detected"
        SIGNAL="HUMAN NEEDED"
        test_pass=false
    fi

    # Coverage check (skip in quick mode)
    if ! $QUICK_MODE && $test_pass; then
        log_layer "TEST" "INFO" "Running coverage check..."
        local cov_output=$(pytest tests/ --cov=vectorforge --cov-report=term-missing --cov-fail-under=70 2>&1) || true

        if echo "$cov_output" | grep -q "TOTAL.*[0-9]\+%" ; then
            local coverage=$(echo "$cov_output" | grep "TOTAL" | awk '{print $NF}' | tr -d '%')
            if [[ -n "$coverage" ]]; then
                if [[ "$coverage" -ge 70 ]]; then
                    log_layer "TEST" "PASS" "Coverage: ${coverage}% (>= 70% required)"
                else
                    log_layer "TEST" "WARN" "Coverage: ${coverage}% (< 70% threshold)"
                    add_failure "Code coverage ${coverage}% is below 70% threshold"
                fi
            fi
        fi
    fi

    cd "$REPO_ROOT"

    if $test_pass; then
        log_layer "TEST" "PASS" "Test validation passed"
        return 0
    else
        log_layer "TEST" "FAIL" "Test validation failed"
        return 1
    fi
}

# Layer 4: Runtime Validation
run_runtime_validation() {
    log_layer "RUNTIME" "START" "Running runtime validation..."

    local runtime_pass=true

    # Smoke test: module import
    log_layer "RUNTIME" "INFO" "Testing module import..."
    if python3 -c "import vectorforge" 2>&1; then
        log_layer "RUNTIME" "PASS" "Module import successful"
    else
        log_layer "RUNTIME" "FAIL" "Module import failed"
        add_failure "Failed to import vectorforge module"
        SIGNAL="HUMAN NEEDED"
        runtime_pass=false
    fi

    # Smoke test: core classes
    if $runtime_pass; then
        log_layer "RUNTIME" "INFO" "Testing core class imports..."
        if python3 -c "from vectorforge import VectorizedBacktester, EventDrivenBacktester, HybridRunner" 2>&1; then
            log_layer "RUNTIME" "PASS" "Core classes available"
        else
            log_layer "RUNTIME" "FAIL" "Core classes import failed"
            add_failure "Failed to import core classes from vectorforge"
            SIGNAL="HUMAN NEEDED"
            runtime_pass=false
        fi
    fi

    # Smoke test: strategy import
    if $runtime_pass; then
        log_layer "RUNTIME" "INFO" "Testing strategy imports..."
        if python3 -c "from vectorforge.strategy import MomentumStrategy, MovingAverageCrossover" 2>&1; then
            log_layer "RUNTIME" "PASS" "Strategy classes available"
        else
            log_layer "RUNTIME" "WARN" "Strategy classes import failed (non-critical)"
        fi
    fi

    if $runtime_pass; then
        log_layer "RUNTIME" "PASS" "Runtime validation passed"
        return 0
    else
        log_layer "RUNTIME" "FAIL" "Runtime validation failed"
        return 1
    fi
}

# Documentation sync check
run_docs_sync() {
    log_layer "DOCS" "START" "Checking documentation sync..."

    if [[ -x "$SCRIPT_DIR/sync-docs.sh" ]]; then
        if "$SCRIPT_DIR/sync-docs.sh" --check 2>&1; then
            log_layer "DOCS" "PASS" "Documentation sync check passed"
        else
            log_layer "DOCS" "WARN" "Documentation may be out of sync"
        fi
    else
        log_layer "DOCS" "SKIP" "sync-docs.sh not found"
    fi
}

# Generate reports
generate_reports() {
    local end_time=$(date +%s)
    local duration=$((end_time - START_TIME))
    local timestamp=$(date -Iseconds)

    # Determine signal emoji
    local signal_emoji=""
    case "$SIGNAL" in
        "PROMOTE")      signal_emoji="PROMOTE" ;;
        "AUTO-FIX")     signal_emoji="AUTO-FIX" ;;
        "HUMAN NEEDED") signal_emoji="HUMAN NEEDED" ;;
        "ERROR")        signal_emoji="ERROR" ;;
        "FLAKY")        signal_emoji="FLAKY" ;;
        "KILL")         signal_emoji="KILL" ;;
    esac

    # Generate JSON report
    cat > "$REPORT_JSON" <<EOF
{
  "version": "1.0.0",
  "timestamp": "$timestamp",
  "duration_seconds": $duration,
  "signal": "$SIGNAL",
  "exit_code": $FINAL_EXIT,
  "decisions": [$(printf '"%s",' "${DECISIONS[@]}" | sed 's/,$//')],
  "failures": [$(printf '"%s",' "${FAILURES[@]}" | sed 's/,$//')],
  "quick_mode": $QUICK_MODE,
  "full_mode": $FULL_MODE,
  "auto_fix_enabled": $AUTO_FIX,
  "auto_fix_attempts": $AUTO_FIX_ATTEMPTS
}
EOF

    # Generate Markdown report
    cat > "$REPORT_MD" <<EOF
# Validation Report

**Timestamp**: $timestamp
**Duration**: ${duration}s
**Signal**: $signal_emoji
**Exit Code**: $FINAL_EXIT

---

## Summary

| Metric | Value |
|--------|-------|
| Signal | $SIGNAL |
| Duration | ${duration}s |
| Quick Mode | $QUICK_MODE |
| Full Mode | $FULL_MODE |
| Auto-Fix Enabled | $AUTO_FIX |
| Auto-Fix Attempts | $AUTO_FIX_ATTEMPTS |

---

## Decisions Made

EOF

    if [[ ${#DECISIONS[@]} -gt 0 ]]; then
        for decision in "${DECISIONS[@]}"; do
            echo "- $decision" >> "$REPORT_MD"
        done
    else
        echo "_No decisions recorded._" >> "$REPORT_MD"
    fi

    cat >> "$REPORT_MD" <<EOF

---

## Failures

EOF

    if [[ ${#FAILURES[@]} -gt 0 ]]; then
        for failure in "${FAILURES[@]}"; do
            echo "- $failure" >> "$REPORT_MD"
        done

        cat >> "$REPORT_MD" <<EOF

---

## Human Actions Required

Based on the failures above, please:

1. Review the specific error messages in the console output
2. Fix the underlying issues in the code
3. Re-run validation with \`run-validation.sh\`

---

## First-Principle Training

### Why This Requires Human Judgment

EOF

        case "$SIGNAL" in
            "HUMAN NEEDED")
                cat >> "$REPORT_MD" <<EOF
The validation detected issues that require human understanding to resolve:

- **Type errors**: The type system cannot infer your intent. You must decide on the correct types.
- **Test failures**: These may indicate logic bugs that require domain knowledge to fix.
- **Coverage gaps**: You need to decide what additional tests are needed.

### Common Patterns

- For type errors: Check return types match expected types, handle Optional cases explicitly
- For test failures: Run the specific test with \`-v --tb=long\` for detailed output
- For coverage: Focus on testing critical code paths first
EOF
                ;;
            "AUTO-FIX")
                cat >> "$REPORT_MD" <<EOF
The auto-fix system could not fully resolve all issues. Remaining issues may need manual intervention.

- **Lint errors**: Some lint rules may require code restructuring
- **Format errors**: Some formatting may conflict with your intended structure

Run \`ruff check . --fix\` manually to see detailed fix suggestions.
EOF
                ;;
        esac
    else
        echo "_No failures recorded._" >> "$REPORT_MD"
    fi

    cat >> "$REPORT_MD" <<EOF

---

## Reproduction Steps

\`\`\`bash
# Re-run full validation
cd $REPO_ROOT
.specify/scripts/bash/run-validation.sh

# Run specific layer
.specify/scripts/bash/run-validation.sh --layer=2  # Build only
.specify/scripts/bash/run-validation.sh --layer=3  # Tests only

# Run with full test suite
.specify/scripts/bash/run-validation.sh --full

# Run in quick mode
.specify/scripts/bash/run-validation.sh --quick
\`\`\`

---

*Report generated by speckit.validate*
EOF

    log_layer "REPORT" "INFO" "Reports generated:"
    log_layer "REPORT" "INFO" "  JSON: $REPORT_JSON"
    log_layer "REPORT" "INFO" "  Markdown: $REPORT_MD"
}

# Main execution
main() {
    log_header "FIREKit Validation Pipeline"
    echo "Started: $(date -Iseconds)"
    echo "Working directory: $REPO_ROOT"
    echo ""

    # Run layers
    if [[ -z "$LAYER" || "$LAYER" == "1" ]]; then
        if ! run_environment_validation; then
            FINAL_EXIT=$EXIT_ERROR
        fi
    fi

    if [[ $FINAL_EXIT -eq 0 ]] && [[ -z "$LAYER" || "$LAYER" == "2" ]]; then
        if ! run_build_validation; then
            FINAL_EXIT=$EXIT_FAIL
        fi
    fi

    if [[ $FINAL_EXIT -eq 0 ]] && [[ -z "$LAYER" || "$LAYER" == "3" ]]; then
        if ! run_test_validation; then
            FINAL_EXIT=$EXIT_FAIL
        fi
    fi

    if [[ $FINAL_EXIT -eq 0 ]] && [[ -z "$LAYER" || "$LAYER" == "4" ]]; then
        if ! run_runtime_validation; then
            FINAL_EXIT=$EXIT_FAIL
        fi
    fi

    # Documentation sync (informational only)
    if [[ -z "$LAYER" ]]; then
        run_docs_sync
    fi

    # Cleanup debug logs (unless --debug and failed)
    if ! $DEBUG_MODE || [[ $FINAL_EXIT -eq 0 ]]; then
        if [[ -x "$SCRIPT_DIR/remove-debug-logs.sh" ]]; then
            "$SCRIPT_DIR/remove-debug-logs.sh" 2>/dev/null || true
        fi
    fi

    # Generate reports
    generate_reports

    # Output final signal
    log_header "Validation Complete"

    case "$SIGNAL" in
        "PROMOTE")
            echo -e "${GREEN}Signal: PROMOTE${NC}"
            echo "All gates pass, ready for human review"
            ;;
        "AUTO-FIX")
            echo -e "${YELLOW}Signal: AUTO-FIX${NC}"
            echo "Fixable issues detected, some may need manual intervention"
            ;;
        "HUMAN NEEDED")
            echo -e "${YELLOW}Signal: HUMAN NEEDED${NC}"
            echo "Requires architectural/logic decisions"
            ;;
        "ERROR")
            echo -e "${RED}Signal: ERROR${NC}"
            echo "Infrastructure/environment errors detected"
            ;;
        "FLAKY")
            echo -e "${YELLOW}Signal: FLAKY${NC}"
            echo "Flaky tests detected"
            ;;
        "KILL")
            echo -e "${RED}Signal: KILL${NC}"
            echo "Repeated failures, consider abandoning"
            ;;
    esac

    echo ""
    echo "Exit code: $FINAL_EXIT"
    echo "Report: $REPORT_MD"

    if $JSON_MODE; then
        cat "$REPORT_JSON"
    fi

    exit $FINAL_EXIT
}

main "$@"
