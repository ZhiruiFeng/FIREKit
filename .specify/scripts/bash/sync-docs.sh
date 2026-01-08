#!/usr/bin/env bash
# sync-docs.sh - Verify documentation is in sync with code
# Checks README examples, exported APIs, and version consistency

set -e

SCRIPT_DIR="$(CDPATH="" cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

CHECK_ONLY=false
JSON_MODE=false
VERBOSE=false

for arg in "$@"; do
    case "$arg" in
        --check) CHECK_ONLY=true ;;
        --json) JSON_MODE=true ;;
        --verbose|-v) VERBOSE=true ;;
        --help|-h)
            echo "Usage: sync-docs.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --check      Check only, don't modify anything"
            echo "  --json       Output as JSON"
            echo "  --verbose    Show detailed output"
            echo "  --help       Show this help"
            exit 0
            ;;
    esac
done

REPO_ROOT=$(get_repo_root)
cd "$REPO_ROOT"

# Track issues and warnings
declare -a ISSUES=()
declare -a WARNINGS=()
declare -a PASSED=()

add_issue() {
    ISSUES+=("$1")
    if $VERBOSE; then
        echo "ISSUE: $1"
    fi
}

add_warning() {
    WARNINGS+=("$1")
    if $VERBOSE; then
        echo "WARNING: $1"
    fi
}

add_passed() {
    PASSED+=("$1")
    if $VERBOSE; then
        echo "PASS: $1"
    fi
}

# Check README code examples for syntax errors
check_readme_examples() {
    local readme="$1"
    local readme_name=$(basename "$readme")

    if [[ ! -f "$readme" ]]; then
        add_warning "README not found: $readme"
        return
    fi

    $VERBOSE && echo "Checking examples in: $readme"

    # Extract and check Python code blocks
    python3 << PYEOF
import re
import sys

readme_path = '''$readme'''
readme_name = '''$readme_name'''

with open(readme_path, 'r') as f:
    content = f.read()

# Find Python code blocks
pattern = r'\`\`\`python\n(.*?)\`\`\`'
blocks = re.findall(pattern, content, re.DOTALL)

if not blocks:
    print(f"INFO:No Python code blocks found in {readme_name}")
    sys.exit(0)

errors = []
for i, block in enumerate(blocks, 1):
    # Skip blocks with obvious placeholders
    if '<your' in block.lower() or 'xxx' in block.lower() or '...' in block:
        continue

    # Skip very short blocks (likely fragments)
    if len(block.strip()) < 20:
        continue

    # Try to compile (syntax check only)
    try:
        compile(block, f'{readme_name}_block_{i}', 'exec')
    except SyntaxError as e:
        errors.append(f"Block {i}: {e.msg} at line {e.lineno}")

if errors:
    for err in errors:
        print(f"SYNTAX_ERROR:{readme_name}:{err}")
else:
    print(f"PASS:All {len(blocks)} code blocks in {readme_name} have valid syntax")
PYEOF
}

# Check that exported APIs are documented
check_api_documented() {
    local module_init="$REPO_ROOT/vectorforge/vectorforge/__init__.py"
    local readme="$REPO_ROOT/vectorforge/README.md"

    if [[ ! -f "$module_init" ]]; then
        add_warning "Module __init__.py not found: $module_init"
        return
    fi

    if [[ ! -f "$readme" ]]; then
        add_warning "Module README not found: $readme"
        return
    fi

    $VERBOSE && echo "Checking API documentation coverage..."

    python3 << PYEOF
import ast
import sys

module_init = '''$module_init'''
readme_path = '''$readme'''

# Read __all__ exports
with open(module_init, 'r') as f:
    tree = ast.parse(f.read())

exports = []
for node in ast.walk(tree):
    if isinstance(node, ast.Assign):
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == '__all__':
                if isinstance(node.value, ast.List):
                    for elt in node.value.elts:
                        if isinstance(elt, ast.Constant):
                            exports.append(elt.value)

if not exports:
    print("INFO:No __all__ exports found")
    sys.exit(0)

# Read README content
with open(readme_path, 'r') as f:
    readme_content = f.read()

undocumented = []
documented = []

for export in exports:
    if export in readme_content:
        documented.append(export)
    else:
        undocumented.append(export)

if undocumented:
    for name in undocumented:
        print(f"UNDOCUMENTED:{name}")
else:
    print(f"PASS:All {len(exports)} exports are documented")

if documented:
    print(f"INFO:Documented exports: {len(documented)}/{len(exports)}")
PYEOF
}

# Check version consistency
check_version_consistency() {
    local pyproject="$REPO_ROOT/vectorforge/pyproject.toml"
    local readme="$REPO_ROOT/vectorforge/README.md"

    if [[ ! -f "$pyproject" ]]; then
        add_warning "pyproject.toml not found"
        return
    fi

    $VERBOSE && echo "Checking version consistency..."

    # Extract version from pyproject.toml
    local pyproject_version=$(grep -E '^version\s*=' "$pyproject" | head -1 | sed 's/.*"\(.*\)".*/\1/' || echo "")

    if [[ -z "$pyproject_version" ]]; then
        add_warning "Could not extract version from pyproject.toml"
        return
    fi

    $VERBOSE && echo "Version in pyproject.toml: $pyproject_version"

    # Check if version is mentioned in README
    if [[ -f "$readme" ]]; then
        if grep -q "$pyproject_version" "$readme"; then
            add_passed "Version $pyproject_version found in README"
        else
            add_warning "Version $pyproject_version not mentioned in README"
        fi
    fi
}

# Run all checks
run_checks() {
    echo "=== Documentation Sync Check ==="
    echo ""

    # Check README examples
    for readme in "$REPO_ROOT/README.md" "$REPO_ROOT/vectorforge/README.md"; do
        result=$(check_readme_examples "$readme" 2>&1)

        while IFS= read -r line; do
            case "$line" in
                SYNTAX_ERROR:*)
                    add_issue "${line#SYNTAX_ERROR:}"
                    ;;
                PASS:*)
                    add_passed "${line#PASS:}"
                    ;;
                INFO:*)
                    $VERBOSE && echo "${line#INFO:}"
                    ;;
            esac
        done <<< "$result"
    done

    # Check API documentation
    result=$(check_api_documented 2>&1)
    while IFS= read -r line; do
        case "$line" in
            UNDOCUMENTED:*)
                add_warning "Export '${line#UNDOCUMENTED:}' not documented in README"
                ;;
            PASS:*)
                add_passed "${line#PASS:}"
                ;;
            INFO:*)
                $VERBOSE && echo "${line#INFO:}"
                ;;
        esac
    done <<< "$result"

    # Check version consistency
    check_version_consistency
}

# Output results
output_results() {
    if $JSON_MODE; then
        # Build JSON arrays
        issues_json="[]"
        warnings_json="[]"
        passed_json="[]"

        if [[ ${#ISSUES[@]} -gt 0 ]]; then
            issues_json=$(printf '%s\n' "${ISSUES[@]}" | python3 -c "
import sys, json
items = [i.strip() for i in sys.stdin.read().strip().split('\n') if i.strip()]
print(json.dumps(items))
" 2>/dev/null || echo "[]")
        fi

        if [[ ${#WARNINGS[@]} -gt 0 ]]; then
            warnings_json=$(printf '%s\n' "${WARNINGS[@]}" | python3 -c "
import sys, json
items = [i.strip() for i in sys.stdin.read().strip().split('\n') if i.strip()]
print(json.dumps(items))
" 2>/dev/null || echo "[]")
        fi

        if [[ ${#PASSED[@]} -gt 0 ]]; then
            passed_json=$(printf '%s\n' "${PASSED[@]}" | python3 -c "
import sys, json
items = [i.strip() for i in sys.stdin.read().strip().split('\n') if i.strip()]
print(json.dumps(items))
" 2>/dev/null || echo "[]")
        fi

        local status="pass"
        if [[ ${#ISSUES[@]} -gt 0 ]]; then
            status="fail"
        elif [[ ${#WARNINGS[@]} -gt 0 ]]; then
            status="warn"
        fi

        cat <<EOF
{
  "status": "$status",
  "issues": $issues_json,
  "warnings": $warnings_json,
  "passed": $passed_json
}
EOF
    else
        echo ""
        echo "=== Results ==="
        echo ""

        if [[ ${#PASSED[@]} -gt 0 ]]; then
            echo "Passed checks: ${#PASSED[@]}"
            for item in "${PASSED[@]}"; do
                echo "  [OK] $item"
            done
            echo ""
        fi

        if [[ ${#WARNINGS[@]} -gt 0 ]]; then
            echo "Warnings: ${#WARNINGS[@]}"
            for item in "${WARNINGS[@]}"; do
                echo "  [WARN] $item"
            done
            echo ""
        fi

        if [[ ${#ISSUES[@]} -gt 0 ]]; then
            echo "Issues: ${#ISSUES[@]}"
            for item in "${ISSUES[@]}"; do
                echo "  [FAIL] $item"
            done
            echo ""
        fi

        if [[ ${#ISSUES[@]} -eq 0 && ${#WARNINGS[@]} -eq 0 ]]; then
            echo "Documentation sync: OK"
        elif [[ ${#ISSUES[@]} -eq 0 ]]; then
            echo "Documentation sync: OK (with warnings)"
        else
            echo "Documentation sync: FAILED"
        fi
    fi
}

# Main
run_checks
output_results

# Exit with appropriate code
if [[ ${#ISSUES[@]} -gt 0 ]]; then
    exit 1
else
    exit 0
fi
