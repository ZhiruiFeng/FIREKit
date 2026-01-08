#!/usr/bin/env bash
# inject-debug-logs.sh - Inject temporary debug logging for validation diagnostics
# Adds marker-delimited debug prints to help diagnose test failures

set -e

SCRIPT_DIR="$(CDPATH="" cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

FILES=""
FUNCTIONS=""
MARKER_START="# [VALIDATION_DEBUG_START]"
MARKER_END="# [VALIDATION_DEBUG_END]"

for arg in "$@"; do
    case "$arg" in
        --files=*) FILES="${arg#*=}" ;;
        --functions=*) FUNCTIONS="${arg#*=}" ;;
        --help|-h)
            echo "Usage: inject-debug-logs.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --files=FILE1,FILE2    Comma-separated list of files to inject into"
            echo "  --functions=FUNC1,FUNC2 Comma-separated list of functions to target"
            echo "  --help                 Show this help"
            echo ""
            echo "Injects debug logging at function entry points. Original files are"
            echo "backed up to .validation/debug_logs/ for restoration."
            exit 0
            ;;
    esac
done

REPO_ROOT=$(get_repo_root)
VALIDATION_DIR="$REPO_ROOT/.validation"
DEBUG_DIR="$VALIDATION_DIR/debug_logs"

# Create backup directory
mkdir -p "$DEBUG_DIR"

# Track what we inject
INJECTED_FILES=()

inject_into_file() {
    local file="$1"
    local full_path=""

    # Resolve to full path
    if [[ -f "$file" ]]; then
        full_path="$file"
    elif [[ -f "$REPO_ROOT/$file" ]]; then
        full_path="$REPO_ROOT/$file"
    else
        echo "Warning: File not found: $file"
        return 1
    fi

    # Skip if not a Python file
    if [[ ! "$full_path" =~ \.py$ ]]; then
        echo "Skipping non-Python file: $full_path"
        return 0
    fi

    # Check if already injected
    if grep -q "$MARKER_START" "$full_path" 2>/dev/null; then
        echo "Already injected: $full_path"
        return 0
    fi

    # Create backup
    local backup_name=$(echo "$full_path" | sed 's|/|_|g')
    cp "$full_path" "$DEBUG_DIR/${backup_name}.orig"

    # Inject debug logging using Python
    python3 << PYEOF
import re
import sys

file_path = '''$full_path'''
marker_start = '''$MARKER_START'''
marker_end = '''$MARKER_END'''

with open(file_path, 'r') as f:
    content = f.read()

# Pattern to match function definitions (def name(...):)
# Capture the indentation, function definition, and any docstring
pattern = r'^(\s*)(def\s+(\w+)\s*\([^)]*\)\s*(?:->\s*[^:]+)?:\s*)(\n\s*"""[^"]*"""\n|\n\s*\'\'\'[^\']*\'\'\'\n|\n)?'

def inject_log(match):
    indent = match.group(1)
    func_def = match.group(2)
    func_name = match.group(3)
    docstring = match.group(4) or '\n'

    # Create debug log with proper indentation
    log_indent = indent + '    '
    debug_code = f'''{log_indent}{marker_start}
{log_indent}print(f'[VALIDATION_DEBUG] Entering {func_name}')
{log_indent}{marker_end}
'''

    return f'{indent}{func_def}{docstring}{debug_code}'

new_content = re.sub(pattern, inject_log, content, flags=re.MULTILINE)

with open(file_path, 'w') as f:
    f.write(new_content)

print(f'Injected debug logs into: {file_path}')
PYEOF

    INJECTED_FILES+=("$full_path")
}

# Process files
if [[ -n "$FILES" ]]; then
    IFS=',' read -ra FILE_ARRAY <<< "$FILES"
    for file in "${FILE_ARRAY[@]}"; do
        inject_into_file "$file"
    done
else
    echo "No files specified. Use --files=file1.py,file2.py"
    echo ""
    echo "Example usage:"
    echo "  inject-debug-logs.sh --files=vectorforge/engine/base.py,vectorforge/strategy/base.py"
    exit 1
fi

# Summary
echo ""
echo "=== Injection Summary ==="
echo "Backup directory: $DEBUG_DIR"
echo "Files injected: ${#INJECTED_FILES[@]}"
for f in "${INJECTED_FILES[@]}"; do
    echo "  - $f"
done
echo ""
echo "To remove debug logs, run: remove-debug-logs.sh"
