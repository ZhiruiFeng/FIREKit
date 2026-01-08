#!/usr/bin/env bash
# remove-debug-logs.sh - Remove injected debug logging after validation
# Restores original files from backups or removes marker-delimited blocks

set -e

SCRIPT_DIR="$(CDPATH="" cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

FORCE=false
VERIFY=true

for arg in "$@"; do
    case "$arg" in
        --force) FORCE=true ;;
        --no-verify) VERIFY=false ;;
        --help|-h)
            echo "Usage: remove-debug-logs.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --force       Force removal even if backups are missing"
            echo "  --no-verify   Skip verification after removal"
            echo "  --help        Show this help"
            exit 0
            ;;
    esac
done

REPO_ROOT=$(get_repo_root)
VALIDATION_DIR="$REPO_ROOT/.validation"
DEBUG_DIR="$VALIDATION_DIR/debug_logs"

MARKER_START="# \[VALIDATION_DEBUG_START\]"
MARKER_END="# \[VALIDATION_DEBUG_END\]"

RESTORED_COUNT=0
CLEANED_COUNT=0

# Method 1: Restore from backups
restore_from_backups() {
    if [[ ! -d "$DEBUG_DIR" ]]; then
        return 1
    fi

    local found_backups=false
    for orig_file in "$DEBUG_DIR"/*.orig 2>/dev/null; do
        if [[ -f "$orig_file" ]]; then
            found_backups=true

            # Extract original path from backup name
            local backup_name=$(basename "$orig_file" .orig)
            # Convert underscores back to slashes (reverse of inject script)
            local original_path=$(echo "$backup_name" | sed 's|_|/|g')

            if [[ -f "$original_path" ]]; then
                cp "$orig_file" "$original_path"
                echo "Restored: $original_path"
                ((RESTORED_COUNT++))
            fi
        fi
    done

    if $found_backups; then
        # Clean up backup directory
        rm -rf "$DEBUG_DIR"
        return 0
    fi

    return 1
}

# Method 2: Remove marker-delimited blocks inline
remove_markers_inline() {
    echo "Scanning for debug markers in repository..."

    # Find all Python files with markers
    local files_with_markers=$(find "$REPO_ROOT" -name "*.py" \
        -not -path "*/.validation/*" \
        -not -path "*/__pycache__/*" \
        -not -path "*/.git/*" \
        -not -path "*/*-env/*" \
        -not -path "*/venv/*" \
        -exec grep -l "\[VALIDATION_DEBUG_START\]" {} \; 2>/dev/null || true)

    if [[ -z "$files_with_markers" ]]; then
        echo "No files with debug markers found"
        return 0
    fi

    for file in $files_with_markers; do
        echo "Cleaning markers from: $file"

        # Use Python for reliable multi-line removal
        python3 << PYEOF
import re

file_path = '''$file'''
marker_start = '# [VALIDATION_DEBUG_START]'
marker_end = '# [VALIDATION_DEBUG_END]'

with open(file_path, 'r') as f:
    content = f.read()

# Pattern to match marker blocks (including the print statement between them)
pattern = r'\s*# \[VALIDATION_DEBUG_START\]\s*\n.*?# \[VALIDATION_DEBUG_END\]\s*\n?'

new_content = re.sub(pattern, '', content, flags=re.DOTALL)

with open(file_path, 'w') as f:
    f.write(new_content)
PYEOF

        ((CLEANED_COUNT++))
    done
}

# Verify no markers remain
verify_cleanup() {
    echo ""
    echo "Verifying cleanup..."

    local remaining=$(find "$REPO_ROOT" -name "*.py" \
        -not -path "*/.validation/*" \
        -not -path "*/__pycache__/*" \
        -not -path "*/.git/*" \
        -not -path "*/*-env/*" \
        -not -path "*/venv/*" \
        -exec grep -l "\[VALIDATION_DEBUG_START\]" {} \; 2>/dev/null || true)

    if [[ -n "$remaining" ]]; then
        echo "WARNING: Debug markers still found in:"
        echo "$remaining"
        return 1
    else
        echo "Verification passed: No debug markers remaining"
        return 0
    fi
}

# Main execution
echo "=== Debug Log Cleanup ==="
echo ""

# Try restore from backups first
if restore_from_backups; then
    echo ""
    echo "Restored $RESTORED_COUNT files from backups"
else
    echo "No backups found, scanning for inline markers..."
    remove_markers_inline
    echo ""
    echo "Cleaned markers from $CLEANED_COUNT files"
fi

# Verify if requested
if $VERIFY; then
    verify_cleanup
fi

echo ""
echo "=== Cleanup Complete ==="
echo "Restored from backup: $RESTORED_COUNT"
echo "Cleaned inline: $CLEANED_COUNT"
