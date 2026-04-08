#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

remove_dir() {
    local target="$1"
    if [[ -d "$target" ]]; then
        rm -rf "$target"
        printf 'removed %s\n' "${target#$ROOT_DIR/}"
    fi
}

remove_file() {
    local target="$1"
    if [[ -f "$target" ]]; then
        rm -f "$target"
        printf 'removed %s\n' "${target#$ROOT_DIR/}"
    fi
}

remove_dir "$ROOT_DIR/.pytest_cache"
remove_dir "$ROOT_DIR/paper/icml2025/build"

while IFS= read -r -d '' cache_dir; do
    remove_dir "$cache_dir"
done < <(find "$ROOT_DIR" -path "$ROOT_DIR/.venv" -prune -o -type d -name '__pycache__' -print0)

while IFS= read -r -d '' pyc_file; do
    remove_file "$pyc_file"
done < <(
    find "$ROOT_DIR" -path "$ROOT_DIR/.venv" -prune -o -type f \( -name '*.pyc' -o -name '*.pyo' \) -print0
)
