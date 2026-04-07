#!/usr/bin/env bash
set -euo pipefail

ACTION="${1:-start}"

TUNNEL_NAME="${VSCODE_TUNNEL_NAME:-uiuc-h200-${USER}}"
WORKSPACE_DIR="${VSCODE_TUNNEL_WORKSPACE:-$HOME/sandbox}"
STATE_DIR="${VSCODE_TUNNEL_STATE_DIR:-$HOME/.vscode-tunnel}"
LOG_FILE="${STATE_DIR}/tunnel.log"
PID_FILE="${STATE_DIR}/tunnel.pid"

find_code_bin() {
    if [[ -n "${VSCODE_TUNNEL_CODE_BIN:-}" && -x "${VSCODE_TUNNEL_CODE_BIN}" ]]; then
        printf '%s\n' "${VSCODE_TUNNEL_CODE_BIN}"
        return 0
    fi
    if command -v code >/dev/null 2>&1; then
        command -v code
        return 0
    fi
    if [[ -x "/sw/icrn/prod/bin/code" ]]; then
        printf '%s\n' "/sw/icrn/prod/bin/code"
        return 0
    fi
    printf 'Could not find a usable VS Code CLI binary.\n' >&2
    printf 'Set VSCODE_TUNNEL_CODE_BIN to the full path if needed.\n' >&2
    exit 1
}

CODE_BIN="$(find_code_bin)"

workspace_url() {
    printf 'https://vscode.dev/tunnel/%s%s\n' "${TUNNEL_NAME}" "${WORKSPACE_DIR}"
}

base_url() {
    printf 'https://vscode.dev/tunnel/%s\n' "${TUNNEL_NAME}"
}

running_pid() {
    if [[ -f "${PID_FILE}" ]]; then
        local pid
        pid="$(cat "${PID_FILE}")"
        if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
            printf '%s\n' "${pid}"
            return 0
        fi
    fi

    local found
    found="$(pgrep -f "code tunnel.*--name ${TUNNEL_NAME}" | head -n 1 || true)"
    if [[ -n "${found}" ]]; then
        printf '%s\n' "${found}" > "${PID_FILE}"
        printf '%s\n' "${found}"
        return 0
    fi
    return 1
}

print_status() {
    if pid="$(running_pid)"; then
        printf 'Tunnel is running.\n'
        printf 'PID: %s\n' "${pid}"
        printf 'Tunnel name: %s\n' "${TUNNEL_NAME}"
        printf 'Workspace: %s\n' "${WORKSPACE_DIR}"
        printf 'Open workspace: %s\n' "$(workspace_url)"
        printf 'Open tunnel root: %s\n' "$(base_url)"
        if [[ -f "${LOG_FILE}" ]]; then
            printf '\nRecent log output:\n'
            tail -n 20 "${LOG_FILE}" || true
        fi
    else
        printf 'Tunnel is not running.\n'
        printf 'Tunnel name: %s\n' "${TUNNEL_NAME}"
        printf 'Workspace: %s\n' "${WORKSPACE_DIR}"
    fi
}

start_tunnel() {
    mkdir -p "${STATE_DIR}"
    cd "${WORKSPACE_DIR}"

    if running_pid >/dev/null; then
        print_status
        return 0
    fi

    nohup "${CODE_BIN}" tunnel \
        --accept-server-license-terms \
        --name "${TUNNEL_NAME}" \
        > "${LOG_FILE}" 2>&1 < /dev/null &
    echo $! > "${PID_FILE}"

    sleep 5

    printf 'Started VS Code tunnel.\n'
    printf 'Tunnel name: %s\n' "${TUNNEL_NAME}"
    printf 'Workspace: %s\n' "${WORKSPACE_DIR}"
    printf 'Open workspace: %s\n' "$(workspace_url)"
    printf 'Open tunnel root: %s\n' "$(base_url)"

    if [[ -f "${LOG_FILE}" ]]; then
        printf '\nRecent log output:\n'
        tail -n 50 "${LOG_FILE}" || true
    fi
}

stop_tunnel() {
    if pid="$(running_pid)"; then
        kill "${pid}" || true
        sleep 1
        if kill -0 "${pid}" 2>/dev/null; then
            kill -9 "${pid}" || true
        fi
        rm -f "${PID_FILE}"
        printf 'Stopped tunnel %s.\n' "${TUNNEL_NAME}"
    else
        printf 'Tunnel %s is not running.\n' "${TUNNEL_NAME}"
    fi
}

show_logs() {
    if [[ -f "${LOG_FILE}" ]]; then
        tail -n 50 "${LOG_FILE}"
    else
        printf 'No log file found at %s\n' "${LOG_FILE}"
    fi
}

case "${ACTION}" in
    start)
        start_tunnel
        ;;
    status)
        print_status
        ;;
    stop)
        stop_tunnel
        ;;
    restart)
        stop_tunnel
        start_tunnel
        ;;
    logs)
        show_logs
        ;;
    *)
        printf 'Usage: %s [start|status|stop|restart|logs]\n' "$0" >&2
        exit 1
        ;;
esac
