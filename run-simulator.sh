#!/bin/bash
# Neptune Data Simulator Runner
# Runs the SeaOWL simulator in a continuous loop

set -e

DEFAULT_OUTPUT_NYC="data/ship/seaowl_live.ndjson"
DEFAULT_OUTPUT_GULF="data/ship/seaowl_gulf_live.ndjson"
DEFAULT_ECO_OUTPUT_NYC="data/ship/ecofl_live.ndjson"
DEFAULT_ECO_OUTPUT_GULF="data/ship/ecofl_gulf_live.ndjson"

# Default configuration (NYC path)
PATTERN_FLAG="--nyc"
OUTPUT_PATH=""
ECO_OUTPUT_PATH=""
EXTRA_ARGS=()

print_banner() {
    echo "🌊 Starting Neptune Data Simulator..."
    echo "📊 Running SeaOWL sensor simulation in continuous loop"
    echo "📍 Pattern: ${PATTERN_FLAG#--}"
    echo "📁 SeaOWL Output: ${OUTPUT_PATH}"
    echo "📁 ECO FL Output: ${ECO_OUTPUT_PATH}"
    echo ""
    echo "Press Ctrl+C to stop the simulator"
    echo ""
}

usage() {
    cat <<'EOF'
Usage: run-simulator.sh [--nyc | --gulf] [--output PATH] [--eco-output PATH] [extra sim args]

  --nyc             Use the original NYC drift (default).
  --gulf            Use the Cerulean Gulf slick route.
  --output PATH     Override the SeaOWL NDJSON output path.
  --eco-output PATH Override the ECO FL NDJSON output path.

Any additional arguments are forwarded to both simulators.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --nyc)
            PATTERN_FLAG="--nyc"
            if [[ -z "${OUTPUT_PATH}" ]]; then
                OUTPUT_PATH="${DEFAULT_OUTPUT_NYC}"
            fi
            shift
            ;;
        --gulf)
            PATTERN_FLAG="--gulf"
            if [[ -z "${OUTPUT_PATH}" ]]; then
                OUTPUT_PATH="${DEFAULT_OUTPUT_GULF}"
            fi
            shift
            ;;
        --output)
            if [[ -z "$2" ]]; then
                echo "⚠️  --output requires a path" >&2
                usage
                exit 1
            fi
            OUTPUT_PATH="$2"
            shift 2
            ;;
        --eco-output)
            if [[ -z "$2" ]]; then
                echo "⚠️  --eco-output requires a path" >&2
                usage
                exit 1
            fi
            ECO_OUTPUT_PATH="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

if [[ -z "${OUTPUT_PATH}" ]]; then
    if [[ "${PATTERN_FLAG}" == "--gulf" ]]; then
        OUTPUT_PATH="${DEFAULT_OUTPUT_GULF}"
    else
        OUTPUT_PATH="${DEFAULT_OUTPUT_NYC}"
    fi
fi

if [[ -z "${ECO_OUTPUT_PATH}" ]]; then
    if [[ "${PATTERN_FLAG}" == "--gulf" ]]; then
        ECO_OUTPUT_PATH="${DEFAULT_ECO_OUTPUT_GULF}"
    else
        ECO_OUTPUT_PATH="${DEFAULT_ECO_OUTPUT_NYC}"
    fi
fi

print_banner

# Create data directory if it doesn't exist
mkdir -p data/ship

# Function to handle cleanup on exit
cleanup() {
    echo ""
    echo "🛑 Stopping data simulator..."
    if [[ -n "${SEAOWL_PID}" ]]; then
        kill "${SEAOWL_PID}" >/dev/null 2>&1 || true
        wait "${SEAOWL_PID}" 2>/dev/null || true
    fi
    if [[ -n "${ECOFL_PID}" ]]; then
        kill "${ECOFL_PID}" >/dev/null 2>&1 || true
        wait "${ECOFL_PID}" 2>/dev/null || true
    fi
    exit 0
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM

# Continuous loop
LOOP_COUNT=1
SEAOWL_PID=""
ECOFL_PID=""
while true; do
    echo "🔄 Starting simulation loop #${LOOP_COUNT}"

    # Run the simulators concurrently
    uv run tools/sim_seaowl.py ${PATTERN_FLAG} --sleep --output "${OUTPUT_PATH}" "${EXTRA_ARGS[@]}" &
    SEAOWL_PID=$!
    uv run tools/sim_ecofl.py ${PATTERN_FLAG} --sleep --output "${ECO_OUTPUT_PATH}" "${EXTRA_ARGS[@]}" &
    ECOFL_PID=$!

    wait $SEAOWL_PID
    SEAOWL_STATUS=$?
    wait $ECOFL_PID
    ECOFL_STATUS=$?

    SEAOWL_PID=""
    ECOFL_PID=""

    if [[ $SEAOWL_STATUS -eq 0 && $ECOFL_STATUS -eq 0 ]]; then
        echo "✅ Simulation loop #${LOOP_COUNT} completed successfully"
    else
        echo "❌ Simulation loop #${LOOP_COUNT} failed (SeaOWL=$SEAOWL_STATUS, ECO_FL=$ECOFL_STATUS)"
        echo "⏸️  Waiting 5 seconds before retrying..."
        sleep 5
    fi

    LOOP_COUNT=$((LOOP_COUNT + 1))

    # Brief pause between loops
    echo "⏳ Waiting 2 seconds before next loop..."
    sleep 2
done
