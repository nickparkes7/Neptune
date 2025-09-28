#!/bin/bash
# Neptune Data Simulator Runner
# Runs the SeaOWL simulator in a continuous loop

set -e

DEFAULT_OUTPUT_NYC="data/ship/seaowl_live.ndjson"
DEFAULT_OUTPUT_GULF="data/ship/seaowl_gulf_live.ndjson"

# Default configuration (NYC path)
PATTERN_FLAG="--nyc"
OUTPUT_PATH=""
EXTRA_ARGS=()

print_banner() {
    echo "🌊 Starting Neptune Data Simulator..."
    echo "📊 Running SeaOWL sensor simulation in continuous loop"
    echo "📍 Pattern: ${PATTERN_FLAG#--}"
    echo "📁 Output: ${OUTPUT_PATH}"
    echo ""
    echo "Press Ctrl+C to stop the simulator"
    echo ""
}

usage() {
    cat <<'EOF'
Usage: run-simulator.sh [--nyc | --gulf] [--output PATH] [extra sim args]

  --nyc           Use the original NYC drift (default).
  --gulf          Use the Cerulean Gulf slick route.
  --output PATH   Override the NDJSON output path.

Any additional arguments are forwarded to tools/sim_seaowl.py.
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

print_banner

# Create data directory if it doesn't exist
mkdir -p data/ship

# Function to handle cleanup on exit
cleanup() {
    echo ""
    echo "🛑 Stopping data simulator..."
    exit 0
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM

# Continuous loop
LOOP_COUNT=1
while true; do
    echo "🔄 Starting simulation loop #${LOOP_COUNT}"

    # Run the SeaOWL simulator
    if uv run tools/sim_seaowl.py ${PATTERN_FLAG} --sleep --output "${OUTPUT_PATH}" "${EXTRA_ARGS[@]}"; then
        echo "✅ Simulation loop #${LOOP_COUNT} completed successfully"
    else
        echo "❌ Simulation loop #${LOOP_COUNT} failed with exit code $?"
        echo "⏸️  Waiting 5 seconds before retrying..."
        sleep 5
    fi

    LOOP_COUNT=$((LOOP_COUNT + 1))

    # Brief pause between loops
    echo "⏳ Waiting 2 seconds before next loop..."
    sleep 2
done
