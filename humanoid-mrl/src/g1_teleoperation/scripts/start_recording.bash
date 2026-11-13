#!/bin/bash
set -e

TOPIC="/record_episode"
MSG_TYPE="std_msgs/msg/UInt8"
LOG_FILE="/tmp/recorder_session.log"

pub_msg() {
    timeout 2 ros2 topic pub --once "$TOPIC" "$MSG_TYPE" "{data: $1}" < /dev/null > /dev/null 2>&1 || true
}

cleanup() {
    trap - EXIT INT TERM
    echo ""
    echo "Stopping current episode and shutting down recorder..."

    if kill -0 "$RECORDER_PID" 2>/dev/null; then
        pub_msg 0 # Gracefully save whatever is buffered
        sleep 1
        kill -INT "$RECORDER_PID" 2>/dev/null || true
    fi
    exit 0
}

echo "Cleaning up any orphaned recorder nodes..."
pkill -f "recorder_node" || true
sleep 0.5

trap cleanup EXIT INT TERM

echo "Starting recorder node in the background..."
echo "(Node logs are being saved to $LOG_FILE)"

ros2 run g1_teleoperation recorder_node < /dev/null > "$LOG_FILE" 2>&1 &
RECORDER_PID=$!

echo -n "Waiting for recorder node to connect"
until ros2 topic info "$TOPIC" 2>/dev/null | grep -q "Subscription count: [1-9]"; do
    sleep 0.5
    echo -n "."
    if ! kill -0 "$RECORDER_PID" 2>/dev/null; then
        echo ""
        echo "ERROR: recorder_node exited unexpectedly. Check $LOG_FILE for details."
        cat "$LOG_FILE"
        exit 1
    fi
done
echo " connected."

while read -r -t 0.1; do :; done < /dev/tty || true

SESSION_EP=1
echo "Starting Session Episode $SESSION_EP..."
pub_msg 1

echo ""
echo "Controls:"
echo "  [Enter]  Progress to the next step"
echo "  [q]      Quit at any time"
echo ""

while true; do
    # --- PHASE 1: RECORDING ---
    set +e
    if ! read -r -p "🔴 [RECORDING] Session Ep $SESSION_EP. Press [Enter] to FINISH attempt (or 'q' to quit): " input < /dev/tty; then
        echo -e "\nTerminal disconnected. Safe quitting..."
        cleanup
    fi
    set -e

    if [[ "$input" == "q" || "$input" == "Q" ]]; then
        cleanup
    fi

    echo "Freezing capture for Session Ep $SESSION_EP..."
    pub_msg 2 # Node stops recording frames, but holds data in memory
    sleep 0.5

    # --- PHASE 1.5: DECISION ---
    set +e
    # -n 1 allows for a single keystroke (no Enter needed) for faster workflow
    if ! read -r -n 1 -p "💾 Save this episode to dataset? (y/n): " choice < /dev/tty; then
        cleanup
    fi
    set -e
    echo "" # Print a newline since -n 1 doesn't create one naturally

    if [[ "$choice" == "n" || "$choice" == "N" ]]; then
        echo "🗑️ Discarding Session Ep $SESSION_EP..."
        pub_msg 3
        # Notice we DO NOT increment SESSION_EP here, so you can retry the same episode number.
    else
        # Defaults to save if you press 'y' or accidentally hit Enter
        echo "Saving Session Ep $SESSION_EP..."
        pub_msg 0
        SESSION_EP=$((SESSION_EP + 1))
    fi
    sleep 0.5

    # --- PHASE 2: PAUSED ---
    set +e
    if ! read -r -p "⚪ [PAUSED] Ready for Session Ep $SESSION_EP. Press [Enter] to START (or 'q' to quit): " input < /dev/tty; then
        echo -e "\nTerminal disconnected. Safe quitting..."
        cleanup
    fi
    set -e

    if [[ "$input" == "q" || "$input" == "Q" ]]; then
        cleanup
    fi

    echo "Starting Session Ep $SESSION_EP..."
    pub_msg 1
done
