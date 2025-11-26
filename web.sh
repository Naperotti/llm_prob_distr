#!/bin/bash

# Function to restart the server
restart_server() {
    echo "Stopping server..."
    pkill -f "uvicorn app:app"
    sleep 1
    echo "Starting server..."
    python -m uvicorn app:app --reload --host 127.0.0.1 --port 8000 &
    echo "Server restarted in background (PID: $!)"
}

# If called with 'restart' argument, restart the server
if [ "$1" = "restart" ]; then
    restart_server
else
    # Default: start server normally
    python -m uvicorn app:app --reload --host 127.0.0.1 --port 8000
fi
