#!/bin/bash

# Function to stop a process by name
stop_process() {
    local process_name=$1
    pkill -f "$process_name"
}

# Stop the Uvicorn server
stop_process "uvicorn score:app"

# Stop the Yarn development server
stop_process "yarn run dev"

# Stop the Python scripts
stop_process "python3 upload_file.py"
stop_process "python3 cron.py"
stop_process "/bin/bash ./start.sh"

echo "All specified servers have been stopped."
