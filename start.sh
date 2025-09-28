#!/bin/bash
set -e

# Display Python and pip version
python --version
pip --version

# List installed packages
echo "Installed packages:"
pip list

# Check if we have the optional packages installed
cat /app/installed_optional_packages.txt

# Start the application
echo "Starting FastAPI application on port $PORT"
exec uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}
