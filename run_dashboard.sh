#!/usr/bin/env bash
# Launch the presentable Streamlit dashboard for the distress-prediction project.
set -e
cd "$(dirname "$0")"
source .venv/bin/activate
export PYTHONPATH=src
echo "Launching dashboard at http://localhost:8501"
streamlit run dashboard/app.py --server.headless true
