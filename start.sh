#!/bin/sh
# Start FastAPI backend internally in the background
python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000 &

# Wait for the backend to initialize
sleep 3

# Start Streamlit frontend in the foreground on the cloud-allocated port
streamlit run frontend/dashboard.py --server.port $PORT --server.address 0.0.0.0
