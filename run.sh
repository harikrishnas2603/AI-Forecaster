#!/bin/bash
echo "Starting backend on http://localhost:8000"
uvicorn backend.main:app --reload &

echo "Serving frontend on http://localhost:5500"
cd frontend
python3 -m http.server 5500
