#!/bin/bash
# NeuroGuard UI Setup Script
# Run this on your GPU server to start the full UI

echo "=============================================="
echo "  NeuroGuard - AI Safety Evaluation Platform"
echo "=============================================="

# Navigate to project root
cd "$(dirname "$0")/.."

# Install backend dependencies
echo ""
echo "[1/4] Installing backend dependencies..."
cd backend
pip install -r requirements.txt
pip install torch transformers accelerate bitsandbytes

# Start backend in background
echo ""
echo "[2/4] Starting backend server..."
python -m uvicorn main:app --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!
echo "Backend PID: $BACKEND_PID"

# Wait for backend to start
sleep 5

# Install frontend dependencies  
echo ""
echo "[3/4] Installing frontend dependencies..."
cd ../frontend
npm install

# Start frontend
echo ""
echo "[4/4] Starting frontend server..."
npm run dev -- --host 0.0.0.0 --port 3000 &
FRONTEND_PID=$!

echo ""
echo "=============================================="
echo "  NeuroGuard is running!"
echo "=============================================="
echo ""
echo "  Frontend: http://$(hostname -I | awk '{print $1}'):3000"
echo "  Backend:  http://$(hostname -I | awk '{print $1}'):8000"
echo ""
echo "  Press Ctrl+C to stop all services"
echo "=============================================="

# Wait for Ctrl+C
trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit" INT
wait
