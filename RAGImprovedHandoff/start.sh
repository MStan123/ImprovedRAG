#!/bin/bash

echo "🚀 Starting Birmarket Support System..."

# Проверяем Redis
if ! pgrep -x "redis-server" > /dev/null
then
    echo "Starting Redis..."
    redis-server --daemonize yes
fi

# Запускаем Chat Server
echo "Starting Chat Server (port 8001)..."
python chat_server.py &
CHAT_PID=$!

# Запускаем Dashboard
echo "Starting Dashboard (port 8000)..."
python support_dashboard.py &
DASHBOARD_PID=$!

echo ""
echo "✅ System started!"
echo "📊 Dashboard: http://localhost:8000"
echo "💬 Chat Server: http://localhost:8001"
echo ""
echo "Press Ctrl+C to stop all services"

# Ожидаем Ctrl+C
trap "echo 'Stopping...'; kill $CHAT_PID $DASHBOARD_PID; exit" INT
wait