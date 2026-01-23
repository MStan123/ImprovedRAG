from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, Body
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
import json
from contextlib import asynccontextmanager
from typing import Set
from support_handoff import handoff
from pathlib import Path

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("🔔 Listening for Redis notifications...")
    yield
    # Shutdown
    print("🛑 Shutting down Redis listener...")

app = FastAPI(lifespan=lifespan)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Создаём папки если их нет
Path("templates").mkdir(exist_ok=True)
Path("static").mkdir(exist_ok=True)

# Templates
templates = Jinja2Templates(directory="templates")

# Static files
try:
    app.mount("/static", StaticFiles(directory="static"), name="static")
except:
    pass


# WebSocket для уведомлений операторов
class DashboardNotifier:
    def __init__(self):
        self.active_connections: Set[WebSocket] = set()

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.add(websocket)
        print(f"📊 Dashboard connected. Total: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        self.active_connections.discard(websocket)
        print(f"📊 Dashboard disconnected. Total: {len(self.active_connections)}")

    async def broadcast(self, message: dict):
        """Отправляет уведомление всем подключенным операторам"""
        dead_connections = set()

        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except:
                dead_connections.add(connection)

        # Удаляем мёртвые подключения
        self.active_connections -= dead_connections


notifier = DashboardNotifier()


@app.get("/", response_class=HTMLResponse)
async def dashboard_home(request: Request):
    """Главная страница dashboard"""
    queue = handoff.get_queue()
    agents = handoff.get_online_agents()

    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "queue_count": len(queue),
        "agents_count": len(agents)
    })


@app.get("/api/queue")
async def get_queue():
    """API: текущая очередь поддержки"""
    queue = handoff.get_queue()
    return {
        "queue": queue,
        "count": len(queue)
    }

from chat_history_manager import chat_history
from fastapi import HTTPException
# 🆕 НОВЫЙ ENDPOINT: Получить полную историю сессии
@app.get("/api/session/{session_id}/history")
async def get_session_history(session_id: str, last_n: int = 100):
    session = handoff.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    user_id = session.get('user_id')
    if not user_id:
        raise HTTPException(status_code=400, detail="No user_id associated with session")

    # Получаем ВСЮ историю из chat_history (предполагаю, что это список объектов Message)
    full_history = chat_history.get_history(user_id)

    if not full_history:
        return {
            "session_id": session_id,
            "user_id": user_id,
            "messages": []
        }

    # Фильтруем только пользовательские и бот-ответы (игнорируем system, tool и т.д.)
    bot_history = [
        msg for msg in full_history
        if getattr(msg, "role", None) in ["user", "assistant"]
    ]

    # Берём последние N сообщений
    recent = bot_history[-last_n:] if len(bot_history) > last_n else bot_history

    # Форматируем для фронта
    messages = []
    for msg in recent:
        messages.append({
            "role": msg.role,
            "content": msg.content or "",
            "timestamp": getattr(msg, "timestamp", "1970-01-01T00:00:00"),
            "source": msg.metadata.get("source", "rag") if hasattr(msg, "metadata") else "rag",
            "token_count": getattr(msg, "token_count", 0)
        })

    # Сортировка по времени (на случай, если история не в порядке)
    messages.sort(key=lambda x: x["timestamp"])

    return {
        "session_id": session_id,
        "user_id": user_id,
        "message_count": len(bot_history),  # общее кол-во в RAG-истории
        "displayed_count": len(messages),
        "messages": messages
    }


# 🆕 НОВЫЙ ENDPOINT: Получить summary для dashboard
@app.get("/api/session/{session_id}/summary")
async def get_session_summary(session_id: str, last_n: int = 10):
    """
    🆕 Получить сводку для отображения в dashboard
    """
    from chat_history_manager import chat_history

    session = handoff.get_session(session_id)
    if not session:
        return JSONResponse(
            {"error": "Session not found"},
            status_code=404
        )

    user_id = session.get('user_id')

    if not user_id:
        return {
            "summary": "История разговора недоступна",
            "message_count": 0
        }

    # Используем метод, который уже фильтрует
    summary = chat_history.get_summary_for_agent(user_id, last_n=last_n)

    # Подсчитываем сообщения (только bot)
    full_history = chat_history.get_history(user_id)
    bot_count = len([msg for msg in full_history if msg.role in ["user", "assistant"]])

    return {
        "summary": summary,
        "message_count": bot_count,
        "user_id": user_id
    }


@app.get("/api/history/stats")
async def get_history_stats():
    """Статистика по всем активным разговорам"""
    from chat_history_manager import chat_history

    # Получаем все ключи истории
    pattern = f"{chat_history.history_prefix}*"
    keys = chat_history.redis.keys(pattern)

    stats = {
        "total_active_conversations": len(keys),
        "conversations": []
    }

    for key in keys[:100]:  # Ограничиваем для производительности
        user_id = key.replace(chat_history.history_prefix, "")
        metadata = chat_history.get_metadata(user_id)

        if metadata:
            stats["conversations"].append({
                "user_id": user_id,
                **metadata
            })

    return stats


@app.get("/queue", response_class=HTMLResponse)
async def queue_page(request: Request):
    """Страница с очередью"""
    queue = handoff.get_queue()

    return templates.TemplateResponse("queue.html", {
        "request": request,
        "queue": queue
    })


@app.get("/chat/{session_id}", response_class=HTMLResponse)
async def chat_interface(request: Request, session_id: str, agent_id: str = None):
    """
    🔧 ИСПРАВЛЕНО: Интерфейс чата для оператора
    Теперь загружает полную историю через новый API
    """
    from chat_history_manager import chat_history

    session = handoff.get_session(session_id)

    if not session:
        return HTMLResponse("<h1>Session not found</h1>", status_code=404)

    # Получаем summary (фильтрованный только bot)
    user_id = session.get('user_id')
    conversation_summary = ""

    if user_id:
        conversation_summary = chat_history.get_summary_for_agent(user_id, last_n=10)

    return templates.TemplateResponse("agent_chat.html", {
        "request": request,
        "session_id": session_id,
        "session": session,
        "agent_id": agent_id or "default_agent",
        "conversation_summary": conversation_summary  # 🆕 Передаём summary в шаблон
    })


@app.websocket("/ws/dashboard")
async def dashboard_websocket(websocket: WebSocket):
    """WebSocket для real-time обновлений dashboard"""
    await notifier.connect(websocket)

    try:
        # Отправляем текущее состояние очереди
        queue = handoff.get_queue()
        await websocket.send_json({
            "type": "queue_update",
            "data": {
                "queue": queue,
                "count": len(queue)
            }
        })

        # Держим соединение открытым
        while True:
            await websocket.receive_text()

    except WebSocketDisconnect:
        notifier.disconnect(websocket)


@app.post("/api/assign/{session_id}")
async def assign_session(session_id: str, data: dict = Body(None)):
    """Назначить сессию оператору"""

    # Получаем данные из body
    agent_id = "agent_default"
    if data:
        agent_id = data.get('agent_id', agent_id)

    success = handoff.assign_agent(session_id, agent_id, f"Agent {agent_id[:8]}")

    if success:
        return {"status": "assigned", "session_id": session_id, "agent_id": agent_id}
    else:
        return JSONResponse(
            {"error": "Session not found"},
            status_code=404
        )


@app.get("/api/stats")
async def get_stats():
    """Получить общую статистику"""
    queue = handoff.get_queue()
    agents = handoff.get_online_agents()

    # Подсчёт по статусам
    waiting = sum(1 for s in queue if s.get('status') == 'waiting')
    assigned = sum(1 for s in queue if s.get('status') == 'assigned')

    return {
        "queue": {
            "total": len(queue),
            "waiting": waiting,
            "assigned": assigned
        },
        "agents": {
            "online": len(agents)
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "ok",
        "service": "dashboard"
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )