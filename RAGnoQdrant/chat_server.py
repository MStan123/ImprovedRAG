"""
WebSocket сервер для real-time чата между пользователями и операторами
"""
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from typing import Dict
import json
import asyncio
from datetime import datetime
from pathlib import Path
from support_handoff import handoff

app = FastAPI(title="Birmarket Chat Server")

# CORS для разных доменов
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Создаём папку templates если её нет
Path("templates").mkdir(exist_ok=True)

# Templates
templates = Jinja2Templates(directory="templates")


class ConnectionManager:
    """Управление WebSocket подключениями"""

    def __init__(self):
        # {session_id: {'user': websocket, 'agent': websocket}}
        self.active_connections: Dict[str, Dict[str, WebSocket]] = {}
        # {websocket: session_id} для обратного поиска
        self.ws_to_session: Dict[WebSocket, str] = {}

    async def connect_user(self, session_id: str, websocket: WebSocket):
        """Подключает пользователя к сессии"""
        await websocket.accept()

        if session_id not in self.active_connections:
            self.active_connections[session_id] = {}

        self.active_connections[session_id]['user'] = websocket
        self.ws_to_session[websocket] = session_id

        # Обновляем статус если оператор уже подключен
        if 'agent' in self.active_connections[session_id]:
            handoff.activate_session(session_id)

        print(f"👤 User connected to session {session_id[:8]}")

    async def connect_agent(self, session_id: str, websocket: WebSocket, agent_id: str, agent_name: str):
        """Подключает оператора к сессии"""
        await websocket.accept()

        if session_id not in self.active_connections:
            self.active_connections[session_id] = {}

        self.active_connections[session_id]['agent'] = websocket
        self.ws_to_session[websocket] = session_id

        # Назначаем оператора в Redis
        handoff.assign_agent(session_id, agent_id, agent_name)

        # Если пользователь уже подключен - активируем сессию
        if 'user' in self.active_connections[session_id]:
            handoff.activate_session(session_id)

            # Отправляем пользователю уведомление
            await self.send_to_user(session_id, {
                "type": "agent_joined",
                "message": f"Оператор {agent_name} подключился к чату",
                "agent_id": agent_id,
                "agent_name": agent_name,
                "timestamp": datetime.now().isoformat()
            })

        # Отправляем оператору историю чата
        session = handoff.get_session(session_id)
        if session and session.get('messages'):
            await websocket.send_json({
                "type": "history",
                "messages": session['messages']
            })

        print(f"🎧 Agent {agent_name} ({agent_id}) connected to session {session_id[:8]}")

    def disconnect(self, websocket: WebSocket):
        """Отключает WebSocket"""
        if websocket not in self.ws_to_session:
            return

        session_id = self.ws_to_session[websocket]

        if session_id in self.active_connections:
            # Определяем роль
            role = None
            if self.active_connections[session_id].get('user') == websocket:
                role = 'user'
                del self.active_connections[session_id]['user']
            elif self.active_connections[session_id].get('agent') == websocket:
                role = 'agent'
                del self.active_connections[session_id]['agent']

            # Если сессия пустая - удаляем
            if not self.active_connections[session_id]:
                del self.active_connections[session_id]

            print(f"❌ {role} disconnected from session {session_id[:8]}")

        del self.ws_to_session[websocket]

    async def send_to_user(self, session_id: str, message: dict):
        """Отправляет сообщение пользователю"""
        if session_id in self.active_connections:
            if 'user' in self.active_connections[session_id]:
                try:
                    await self.active_connections[session_id]['user'].send_json(message)
                except:
                    pass

    async def send_to_agent(self, session_id: str, message: dict):
        """Отправляет сообщение оператору"""
        if session_id in self.active_connections:
            if 'agent' in self.active_connections[session_id]:
                try:
                    await self.active_connections[session_id]['agent'].send_json(message)
                except:
                    pass

    def is_agent_connected(self, session_id: str) -> bool:
        """Проверяет, подключен ли оператор"""
        return (session_id in self.active_connections and
                'agent' in self.active_connections[session_id])

    def is_user_connected(self, session_id: str) -> bool:
        """Проверяет, подключен ли пользователь"""
        return (session_id in self.active_connections and
                'user' in self.active_connections[session_id])


manager = ConnectionManager()


# ============================================================
# HTML ENDPOINTS
# ============================================================

@app.get("/", response_class=HTMLResponse)
async def root():
    """Главная страница"""
    return HTMLResponse("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Birmarket Chat Server</title>
        <style>
            body {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                max-width: 900px;
                margin: 50px auto;
                padding: 20px;
                background: #f5f7fa;
            }
            h1 { color: #667eea; font-size: 36px; }
            .status { 
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 20px;
                border-radius: 12px;
                margin: 20px 0;
            }
            .info { 
                background: white;
                padding: 25px;
                border-radius: 12px;
                margin: 20px 0;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            }
            code { 
                background: #f0f0f0;
                padding: 3px 8px;
                border-radius: 4px;
                font-family: 'Courier New', monospace;
                font-size: 13px;
            }
            ul { line-height: 1.8; }
            a { color: #667eea; text-decoration: none; font-weight: 600; }
            a:hover { text-decoration: underline; }
        </style>
    </head>
    <body>
        <h1>🎧 Birmarket Chat Server</h1>
        
        <div class="status">
            <h3 style="margin:0;">✅ Server Running</h3>
            <p style="margin:10px 0 0 0; opacity:0.9;">WebSocket server is operational</p>
        </div>
        
        <div class="info">
            <h3>📡 WebSocket Endpoints</h3>
            <ul>
                <li>User: <code>ws://localhost:8001/ws/chat/user/{session_id}</code></li>
                <li>Agent: <code>ws://localhost:8001/ws/chat/agent/{session_id}/{agent_id}</code></li>
            </ul>
        </div>
        
        <div class="info">
            <h3>💬 Open User Chat</h3>
            <p>To open a user chat interface:</p>
            <p><code>http://localhost:8001/chat?session=YOUR_SESSION_ID</code></p>
        </div>
        
        <div class="info">
            <h3>📊 Dashboard</h3>
            <p>Operator dashboard available at: <a href="http://localhost:8000" target="_blank">http://localhost:8000</a></p>
        </div>
        
        <div class="info">
            <h3>📚 API Documentation</h3>
            <p><a href="/docs">Interactive API Docs (Swagger)</a></p>
            <p><a href="/redoc">Alternative API Docs (ReDoc)</a></p>
        </div>
    </body>
    </html>
    """)


@app.get("/agent-chat/{session_id}", response_class=HTMLResponse)
async def agent_chat_page(request: Request, session_id: str, agent_id: str = None):
    """Страница чата для оператора"""
    if not agent_id:
        return HTMLResponse("""
        <html>
        <head><title>Error</title></head>
        <body style="font-family: Arial; padding: 50px; text-align: center;">
            <h1>❌ Error: Missing agent_id parameter</h1>
            <p>Usage: <code>/agent-chat/{session_id}?agent_id=YOUR_AGENT_ID</code></p>
        </body>
        </html>
        """, status_code=400)

    # Проверяем существование сессии
    session_data = handoff.get_session(session_id)
    if not session_data:
        return HTMLResponse(f"""
        <html>
        <head><title>Session Not Found</title></head>
        <body style="font-family: Arial; padding: 50px; text-align: center;">
            <h1>❌ Session Not Found</h1>
            <p>Session ID: <code>{session_id}</code></p>
            <p>This session may have expired or doesn't exist.</p>
        </body>
        </html>
        """, status_code=404)

    return templates.TemplateResponse("agent_chat.html", {
        "request": request,
        "session_id": session_id,
        "session": session_data,
        "agent_id": agent_id
    })

@app.get("/chat", response_class=HTMLResponse)
async def user_chat_page(request: Request, session: str = None):
    """Страница чата для пользователя"""
    if not session:
        return HTMLResponse("""
        <html>
        <head><title>Error</title></head>
        <body style="font-family: Arial; padding: 50px; text-align: center;">
            <h1>❌ Error: Missing session parameter</h1>
            <p>Usage: <code>/chat?session=YOUR_SESSION_ID</code></p>
        </body>
        </html>
        """, status_code=400)

    # Проверяем существование сессии
    session_data = handoff.get_session(session)
    if not session_data:
        return HTMLResponse(f"""
        <html>
        <head><title>Session Not Found</title></head>
        <body style="font-family: Arial; padding: 50px; text-align: center;">
            <h1>❌ Session Not Found</h1>
            <p>Session ID: <code>{session}</code></p>
            <p>This session may have expired or doesn't exist.</p>
        </body>
        </html>
        """, status_code=404)

    return templates.TemplateResponse("user_chat.html", {
        "request": request,
        "session_id": session
    })


# ============================================================
# WEBSOCKET ENDPOINTS
# ============================================================

@app.websocket("/ws/chat/user/{session_id}")
async def user_websocket(websocket: WebSocket, session_id: str):
    """WebSocket endpoint для пользователя"""

    # Проверяем существование сессии
    session = handoff.get_session(session_id)
    if not session:
        await websocket.close(code=4004, reason="Session not found")
        return

    await manager.connect_user(session_id, websocket)

    try:
        while True:
            # Получаем сообщение от пользователя
            data = await websocket.receive_text()
            message_data = json.loads(data)

            content = message_data.get('content', '').strip()
            if not content:
                continue

            # Сохраняем в Redis
            handoff.add_message(session_id, "user", content)

            # Отправляем оператору (если подключен)
            await manager.send_to_agent(session_id, {
                "type": "message",
                "role": "user",
                "content": content,
                "timestamp": datetime.now().isoformat()
            })

            # Если оператор не подключен - отправляем auto-reply
            if not manager.is_agent_connected(session_id):
                queue_position = handoff.get_queue_position(session_id)
                if queue_position:
                    await websocket.send_json({
                        "type": "system",
                        "content": f"Ваше сообщение получено. Позиция в очереди: {queue_position}",
                        "timestamp": datetime.now().isoformat()
                    })

    except WebSocketDisconnect:
        manager.disconnect(websocket)

        # Уведомляем оператора
        await manager.send_to_agent(session_id, {
            "type": "user_disconnected",
            "message": "Пользователь отключился",
            "timestamp": datetime.now().isoformat()
        })

    except Exception as e:
        print(f"Error in user websocket: {e}")
        manager.disconnect(websocket)


@app.websocket("/ws/chat/agent/{session_id}/{agent_id}")
async def agent_websocket(websocket: WebSocket, session_id: str, agent_id: str):
    """WebSocket endpoint для оператора"""

    # Проверяем существование сессии
    session = handoff.get_session(session_id)
    if not session:
        await websocket.close(code=4004, reason="Session not found")
        return

    # Получаем имя агента (можно из БД или передавать отдельно)
    agent_name = f"Agent {agent_id[:4]}"

    await manager.connect_agent(session_id, websocket, agent_id, agent_name)

    # Отмечаем агента онлайн
    handoff.mark_agent_online(agent_id, agent_name)

    try:
        while True:
            # Получаем сообщение от оператора
            data = await websocket.receive_text()
            message_data = json.loads(data)

            msg_type = message_data.get('type', 'message')

            if msg_type == 'message':
                content = message_data.get('content', '').strip()
                if not content:
                    continue

                # Сохраняем в Redis
                handoff.add_message(session_id, "agent", content, {
                    "agent_id": agent_id,
                    "agent_name": agent_name
                })

                # НОВОЕ: Сохраняем в историю чата пользователя
                session = handoff.get_session(session_id)
                if session:
                    user_id = session.get("user_id")
                    if user_id:
                        from chat_history_manager import chat_history
                        chat_history.add_message(
                            user_id,
                            "agent",
                            content,
                            metadata={"agent_name": agent_name}
                        )

                # Отправляем пользователю
                await manager.send_to_user(session_id, {
                    "type": "message",
                    "role": "agent",
                    "content": content,
                    "agent_name": agent_name,
                    "timestamp": datetime.now().isoformat()
                })

            elif msg_type == 'typing':
                # Индикатор печати
                await manager.send_to_user(session_id, {
                    "type": "typing",
                    "agent_name": agent_name
                })

            elif msg_type == 'close':
                # Закрытие чата
                resolution = message_data.get('resolution', 'resolved')
                handoff.close_session(session_id, resolution)

                await manager.send_to_user(session_id, {
                    "type": "chat_closed",
                    "message": "Чат завершён оператором",
                    "timestamp": datetime.now().isoformat()
                })

                break

    except WebSocketDisconnect:
        manager.disconnect(websocket)
        handoff.mark_agent_offline(agent_id)

        # Уведомляем пользователя
        await manager.send_to_user(session_id, {
            "type": "agent_disconnected",
            "message": f"Оператор {agent_name} отключился",
            "timestamp": datetime.now().isoformat()
        })

    except Exception as e:
        print(f"Error in agent websocket: {e}")
        manager.disconnect(websocket)
        handoff.mark_agent_offline(agent_id)


# ============================================================
# REST API ENDPOINTS
# ============================================================

@app.get("/api/session/{session_id}")
async def get_session_info(session_id: str):
    """Получить информацию о сессии"""
    session = handoff.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session


@app.get("/api/session/{session_id}/history")
async def get_chat_history(session_id: str):
    """Получить историю чата"""
    session = handoff.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    return {
        "session_id": session_id,
        "messages": session.get('messages', [])
    }


@app.get("/api/history/{user_id}")
async def get_user_history(user_id: str, limit: int = 20):
    """Получить историю чата пользователя"""
    from chat_history_manager import chat_history

    history = chat_history.get_history(user_id, limit=limit)
    stats = chat_history.get_conversation_stats(user_id)

    return {
        "user_id": user_id,
        "history": [msg.to_dict() for msg in history],
        "stats": stats
    }


@app.delete("/api/history/{user_id}")
async def clear_user_history(user_id: str):
    """Очистить историю пользователя"""
    from chat_history_manager import chat_history
    chat_history.clear_history(user_id)
    return {"message": "History cleared", "user_id": user_id}

@app.get("/api/queue")
async def get_support_queue():
    """Получить очередь ожидающих"""
    queue = handoff.get_queue()
    return {
        "count": len(queue),
        "queue": queue
    }


@app.get("/api/agents/online")
async def get_online_agents():
    """Получить список онлайн операторов"""
    agents = handoff.get_online_agents()
    return {
        "count": len(agents),
        "agents": agents
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "ok",
        "service": "chat_server",
        "active_connections": len(manager.active_connections)
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8001,
        log_level="info"
    )