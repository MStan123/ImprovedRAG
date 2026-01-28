"""
FastAPI приложение для RAG системы Birmarket с встроенным UI и сохранением истории чата
"""
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from useful_func import detect_lang
from support_handoff import handoff
from datetime import datetime
from contextlib import asynccontextmanager

# Импорты из вашей RAG системы
from rag_pipeline import answer_query
from stats import stats, print_cost_report
from logger_setup import setup_logger
from chat_history_manager import chat_history  # ← Убедись, что этот импорт есть

# Настройка логгера
logger = setup_logger()


# ============================================================
# MODELS (Pydantic схемы)
# ============================================================

from pydantic import BaseModel
from typing import Optional, List, Dict, Any

class QueryRequest(BaseModel):
    query: str
    include_sources: bool = True
    include_metadata: bool = False
    user_id: Optional[str] = None


class Source(BaseModel):
    file: str
    preview: str
    chunk_id: Optional[str] = None


class QueryResponse(BaseModel):
    """Ответ RAG системы"""
    answer: str
    query: str
    sources: Optional[List[Source]] = None  # ← Изменить на List[Source]
    selected_files: Optional[List[str]] = None
    feedback_id: Optional[str] = None
    metadata: Optional[dict] = None
    from_cache: bool = False
    processing_time: float = 0.0
    timestamp: str = ""

# Модель для отправки feedback
class FeedbackRequest(BaseModel):
    feedback_id: str
    rating: str  # "yes" или "no"


class FeedbackResponse(BaseModel):
    status: str
    message: str
    feedback_id: str
    rating: str
    handoff_session_id: Optional[str] = None

class HistoryMessage(BaseModel):
    """Сообщение из истории"""
    role: str
    content: str
    timestamp: str
    from_cache: Optional[bool] = None


class HistoryResponse(BaseModel):
    """Ответ с историей чата"""
    messages: List[HistoryMessage]


class HealthResponse(BaseModel):
    status: str
    uptime: float
    total_queries: int
    cache_hit_rate: float
    llm_calls: int
    cache_hits: int


class StatsResponse(BaseModel):
    total_queries: int
    llm_calls: int
    cache_hits: int
    cache_hit_rate: float
    spent_tokens: int
    saved_tokens: int
    total_tokens: int
    savings_percent: float


# ============================================================
# HTML TEMPLATE с загрузкой истории
# ============================================================

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="az">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Birmarket AI Assistant</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 20px;
        }

        .container {
            width: 100%;
            max-width: 1200px;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
            display: flex;
            flex-direction: column;
            height: 90vh;
        }

        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 25px 30px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .header h1 {
            font-size: 24px;
            font-weight: 600;
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .stats-badge {
            background: rgba(255,255,255,0.2);
            padding: 8px 15px;
            border-radius: 20px;
            font-size: 13px;
            display: flex;
            gap: 15px;
        }

        .stats-item {
            display: flex;
            align-items: center;
            gap: 5px;
        }

        .chat-container {
            flex: 1;
            overflow-y: auto;
            padding: 30px;
            background: #f7f9fc;
        }

        .message {
            margin-bottom: 20px;
            display: flex;
            gap: 15px;
            animation: fadeIn 0.3s ease-in;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .message.user {
            flex-direction: row-reverse;
        }

        .message.system .message-content {
            background: #fff3cd;
            color: #856404;
            border-left: 4px solid #ffc107;
            max-width: 90%;
            text-align: center;
            font-size: 13px;
            font-style: italic;
        }

        .message-avatar {
            width: 40px;
            height: 40px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 20px;
            flex-shrink: 0;
        }

        .user .message-avatar {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }

        .assistant .message-avatar {
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        }

        .message-content {
            max-width: 70%;
            background: white;
            padding: 15px 20px;
            border-radius: 15px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }

        .user .message-content {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }

        .message-text {
            line-height: 1.6;
            white-space: pre-wrap;
        }

        .message-meta {
            margin-top: 10px;
            font-size: 12px;
            opacity: 0.7;
            display: flex;
            gap: 15px;
        }

        .sources {
            margin-top: 15px;
            padding: 15px;
            background: #f7f9fc;
            border-radius: 10px;
            border-left: 4px solid #667eea;
        }

        .sources-title {
            font-weight: 600;
            margin-bottom: 10px;
            color: #667eea;
            font-size: 13px;
        }

        .source-item {
            background: white;
            padding: 10px;
            margin-bottom: 8px;
            border-radius: 8px;
            font-size: 12px;
            border: 1px solid #e0e0e0;
        }

        .source-file {
            font-weight: 600;
            color: #764ba2;
            margin-bottom: 5px;
        }

        .source-preview {
            color: #666;
            line-height: 1.4;
        }

        /* НОВОЕ: Стили для feedback кнопок */
        .feedback-container {
            margin-top: 15px;
            padding-top: 15px;
            border-top: 1px solid #e9ecef;
            display: flex;
            align-items: center;
            gap: 12px;
        }

        .feedback-question {
            color: #666;
            font-weight: 500;
            font-size: 13px;
        }

        .feedback-buttons {
            display: flex;
            gap: 8px;
        }

        .feedback-btn {
            padding: 6px 14px;
            border: 1.5px solid #ddd;
            border-radius: 8px;
            background: white;
            cursor: pointer;
            font-size: 13px;
            font-weight: 500;
            transition: all 0.2s;
            display: flex;
            align-items: center;
            gap: 5px;
        }

        .feedback-btn:hover:not(:disabled) {
            transform: translateY(-1px);
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }

        .feedback-btn.yes:hover:not(:disabled) {
            background: #e8f5e9;
            border-color: #4caf50;
            color: #2e7d32;
        }

        .feedback-btn.no:hover:not(:disabled) {
            background: #ffebee;
            border-color: #f44336;
            color: #c62828;
        }

        .feedback-btn:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }

        .feedback-submitted {
            margin-top: 15px;
            padding: 10px 15px;
            border-radius: 8px;
            font-size: 13px;
            font-weight: 500;
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .feedback-submitted.positive {
            background: #e8f5e9;
            color: #2e7d32;
        }

        .feedback-submitted.negative {
            background: #fff3e0;
            color: #e65100;
        }

        .input-container {
            padding: 20px 30px;
            background: white;
            border-top: 1px solid #e0e0e0;
        }

        .input-wrapper {
            display: flex;
            gap: 10px;
            align-items: center;
        }

        #queryInput {
            flex: 1;
            padding: 15px 20px;
            border: 2px solid #e0e0e0;
            border-radius: 25px;
            font-size: 15px;
            outline: none;
            transition: all 0.3s;
        }

        #queryInput:focus {
            border-color: #667eea;
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
        }

        #sendButton {
            padding: 15px 30px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 25px;
            font-size: 15px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s;
            display: flex;
            align-items: center;
            gap: 8px;
        }

        #sendButton:hover:not(:disabled) {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
        }

        #sendButton:disabled {
            opacity: 0.6;
            cursor: not-allowed;
        }

        .loading {
            display: flex;
            gap: 5px;
            padding: 20px;
        }

        .loading-dot {
            width: 10px;
            height: 10px;
            background: #667eea;
            border-radius: 50%;
            animation: bounce 1.4s infinite ease-in-out;
        }

        .loading-dot:nth-child(1) { animation-delay: -0.32s; }
        .loading-dot:nth-child(2) { animation-delay: -0.16s; }

        @keyframes bounce {
            0%, 80%, 100% { transform: scale(0); }
            40% { transform: scale(1); }
        }

        .empty-state {
            text-align: center;
            padding: 60px 20px;
            color: #999;
        }

        .empty-state-icon {
            font-size: 64px;
            margin-bottom: 20px;
        }

        .empty-state-text {
            font-size: 18px;
            margin-bottom: 10px;
        }

        .empty-state-subtext {
            font-size: 14px;
            opacity: 0.7;
        }

        .example-queries {
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
            justify-content: center;
            margin-top: 20px;
        }

        .example-query {
            background: white;
            padding: 10px 20px;
            border-radius: 20px;
            border: 2px solid #e0e0e0;
            cursor: pointer;
            transition: all 0.3s;
            font-size: 13px;
        }

        .example-query:hover {
            border-color: #667eea;
            background: #667eea;
            color: white;
            transform: translateY(-2px);
        }

        @media (max-width: 768px) {
            .container {
                height: 100vh;
                border-radius: 0;
            }

            .message-content {
                max-width: 85%;
            }

            .stats-badge {
                display: none;
            }

            .feedback-container {
                flex-direction: column;
                align-items: flex-start;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>
                <span>🤖</span>
                Birmarket AI Assistant
            </h1>
            <div class="stats-badge" id="statsBar">
                <div class="stats-item">
                    <span>💬</span>
                    <span id="totalQueries">0</span>
                </div>
                <div class="stats-item">
                    <span>💾</span>
                    <span id="cacheRate">0%</span>
                </div>
            </div>
        </div>

        <div class="chat-container" id="chatContainer">
            <div class="empty-state">
                <div class="empty-state-icon">👋</div>
                <div class="empty-state-text">Salam! Necə kömək edə bilərəm?</div>
                <div class="empty-state-subtext">Sualınızı yazın və ya nümunə suallardan birini seçin</div>
                <div class="example-queries">
                    <div class="example-query" onclick="sendExampleQuery('Birmarket nədir?')">
                        Birmarket nədir?
                    </div>
                    <div class="example-query" onclick="sendExampleQuery('Hansı ödəniş üsulları mövcuddur?')">
                        Hansı ödəniş üsulları mövcuddur?
                    </div>
                    <div class="example-query" onclick="sendExampleQuery('BirBonus nə deməkdir?')">
                        BirBonus nə deməkdir?
                    </div>
                    <div class="example-query" onclick="sendExampleQuery('Çatdırılma haqqında məlumat ver')">
                        Çatdırılma haqqında məlumat ver
                    </div>
                </div>
            </div>
        </div>

        <div class="input-container">
            <div class="input-wrapper">
                <input 
                    type="text" 
                    id="queryInput" 
                    placeholder="Sualınızı yazın..."
                    onkeypress="handleKeyPress(event)"
                />
                <button id="sendButton" onclick="sendQuery()">
                    <span>📤</span>
                    Göndər
                </button>
            </div>
        </div>
    </div>

    <script>
        let isProcessing = false;

        async function sendQuery() {
            const input = document.getElementById('queryInput');
            const query = input.value.trim();

            if (!query || isProcessing) return;

            isProcessing = true;
            const sendButton = document.getElementById('sendButton');
            sendButton.disabled = true;
            sendButton.innerHTML = '<span>⏳</span> Gözləyin...';

            // Remove empty state
            const emptyState = document.querySelector('.empty-state');
            if (emptyState) emptyState.remove();

            // Add user message
            addMessage(query, 'user');
            input.value = '';

            // Add loading indicator
            const loadingId = 'loading-' + Date.now();
            addLoadingMessage(loadingId);

            try {
                const response = await fetch('/query', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        query: query,
                        include_sources: true,
                        include_metadata: false
                    })
                });

                if (!response.ok) {
                    throw new Error('Network response was not ok');
                }

                const data = await response.json();

                // Remove loading
                document.getElementById(loadingId)?.remove();

                // Add assistant message with feedback buttons
                addMessage(data.answer, 'assistant', {
                    sources: data.sources,
                    fromCache: data.from_cache,
                    processingTime: data.processing_time,
                    feedbackId: data.feedback_id  // НОВОЕ: получаем feedback_id
                });

                // Update stats
                updateStats();

            } catch (error) {
                document.getElementById(loadingId)?.remove();
                addMessage('Üzr istəyirik, xəta baş verdi. Zəhmət olmasa yenidən cəhd edin.', 'assistant', {
                    error: true
                });
                console.error('Error:', error);
            } finally {
                isProcessing = false;
                sendButton.disabled = false;
                sendButton.innerHTML = '<span>📤</span> Göndər';
            }
        }

        function addMessage(text, role, metadata = {}) {
            const container = document.getElementById('chatContainer');
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${role}`;

            const avatar = role === 'user' ? '👤' : (role === 'system' ? '🔔' : '🤖');

            let metaHTML = '';
            if (metadata.processingTime !== undefined && role !== 'system') {
                const cacheIcon = metadata.fromCache ? '💾' : '🔄';
                metaHTML = `
                    <div class="message-meta">
                        <span>${cacheIcon} ${metadata.fromCache ? 'Cache' : 'LLM'}</span>
                        <span>⏱️ ${metadata.processingTime.toFixed(2)}s</span>
                    </div>
                `;
            }

            let sourcesHTML = '';
            if (metadata.sources && metadata.sources.length > 0 && role !== 'system') {
                sourcesHTML = `
                    <div class="sources">
                        <div class="sources-title">📚 Mənbələr (${metadata.sources.length})</div>
                        ${metadata.sources.slice(0, 3).map(source => `
                            <div class="source-item">
                                <div class="source-file">${source.file}</div>
                                <div class="source-preview">${source.preview.substring(0, 150)}...</div>
                            </div>
                        `).join('')}
                    </div>
                `;
            }

            // Feedback кнопки (только для ответов AI, не для системных сообщений и ошибок)
            let feedbackHTML = '';
            if (role === 'assistant' && metadata.feedbackId && !metadata.error) {
                feedbackHTML = `
                    <div class="feedback-container" id="feedback-${metadata.feedbackId}">
                        <span class="feedback-question">Cavab kömək etdi?</span>
                        <div class="feedback-buttons">
                            <button class="feedback-btn yes" onclick="submitFeedback('${metadata.feedbackId}', 'yes')">
                                👍 Bəli
                            </button>
                            <button class="feedback-btn no" onclick="submitFeedback('${metadata.feedbackId}', 'no')">
                                👎 Xeyr
                            </button>
                        </div>
                    </div>
                `;
            }

            messageDiv.innerHTML = `
                <div class="message-avatar">${avatar}</div>
                <div class="message-content">
                    <div class="message-text">${text}</div>
                    ${metaHTML}
                    ${sourcesHTML}
                    ${feedbackHTML}
                </div>
            `;

            container.appendChild(messageDiv);
            container.scrollTop = container.scrollHeight;
        }

        // НОВОЕ: Функция отправки feedback с автоматическим handoff
        async function submitFeedback(feedbackId, rating) {
            const feedbackContainer = document.getElementById(`feedback-${feedbackId}`);
            if (!feedbackContainer) return;

            // Отключаем кнопки
            const buttons = feedbackContainer.querySelectorAll('.feedback-btn');
            buttons.forEach(btn => btn.disabled = true);

            try {
                const response = await fetch('/api/feedback/submit', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        feedback_id: feedbackId,
                        rating: rating
                    })
                });

                if (!response.ok) {
                    throw new Error('Failed to submit feedback');
                }

                const data = await response.json();

                // НОВОЕ: Если есть handoff_session_id - открываем чат с оператором
                if (data.handoff_session_id && rating === 'no') {
                    const chatUrl = `http://localhost:8001/chat?session=${data.handoff_session_id}`;

                    // Показываем сообщение о переводе
                    feedbackContainer.innerHTML = `
                        <div class="feedback-submitted negative">
                            ${data.message}
                        </div>
                    `;

                    // Автоматически открываем чат через 2 секунды
                    setTimeout(() => {
                        window.open(chatUrl, '_blank');

                        // Добавляем системное сообщение в текущий чат
                        addMessage(
                            'Вы были переведены на оператора поддержки. Чат открылся в новом окне.',
                            'system'
                        );
                    }, 2000);

                } else {
                    // Обычная благодарность для положительного feedback
                    feedbackContainer.innerHTML = `
                        <div class="feedback-submitted positive">
                            ${data.message}
                        </div>
                    `;
                }

            } catch (error) {
                console.error('Error submitting feedback:', error);

                // Показываем ошибку
                feedbackContainer.innerHTML = `
                    <div class="feedback-submitted" style="background: #ffebee; color: #c62828;">
                        ⚠️ Rəy göndərilmədi. Yenidən cəhd edin.
                    </div>
                `;

                // Возвращаем кнопки через 2 секунды
                setTimeout(() => {
                    buttons.forEach(btn => btn.disabled = false);
                }, 2000);
            }
        }

        function addLoadingMessage(id) {
            const container = document.getElementById('chatContainer');
            const messageDiv = document.createElement('div');
            messageDiv.id = id;
            messageDiv.className = 'message assistant';
            messageDiv.innerHTML = `
                <div class="message-avatar">🤖</div>
                <div class="message-content">
                    <div class="loading">
                        <div class="loading-dot"></div>
                        <div class="loading-dot"></div>
                        <div class="loading-dot"></div>
                    </div>
                </div>
            `;
            container.appendChild(messageDiv);
            container.scrollTop = container.scrollHeight;
        }

        async function updateStats() {
            try {
                const response = await fetch('/stats');
                const data = await response.json();

                document.getElementById('totalQueries').textContent = data.total_queries;
                document.getElementById('cacheRate').textContent = data.cache_hit_rate.toFixed(0) + '%';
            } catch (error) {
                console.error('Error updating stats:', error);
            }
        }

        function handleKeyPress(event) {
            if (event.key === 'Enter' && !event.shiftKey) {
                event.preventDefault();
                sendQuery();
            }
        }

        function sendExampleQuery(query) {
            document.getElementById('queryInput').value = query;
            sendQuery();
        }

        // Initial stats update
        updateStats();

        // Update stats every 30 seconds
        setInterval(updateStats, 30000);
    </script>
</body>
</html>
"""
# ============================================================
# LIFESPAN & APP
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Starting Birmarket RAG API + UI...")
    app.state.start_time = datetime.now()
    yield
    logger.info("🛑 Shutting down...")
    print_cost_report()

app = FastAPI(
    title="Birmarket RAG API",
    description="RAG система с UI и сохранением истории чата",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])


# ============================================================
# ENDPOINTS
# ============================================================

@app.get("/", response_class=HTMLResponse)
async def root():
    return HTML_TEMPLATE


@app.get("/health", response_model=HealthResponse)
async def health_check():
    uptime = (datetime.now() - app.state.start_time).total_seconds()
    total = stats.llm_calls + stats.cache_hits
    cache_rate = (stats.cache_hits / total * 100) if total > 0 else 0
    return HealthResponse(
        status="healthy",
        uptime=uptime,
        total_queries=total,
        cache_hit_rate=cache_rate,
        llm_calls=stats.llm_calls,
        cache_hits=stats.cache_hits
    )


@app.get("/api/history", response_model=HistoryResponse)
async def get_history(user_id: str = Query(..., description="User ID из localStorage")):
    """Получить историю чата для пользователя"""
    full_history = chat_history.get_history(user_id)

    messages = []
    for msg_dict in full_history:  # get_history возвращает список dict'ов
        if msg_dict.get("role") in ["user", "assistant"]:
            messages.append(HistoryMessage(
                role=msg_dict["role"],
                content=msg_dict.get("content", ""),
                timestamp=msg_dict.get("timestamp", datetime.now().isoformat()),
                from_cache=msg_dict.get("metadata", {}).get("from_cache")
            ))

    return HistoryResponse(messages=messages)


@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """
    Обработка запроса пользователя с возвратом feedback_id
    """
    start_time = datetime.now()

    try:
        user_id = request.user_id or "anonymous_guest"
        logger.info(f"📥 Query from {user_id}: {request.query[:100]}...")

        initial_cache = stats.cache_hits

        # ВАЖНО: теперь answer_query возвращает 4 элемента вместо 3
        response_text, docs, selected_files, feedback_id = await answer_query(
            query=request.query,
            user_id=user_id
        )

        from_cache = stats.cache_hits > initial_cache

        # Формируем sources
        sources = None
        if request.include_sources and docs:
            sources = [
                Source(
                    file=doc.metadata.get("source", "Unknown"),
                    preview=doc.page_content[:200] + ("..." if len(doc.page_content) > 200 else ""),
                    chunk_id=doc.metadata.get("chunk_id")
                )
                for doc in docs[:5]
            ]

        # Формируем metadata
        metadata = None
        if request.include_metadata and docs:
            metadata = {
                "num_sources": len(docs),
                "selected_files_count": len(selected_files)
            }

        processing_time = (datetime.now() - start_time).total_seconds()

        return QueryResponse(
            answer=response_text,
            query=request.query,
            sources=sources,
            selected_files=selected_files if request.include_sources else None,
            metadata=metadata,
            from_cache=from_cache,
            processing_time=processing_time,
            timestamp=datetime.now().isoformat(),
            feedback_id=feedback_id  # НОВОЕ: добавляем feedback_id
        )

    except Exception as e:
        logger.error(f"❌ Query error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal error")


@app.get("/stats", response_model=StatsResponse)
async def get_stats():
    total = stats.llm_calls + stats.cache_hits
    potential = stats.spent_tokens + stats.saved_tokens
    cache_rate = (stats.cache_hits / total * 100) if total > 0 else 0
    savings = (stats.saved_tokens / potential * 100) if potential > 0 else 0
    return StatsResponse(
        total_queries=total,
        llm_calls=stats.llm_calls,
        cache_hits=stats.cache_hits,
        cache_hit_rate=cache_rate,
        spent_tokens=stats.spent_tokens,
        saved_tokens=stats.saved_tokens,
        total_tokens=potential,
        savings_percent=savings
    )

@app.get("/api/stats")
async def api_stats():
    # Просто возвращаем то же самое, что и /stats
    return await get_stats()

@app.post("/stats/reset")
async def reset_stats():
    stats.llm_calls = stats.cache_hits = stats.spent_tokens = stats.saved_tokens = 0
    logger.info("📊 Stats reset")
    return {"message": "Stats reset"}


from fastapi import HTTPException
from feedback_manager import feedback_manager


@app.post("/api/feedback/submit", response_model=FeedbackResponse)
async def submit_feedback(request: FeedbackRequest):
    """
    Endpoint для сохранения оценки пользователя

    ВАЖНО: Если rating = "no", автоматически создаётся handoff к оператору
    """
    logger.info(f"📊 Feedback received: {request.feedback_id} -> {request.rating}")

    # Проверяем валидность rating
    if request.rating not in ["yes", "no"]:
        raise HTTPException(
            status_code=400,
            detail="Rating must be 'yes' or 'no'"
        )

    # Получаем данные pending feedback ДО сохранения
    # (чтобы использовать для handoff)
    pending = feedback_manager.pending_feedback.get(request.feedback_id)

    if not pending:
        logger.warning(f"⚠️ Feedback ID not found: {request.feedback_id}")
        raise HTTPException(
            status_code=404,
            detail=f"Feedback ID {request.feedback_id} not found or already submitted"
        )

    # Сохраняем feedback в базу
    success = feedback_manager.submit_feedback(
        feedback_id=request.feedback_id,
        rating=request.rating
    )

    if not success:
        raise HTTPException(status_code=500, detail="Failed to save feedback")

    # НОВОЕ: Если ответ НЕ помог - создаём handoff
    handoff_session_id = None
    if request.rating == "no":
        logger.warning(f"🔴 Negative feedback - triggering handoff for {request.feedback_id}")

        # Создаём сессию handoff
        handoff_session_id = handoff.create_session(
            query=pending["original_query"],
            context=pending["ai_response"],
            user_id=pending.get("user_id"),
            user_phone=None,  # Можно запросить у пользователя
            user_name=None,
            metadata={
                "feedback_id": request.feedback_id,
                "reason": "negative_feedback",
                "contextualized_query": pending.get("contextualized_query"),
                "selected_files": pending.get("selected_files", [])
            }
        )

        logger.info(f"✅ Handoff session created: {handoff_session_id}")

    # Сообщения на разных языках
    messages = {
        "yes": {
            "az": "✅ Təşəkkürlər! Kömək edə bildiyimizə şadıq.",
            "ru": "✅ Спасибо! Рады, что смогли помочь.",
            "en": "✅ Thank you! Glad we could help."
        },
        "no": {
            "az": "✅ Rəyiniz üçün təşəkkür edirik. Dəstək komandamız sizinlə əlaqə saxlayacaq.",
            "ru": "✅ Спасибо за отзыв. Команда поддержки свяжется с вами.",
            "en": "✅ Thank you for your feedback. Our support team will contact you."
        }
    }

    # Определяем язык из оригинального запроса
    try:
        lang = detect_lang(pending["original_query"])
        if lang not in ["az", "ru"]:
            lang = "az"  # fallback
    except:
        lang = "az"

    response_message = messages[request.rating][lang]

    # Если создан handoff - добавляем информацию о чате
    if handoff_session_id:
        chat_url = f"http://localhost:8001/chat?session={handoff_session_id}"

        handoff_messages = {
            'az': f"\n\n📞 Sizi dəstək mütəxəssisi ilə əlaqələndirirəm...\n🎫 Müraciət nömrəsi: #{handoff_session_id[:8].upper()}\n⏱️ Orta gözləmə vaxtı: ~2-3 dəqiqə\n\n🔗 Çat linki:\n{chat_url}",
            'ru': f"\n\n📞 Соединяю вас со специалистом поддержки...\n🎫 Номер обращения: #{handoff_session_id[:8].upper()}\n⏱️ Среднее время ожидания: ~2-3 минуты\n\n🔗 Ссылка на чат:\n{chat_url}",
            'en': f"\n\n📞 Connecting you with a support specialist...\n🎫 Ticket number: #{handoff_session_id[:8].upper()}\n⏱️ Average wait time: ~2-3 minutes\n\n🔗 Chat link:\n{chat_url}"
        }

        response_message += handoff_messages.get(lang, handoff_messages['en'])

    logger.info(f"✅ Feedback saved: {request.feedback_id} -> {request.rating}")

    return FeedbackResponse(
        status="success",
        message=response_message,
        feedback_id=request.feedback_id,
        rating=request.rating,
        handoff_session_id=handoff_session_id  # НОВОЕ: возвращаем ID сессии
    )


# Опциональный endpoint для аналитики (только для админов)
@app.get("/api/feedback/analytics")
async def get_feedback_analytics(
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        category: Optional[str] = None
):
    """
    Получить аналитику по feedback

    Query params:
        - start_date: ISO datetime (опционально)
        - end_date: ISO datetime (опционально)
        - category: категория вопросов (опционально)

    Returns:
        {
            "total": 150,
            "yes_count": 120,
            "no_count": 30,
            "yes_percentage": 80.0,
            "no_percentage": 20.0,
            "by_category": {...},
            "by_date": {...},
            "by_kb_version": {...}
        }
    """
    from datetime import datetime

    logger.info(f"📈 Analytics requested: {start_date} to {end_date}, category: {category}")

    start = datetime.fromisoformat(start_date) if start_date else None
    end = datetime.fromisoformat(end_date) if end_date else None

    analytics = feedback_manager.get_analytics(
        start_date=start,
        end_date=end,
        category=category
    )

    return {
        "status": "success",
        "data": analytics
    }


# Health check для feedback системы
@app.get("/api/feedback/health")
async def feedback_health_check():
    """Проверка работоспособности системы обратной связи"""
    pending_count = len(feedback_manager.pending_feedback)

    return {
        "status": "healthy",
        "pending_feedback_count": pending_count,
        "kb_version": feedback_manager.kb_version,
        "storage_path": str(feedback_manager.storage_path)
    }


if __name__ == "__main__":
    import uvicorn
    print("🚀 Birmarket RAG API + UI запущен: http://localhost:8000")
    uvicorn.run("app1:app", host="0.0.0.0", port=8000, log_level="info")
