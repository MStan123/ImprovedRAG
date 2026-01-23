import redis
import json
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
import os
import hashlib


@dataclass
class ChatMessage:
    """Структура сообщения в истории"""
    role: str
    content: str
    timestamp: str
    token_count: Optional[int] = None
    metadata: Optional[Dict] = None
    is_summarized: bool = False

    def to_dict(self):
        return asdict(self)

    @staticmethod
    def from_dict(data: Dict) -> 'ChatMessage':
        return ChatMessage(**data)


class ChatHistoryManager:
    """
    Advanced Chat History Manager с концепциями из AWS sample:
    - Token-aware windowing
    - Automatic summarization
    - Efficient storage
    """

    def __init__(
            self,
            redis_client=None,
            max_history_messages: int = 50,  # Максимум сообщений в хранилище
            max_context_tokens: int = 4000,  # Максимум токенов для LLM context
            summary_threshold: int = 20,  # После скольких сообщений суммаризировать
            ttl_hours: int = 72  # TTL истории (3 дня)
    ):
        self.redis = redis_client or redis.Redis(
            host='localhost',
            port=6379,
            db=2,
            decode_responses=True
        )
        self.max_history_messages = max_history_messages
        self.max_context_tokens = max_context_tokens
        self.summary_threshold = summary_threshold
        self.ttl_seconds = ttl_hours * 3600

        # Prefixes для разных типов данных
        self.history_prefix = "chat_history:"
        self.summary_prefix = "chat_summary:"
        self.metadata_prefix = "chat_metadata:"

    def estimate_tokens(self, text: str) -> int:
        """
        Примерная оценка токенов (правило большого пальца: 1 токен ≈ 4 символа)
        Для точности можно использовать tiktoken
        """
        return len(text) // 4

    def add_message(
            self,
            user_id: str,
            role: str,
            content: str,
            metadata: Optional[Dict] = None
    ) -> None:
        """
        Добавляет сообщение с учётом токенов
        """
        token_count = self.estimate_tokens(content)

        message = ChatMessage(
            role=role,
            content=content,
            timestamp=datetime.now().isoformat(),
            token_count=token_count,
            metadata=metadata or {},
            is_summarized=False
        )

        key = f"{self.history_prefix}{user_id}"

        # Получаем текущую историю
        history_json = self.redis.get(key)
        history = json.loads(history_json) if history_json else []

        # Добавляем новое сообщение
        history.append(message.to_dict())

        # Проверяем нужна ли суммаризация

        # Ограничиваем длину
        if len(history) > self.max_history_messages:
            history = history[-self.max_history_messages:]

        # Сохраняем
        self.redis.setex(
            key,
            self.ttl_seconds,
            json.dumps(history, ensure_ascii=False)
        )

        # Обновляем метаданные
        self._update_metadata(user_id)

    def get_optimized_context_for_llm(
            self,
            user_id: str,
            current_query: str
    ) -> Tuple[List[Dict], int]:
        """
        AWS Pattern: Token-aware context window

        Returns:
            (messages, total_tokens)
        """
        # Получаем историю
        history = self.get_history(user_id)

        # Учитываем токены текущего запроса
        current_query_tokens = self.estimate_tokens(current_query)
        available_tokens = self.max_context_tokens - current_query_tokens - 500  # резерв для system prompt

        # Строим контекст с конца (самые свежие сообщения)
        selected_messages = []
        total_tokens = 0

        for msg in reversed(history):
            msg_tokens = msg.token_count or self.estimate_tokens(msg.content)

            # Проверяем влезет ли сообщение
            if total_tokens + msg_tokens > available_tokens:
                break

            selected_messages.insert(0, msg)
            total_tokens += msg_tokens

        # Преобразуем в формат LLM
        llm_messages = []
        for msg in selected_messages:
            role = msg.role
            if role == "agent":
                role = "assistant"

            llm_messages.append({
                "role": role,
                "content": msg.content
            })

        return llm_messages, total_tokens

    def get_history(self, user_id: str) -> List[ChatMessage]:
        """Получает полную историю"""
        key = f"{self.history_prefix}{user_id}"
        history_json = self.redis.get(key)

        if not history_json:
            return []

        history = json.loads(history_json)
        return [ChatMessage.from_dict(msg) for msg in history]

    def get_summary_for_agent(self, user_id: str, last_n: int = 10) -> str:
        """
        Получает сводку для оператора с учётом summarization
        Фильтрует только чат с ботом (без agent)
        """
        # Получаем историю, фильтруя agent
        full_history = self.get_history(user_id)
        bot_history = [msg for msg in full_history if msg.role != "agent"]

        # Проверяем есть ли summary
        summary_key = f"{self.summary_prefix}{user_id}"
        summary = self.redis.get(summary_key)

        # Получаем последние сообщения
        recent_messages = bot_history[-last_n:] if len(bot_history) > last_n else bot_history

        lines = [""]

        for msg in recent_messages:
            emoji = {"user": "👤", "assistant": "🤖", "system": "ℹ️"}.get(msg.role, "💬")
            time_str = msg.timestamp[11:16]
            content = msg.content[:10] + "..." if len(msg.content) > 100 else msg.content
            lines.append(f"{emoji} [{time_str}] {content}")

        return "\n".join(lines)

    def _update_metadata(self, user_id: str):
        """Обновляет метаданные разговора"""
        history = self.get_history(user_id)

        if not history:
            return

        metadata = {
            "total_messages": len(history),
            "total_tokens": sum(m.token_count or 0 for m in history),
            "user_messages": sum(1 for m in history if m.role == "user"),
            "bot_messages": sum(1 for m in history if m.role == "assistant"),
            "agent_messages": sum(1 for m in history if m.role == "agent"),
            "has_summary": any(m.is_summarized for m in history),
            "first_message_at": history[0].timestamp,
            "last_message_at": history[-1].timestamp,
            "last_updated": datetime.now().isoformat()
        }

        key = f"{self.metadata_prefix}{user_id}"
        self.redis.setex(
            key,
            self.ttl_seconds,
            json.dumps(metadata)
        )

    def get_metadata(self, user_id: str) -> Optional[Dict]:
        """Получает метаданные разговора"""
        key = f"{self.metadata_prefix}{user_id}"
        metadata_json = self.redis.get(key)
        return json.loads(metadata_json) if metadata_json else None

    def clear_history(self, user_id: str) -> None:
        """Очищает всё связанное с пользователем"""
        keys = [
            f"{self.history_prefix}{user_id}",
            f"{self.summary_prefix}{user_id}",
            f"{self.metadata_prefix}{user_id}"
        ]
        for key in keys:
            self.redis.delete(key)

    def get_conversation_stats(self, user_id: str) -> Dict:
        """Получает статистику по разговору"""
        history = self.get_history(user_id)

        if not history:
            return {
                "total_messages": 0,
                "user_messages": 0,
                "bot_messages": 0,
                "agent_messages": 0
            }

        stats = {
            "total_messages": len(history),
            "user_messages": sum(1 for m in history if m.role == "user"),
            "bot_messages": sum(1 for m in history if m.role == "assistant"),
            "agent_messages": sum(1 for m in history if m.role == "agent"),
            "started_at": history[0].timestamp if history else None,
            "last_message_at": history[-1].timestamp if history else None
        }

        return stats

# Глобальный экземпляр
chat_history = ChatHistoryManager(
    max_history_messages=50,
    max_context_tokens=4000,
    summary_threshold=20,
    ttl_hours=72
)