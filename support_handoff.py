import redis
import uuid
from datetime import datetime
from typing import Optional, Dict, List, Any
import json
from langdetect import detect


class SupportHandoff:
    def __init__(self, redis_host='localhost', redis_port=6379, redis_db=0):
        self.redis_client = redis.Redis(
            host=redis_host,
            port=redis_port,
            db=redis_db,
            decode_responses=True
        )
        self.queue_key = "birmarket:support_queue"
        self.session_prefix = "birmarket:session:"
        self.agents_key = "birmarket:agents:online"

    def create_session(
            self,
            query: str,
            context: str,
            user_id: Optional[str] = None,
            user_phone: Optional[str] = None,
            user_name: Optional[str] = None,
            user_email: Optional[str] = None,
            metadata: Optional[Dict[str, Any]] = None
    ) -> str:

        session_id = str(uuid.uuid4())
        if not user_id:
            user_id = f"guest_{uuid.uuid4().hex[:8]}"

        # Определяем язык и категорию
        language = self._detect_language(query)
        category = self._detect_category(query)

        # Получаем summary для sidebar (без полной истории в messages)
        from rag_pipeline import chat_history
        conversation_summary = chat_history.get_summary_for_agent(user_id, last_n=15)

        # Initial messages: только текущий query (handoff начинается отсюда)
        initial_messages = [{
            "role": "user",
            "content": query,
            "timestamp": datetime.now().isoformat()
        }]

        session_data = {
            "session_id": session_id,
            "user_id": user_id,
            "user_phone": user_phone or "",
            "user_name": user_name or "Guest",
            "user_email": user_email or "",
            "status": "waiting",
            "query": query,
            "context_preview": context[:1000] if context else "",
            "created_at": datetime.now().isoformat(),
            "language": language,
            "category": category,
            "priority": self._calculate_priority(query, user_id),
            "conversation_history": conversation_summary,
            "agent_id": "",
            "agent_name": "",
            "assigned_at": "",
            "closed_at": "",
            "resolution": "",
            "rating": "",
            "messages": json.dumps(initial_messages),
            "metadata": json.dumps(metadata or {})  # FIX: Convert dict to JSON string
        }

        # Сохраняем сессию
        session_key = f"{self.session_prefix}{session_id}"
        self.redis_client.hset(session_key, mapping=session_data)

        # Добавляем в очередь
        if session_data["priority"] == "high":
            self.redis_client.lpush(self.queue_key, session_id)
        else:
            self.redis_client.rpush(self.queue_key, session_id)

        self.redis_client.expire(session_key, 10800)

        self._notify_operators(session_id, session_data)

        print(f"Session created: {session_id[:8]} | User: {user_name} | Messages from RAG: 0")

        return session_id

    def _detect_language(self, text: str) -> str:
        """Определяет язык текста"""
        try:
            return detect(text)
        except:
            # Fallback - по характерным символам
            if any(c in text for c in 'ğüşöçəİ'):
                return 'az'
            elif any(c in text for c in 'йцукенгшщзхъфывапролджэячсмитьбюЙЦУКЕНГШЩЗХЪФЫВАПРОЛДЖЭЯЧСМИТЬБЮ'):
                return 'ru'
            return 'en'

    def _detect_category(self, query: str) -> str:
        """Определяет категорию вопроса"""
        query_lower = query.lower()

        keywords = {
            'delivery': ['доставка', 'çatdırılma', 'delivery', 'курьер', 'kuryer'],
            'payment': ['оплата', 'ödəniş', 'payment', 'карта', 'kart', 'cash'],
            'return': ['возврат', 'qaytarma', 'return', 'обмен', 'dəyişdirmə'],
            'bonus': ['бонус', 'bonus', 'бирбонус', 'birbonus', 'cashback'],
            'product': ['товар', 'məhsul', 'product', 'качество', 'keyfiyyət'],
            'order': ['заказ', 'sifariş', 'order', 'статус', 'status'],
            'account': ['аккаунт', 'hesab', 'account', 'регистрация', 'qeydiyyat']
        }

        for category, words in keywords.items():
            if any(word in query_lower for word in words):
                return category

        return 'general'

    def _calculate_priority(self, query: str, user_id: str) -> str:
        """Рассчитывает приоритет обращения"""
        query_lower = query.lower()

        # Высокий приоритет
        high_priority_keywords = [
            'срочно', 'urgent', 'təcili',
            'не работает', 'işləmir', 'not working',
            'ошибка', 'xəta', 'error',
            'деньги', 'pul', 'money',
            'не пришёл', 'gəlmədi', 'didn\'t arrive'
        ]

        if any(word in query_lower for word in high_priority_keywords):
            return 'high'

        # Можно добавить проверку VIP пользователей
        # if self._is_vip_user(user_id):
        #     return 'high'

        return 'normal'

    def _notify_operators(self, session_id: str, session_data: Dict):
        """Уведомляет операторов о новом запросе через Redis Pub/Sub"""
        notification = {
            "event": "new_support_request",
            "session_id": session_id,
            "user_name": session_data["user_name"],
            "language": session_data["language"],
            "category": session_data["category"],
            "priority": session_data["priority"],
            "query_preview": session_data["query"][:100],
            "timestamp": datetime.now().isoformat()
        }

        self.redis_client.publish(
            "birmarket:support_notifications",
            json.dumps(notification)
        )

    def get_session(self, session_id: str) -> Optional[Dict]:
        """Получает данные сессии"""
        session_key = f"{self.session_prefix}{session_id}"
        session_data = self.redis_client.hgetall(session_key)

        if not session_data:
            return None

        # Парсим JSON поля
        if 'messages' in session_data:
            session_data['messages'] = json.loads(session_data['messages'])

        # FIX: Parse metadata JSON string back to dict
        if 'metadata' in session_data:
            try:
                session_data['metadata'] = json.loads(session_data['metadata'])
            except (json.JSONDecodeError, TypeError):
                session_data['metadata'] = {}

        return session_data

    def get_queue(self) -> List[Dict]:
        """Получает список ожидающих сессий"""
        session_ids = self.redis_client.lrange(self.queue_key, 0, -1)
        queue = []

        for session_id in session_ids:
            session = self.get_session(session_id)
            if session:
                queue.append(session)

        return queue

    def get_queue_position(self, session_id: str) -> Optional[int]:
        """Возвращает позицию сессии в очереди"""
        queue = self.redis_client.lrange(self.queue_key, 0, -1)
        try:
            return queue.index(session_id) + 1
        except ValueError:
            return None

    def assign_agent(self, session_id: str, agent_id: str, agent_name: str) -> bool:
        """
        Назначает оператора на сессию

        Args:
            session_id: ID сессии
            agent_id: ID оператора
            agent_name: имя оператора

        Returns:
            True если успешно, False если сессия не найдена
        """
        session_key = f"{self.session_prefix}{session_id}"

        if not self.redis_client.exists(session_key):
            return False

        # Обновляем статус сессии
        self.redis_client.hset(
            session_key,
            mapping={
                "status": "assigned",
                "agent_id": agent_id,
                "agent_name": agent_name,
                "assigned_at": datetime.now().isoformat()
            }
        )

        # Убираем из очереди ожидания
        self.redis_client.lrem(self.queue_key, 0, session_id)

        print(f"🎧 Agent {agent_name} assigned to session {session_id[:8]}")

        return True

    def activate_session(self, session_id: str):
        """Переводит сессию в активное состояние"""
        session_key = f"{self.session_prefix}{session_id}"
        self.redis_client.hset(session_key, "status", "active")

    def add_message(self, session_id: str, role: str, content: str, metadata: Dict = None):
        """
        Добавляет сообщение в историю сессии

        Args:
            session_id: ID сессии
            role: user, agent, system
            content: текст сообщения
            metadata: дополнительные данные
        """
        session_key = f"{self.session_prefix}{session_id}"
        messages_json = self.redis_client.hget(session_key, "messages")

        if messages_json:
            messages = json.loads(messages_json)
        else:
            messages = []

        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        }

        if metadata:
            message["metadata"] = metadata

        messages.append(message)

        self.redis_client.hset(session_key, "messages", json.dumps(messages))

        # Публикуем новое сообщение через Pub/Sub
        self.redis_client.publish(
            f"birmarket:chat:{session_id}",
            json.dumps(message)
        )

    def close_session(self, session_id: str, resolution: str = "resolved", rating: int = None):
        """
        Закрывает сессию поддержки

        Args:
            session_id: ID сессии
            resolution: результат (resolved, unresolved, escalated)
            rating: оценка пользователя (1-5)
        """
        session_key = f"{self.session_prefix}{session_id}"

        update_data = {
            "status": "closed",
            "closed_at": datetime.now().isoformat(),
            "resolution": resolution
        }

        if rating:
            update_data["rating"] = str(rating)

        self.redis_client.hset(session_key, mapping=update_data)

        # Добавляем системное сообщение
        self.add_message(session_id, "system", "Чат завершён")

        # Убираем из очереди если там ещё есть
        self.redis_client.lrem(self.queue_key, 0, session_id)

        print(f"🏁 Session closed: {session_id[:8]} | Resolution: {resolution}")

    def get_agent_stats(self, agent_id: str) -> Dict:
        """Получает статистику по оператору"""
        # Можно расширить для сбора метрик
        return {
            "agent_id": agent_id,
            "active_chats": 0,  # подсчитать из Redis
            "total_chats_today": 0,
            "avg_response_time": 0
        }

    def mark_agent_online(self, agent_id: str, agent_name: str):
        """Отмечает оператора как онлайн"""
        self.redis_client.hset(
            self.agents_key,
            agent_id,
            json.dumps({
                "name": agent_name,
                "status": "online",
                "last_seen": datetime.now().isoformat()
            })
        )
        self.redis_client.expire(self.agents_key, 300)  # 5 минут

    def mark_agent_offline(self, agent_id: str):
        """Отмечает оператора как оффлайн"""
        self.redis_client.hdel(self.agents_key, agent_id)

    def get_online_agents(self) -> List[Dict]:
        """Получает список онлайн операторов"""
        agents_data = self.redis_client.hgetall(self.agents_key)
        return [
            {"agent_id": aid, **json.loads(data)}
            for aid, data in agents_data.items()
        ]


# Глобальный экземпляр для использования в других модулях
handoff = SupportHandoff()