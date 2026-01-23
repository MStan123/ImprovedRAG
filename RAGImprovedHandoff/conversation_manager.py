"""
Управление состоянием разговора для подтверждения действий
"""
import redis
import json
from datetime import datetime, timedelta
from typing import Optional, Dict, Literal


class ConversationState:
    """Состояние разговора с пользователем"""

    def __init__(self, redis_client=None):
        self.redis = redis_client or redis.Redis(
            host='localhost',
            port=6379,
            db=1,  # Используем отдельную БД для состояний
            decode_responses=True
        )

    def create_handoff_confirmation(
            self,
            user_id: str,
            original_query: str,
            contextualized_query: str,
            ai_response: str,
            context: str,
            ttl_minutes: int = 10
    ) -> str:
        """
        Создать состояние ожидания подтверждения handoff

        Args:
            user_id: ID пользователя
            original_query: Оригинальный вопрос
            contextualized_query: Контекстуализированный вопрос
            ai_response: Ответ AI с предложением handoff
            context: Контекст для оператора
            ttl_minutes: Время жизни состояния в минутах

        Returns:
            action_id для отслеживания
        """
        import uuid
        action_id = str(uuid.uuid4())

        state_data = {
            "action_id": action_id,
            "action_type": "handoff_confirmation",
            "user_id": user_id,
            "created_at": datetime.now().isoformat(),
            "expires_at": (datetime.now() + timedelta(minutes=ttl_minutes)).isoformat(),
            "data": {
                "original_query": original_query,
                "contextualized_query": contextualized_query,
                "ai_response": ai_response,
                "context": context
            }
        }

        key = f"pending_handoff:{user_id}"
        ttl_seconds = ttl_minutes * 60

        self.redis.setex(
            key,
            ttl_seconds,
            json.dumps(state_data, ensure_ascii=False)
        )

        return action_id

    def set_pending_action(
            self,
            user_id: str,
            action_type: str,
            action_params: Dict,
            ttl: int = 300  # 5 минут
    ) -> str:
        """
        ОБЩАЯ функция для других типов действий
        (отмена заказа, изменение адреса и т.д.)

        Args:
            user_id: ID пользователя
            action_type: тип действия (cancel_order, change_address, etc.)
            action_params: параметры действия
            ttl: время жизни состояния в секундах

        Returns:
            confirmation_token: токен для подтверждения
        """
        import uuid
        confirmation_token = str(uuid.uuid4())[:8]

        state_data = {
            "user_id": user_id,
            "action_type": action_type,
            "action_params": action_params,
            "created_at": datetime.now().isoformat(),
            "confirmation_token": confirmation_token
        }

        key = f"pending_action:{user_id}"
        self.redis.setex(
            key,
            ttl,
            json.dumps(state_data, ensure_ascii=False)
        )

        return confirmation_token

    def get_pending_action(self, user_id: str) -> Optional[Dict]:
        """
        Получает ЛЮБОЕ ожидающее действие (handoff или другое)
        Сначала проверяет handoff, потом общие действия
        """
        # Сначала проверяем handoff
        handoff_key = f"pending_handoff:{user_id}"
        handoff_data = self.redis.get(handoff_key)

        if handoff_data:
            return json.loads(handoff_data)

        # Потом проверяем другие действия
        action_key = f"pending_action:{user_id}"
        action_data = self.redis.get(action_key)

        if action_data:
            return json.loads(action_data)

        return None

    def is_awaiting_handoff_confirmation(self, user_id: str) -> bool:
        """Проверить, ждет ли пользователь подтверждения handoff"""
        action = self.get_pending_action(user_id)
        return action and action.get("action_type") == "handoff_confirmation"

    def confirm_action(self, user_id: str, confirmation_token: str = None) -> bool:
        """
        Проверяет и подтверждает действие

        Для handoff токен не нужен (просто Yes/No)
        Для других действий - проверяем токен

        Returns:
            True если действие валидно, False иначе
        """
        pending = self.get_pending_action(user_id)

        if not pending:
            return False

        # Для handoff токен не нужен
        if pending.get("action_type") == "handoff_confirmation":
            return True

        # Для других действий проверяем токен
        if confirmation_token and pending.get("confirmation_token") == confirmation_token:
            return True

        return False

    def clear_pending_action(self, user_id: str) -> bool:
        """
        Удаляет ожидающее действие
        Проверяет оба типа ключей
        """
        handoff_key = f"pending_handoff:{user_id}"
        action_key = f"pending_action:{user_id}"

        deleted = False

        if self.redis.exists(handoff_key):
            self.redis.delete(handoff_key)
            deleted = True

        if self.redis.exists(action_key):
            self.redis.delete(action_key)
            deleted = True

        return deleted

    def parse_user_response(self, query: str) -> Literal["yes", "no", "unclear"]:
        """
        Определяет намерение пользователя (да/нет/непонятно)

        Returns:
            "yes", "no", или "unclear"
        """
        query_lower = query.lower().strip()

        # Паттерны для "ДА"
        yes_patterns = [
            'да', 'yes', 'bəli', 'evet', 'hai',  # прямое да
            'конечно', 'of course', 'əlbəttə', 'tabii', 'sure',  # конечно
            'хорошо', 'ok', 'okay', 'yaxşı', 'tamam', 'alright',  # хорошо
            'давай', 'gəl', 'let\'s go',  # давай
            'соедини', 'connect', 'bağla',  # соедини
            'подтверждаю', 'confirm', 'təsdiq',  # подтверждаю
            '+', '👍', '✅', '✓'  # символы согласия
        ]

        # Паттерны для "НЕТ"
        no_patterns = [
            'нет', 'no', 'xeyr', 'hayır', 'yok',  # прямое нет
            'не надо', 'not needed', 'lazım deyil', 'gerek yok',  # не надо
            'отмена', 'cancel', 'ləğv et', 'iptal',  # отмена
            'не хочу', 'don\'t want', 'istəmirəm', 'istemiyorum',  # не хочу
            'откажусь', 'refuse', 'imtina',  # откажусь
            '-', '👎', '❌', '✗'  # символы отказа
        ]

        # Проверяем совпадения
        is_yes = any(pattern in query_lower for pattern in yes_patterns)
        is_no = any(pattern in query_lower for pattern in no_patterns)

        # Если оба - приоритет "да"
        if is_yes and is_no:
            return "yes" if query_lower.index(next(p for p in yes_patterns if p in query_lower)) < \
                            query_lower.index(next(p for p in no_patterns if p in query_lower)) else "no"

        if is_yes:
            return "yes"
        if is_no:
            return "no"

        return "unclear"

    def is_confirmation_response(self, query: str) -> bool:
        """
        Проверяет является ли сообщение подтверждением
        (для обратной совместимости со старым кодом)
        """
        return self.parse_user_response(query) in ["yes", "no"]


# Глобальный экземпляр
conversation_state = ConversationState()