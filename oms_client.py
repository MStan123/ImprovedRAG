"""
Mock OMS Client с полным функционалом
"""
from typing import Optional, Dict, Any, List
import time
from datetime import datetime, timedelta
from conversation_manager import conversation_state


class OMSClientMock:
    """Mock клиент для тестирования всех действий"""

    def __init__(self):
        # Тестовые заказы
        self.mock_orders = {
            "12345": {
                "order_id": "12345",
                "status": "CONFIRMED",
                "total": 150.50,
                "delivery_address": "ул. Низами 10, Баку",
                "phone": "+994501234567",
                "customer": {
                    "name": "Тестовый Пользователь",
                    "phone": "+994501234567",
                    "email": "test@example.com"
                },
                "created_at": (datetime.now() - timedelta(days=1)).isoformat(),
                "estimated_delivery": (datetime.now() + timedelta(days=2)).isoformat(),
                "can_cancel": True,
                "can_change_address": True,
                "items": [
                    {
                        "product_id": "P001",
                        "name": "Ноутбук ASUS",
                        "quantity": 1,
                        "price": 150.50,
                        "can_return": False
                    }
                ]
            },
            "67890": {
                "order_id": "67890",
                "status": "DELIVERED",
                "total": 299.99,
                "delivery_address": "пр. Гейдара Алиева 25, Баку",
                "phone": "+994551234567",
                "customer": {
                    "name": "Другой Пользователь",
                    "phone": "+994551234567",
                    "email": "user@example.com"
                },
                "created_at": (datetime.now() - timedelta(days=10)).isoformat(),
                "delivered_at": (datetime.now() - timedelta(days=3)).isoformat(),
                "can_cancel": False,
                "can_change_address": False,
                "items": [
                    {
                        "product_id": "P003",
                        "name": "iPhone 15",
                        "quantity": 1,
                        "price": 299.99,
                        "can_return": True
                    }
                ]
            },
            "11111": {
                "order_id": "11111",
                "status": "PENDING",
                "total": 75.00,
                "delivery_address": "ул. 28 Мая 5, Баку",
                "phone": "+994701234567",
                "customer": {
                    "name": "Новый Клиент",
                    "phone": "+994701234567",
                    "email": "new@example.com"
                },
                "created_at": (datetime.now() - timedelta(hours=2)).isoformat(),
                "estimated_delivery": (datetime.now() + timedelta(days=3)).isoformat(),
                "can_cancel": True,
                "can_change_address": True,
                "items": [
                    {
                        "product_id": "P005",
                        "name": "Наушники Sony",
                        "quantity": 2,
                        "price": 37.50,
                        "can_return": False
                    }
                ]
            }
        }

        # Счётчик для генерации ID возвратов
        self.return_counter = 1000

    def get_order(self, order_id: str) -> Optional[Dict]:
        """Получает заказ"""
        print(f"🔍 [MOCK OMS] Getting order {order_id}")
        time.sleep(0.3)
        return self.mock_orders.get(order_id)

    def cancel_order(self, order_id: str, reason: str = "Customer request") -> Dict[str, Any]:
        """Отменяет заказ"""
        print(f"🚫 [MOCK OMS] Cancelling order {order_id}")
        time.sleep(0.5)

        order = self.mock_orders.get(order_id)

        if not order:
            return {"success": False, "message": "Заказ не найден"}

        if not order["can_cancel"]:
            return {
                "success": False,
                "message": f"Нельзя отменить заказ со статусом: {order['status']}"
            }

        # Отменяем
        self.mock_orders[order_id]["status"] = "CANCELLED"
        self.mock_orders[order_id]["can_cancel"] = False

        print(f"✅ [MOCK OMS] Order {order_id} cancelled!")

        return {
            "success": True,
            "order_id": order_id,
            "status": "CANCELLED",
            "message": "Заказ успешно отменён",
            "refund_info": {
                "amount": order["total"],
                "estimated_days": "3-5"
            }
        }

    def change_delivery_address(
            self,
            order_id: str,
            new_address: str,
            new_phone: Optional[str] = None
    ) -> Dict[str, Any]:
        """Изменяет адрес доставки"""
        print(f"📍 [MOCK OMS] Changing address for order {order_id}")
        time.sleep(0.5)

        order = self.mock_orders.get(order_id)

        if not order:
            return {"success": False, "message": "Заказ не найден"}

        if not order["can_change_address"]:
            return {
                "success": False,
                "message": f"Нельзя изменить адрес для заказа со статусом: {order['status']}"
            }

        # Обновляем
        old_address = order["delivery_address"]
        self.mock_orders[order_id]["delivery_address"] = new_address

        if new_phone:
            self.mock_orders[order_id]["phone"] = new_phone

        print(f"✅ [MOCK OMS] Address changed!")

        return {
            "success": True,
            "order_id": order_id,
            "message": "Адрес доставки успешно изменён",
            "old_address": old_address,
            "new_address": new_address
        }

    def create_return(
            self,
            order_id: str,
            item_ids: List[str],
            reason: str
    ) -> Dict[str, Any]:
        """Создаёт запрос на возврат товара"""
        print(f"↩️ [MOCK OMS] Creating return for order {order_id}")
        time.sleep(0.5)

        order = self.mock_orders.get(order_id)

        if not order:
            return {"success": False, "message": "Заказ не найден"}

        if order["status"] != "DELIVERED":
            return {
                "success": False,
                "message": "Возврат возможен только для доставленных заказов"
            }

        # Проверяем что товары можно вернуть
        items_to_return = [
            item for item in order["items"]
            if item["product_id"] in item_ids and item["can_return"]
        ]

        if not items_to_return:
            return {
                "success": False,
                "message": "Выбранные товары не подлежат возврату"
            }

        # Создаём возврат
        return_id = f"RET-{self.return_counter}"
        self.return_counter += 1

        refund_amount = sum(item["price"] * item["quantity"] for item in items_to_return)

        print(f"✅ [MOCK OMS] Return {return_id} created!")

        return {
            "success": True,
            "return_id": return_id,
            "order_id": order_id,
            "message": "Запрос на возврат создан",
            "items": items_to_return,
            "refund_amount": refund_amount,
            "status": "PENDING_APPROVAL",
            "instructions": "Курьер заберёт товар в течение 2-3 рабочих дней"
        }

    def track_order(self, order_id: str) -> Dict[str, Any]:
        """Отслеживает заказ"""
        print(f"📦 [MOCK OMS] Tracking order {order_id}")
        time.sleep(0.3)

        order = self.mock_orders.get(order_id)

        if not order:
            return {"success": False, "message": "Заказ не найден"}

        # Генерируем историю статусов
        status_history = []
        created = datetime.fromisoformat(order["created_at"])

        status_history.append({
            "status": "CREATED",
            "timestamp": created.isoformat(),
            "description": "Заказ создан"
        })

        if order["status"] in ["CONFIRMED", "SHIPPED", "DELIVERED", "CANCELLED"]:
            status_history.append({
                "status": "CONFIRMED",
                "timestamp": (created + timedelta(hours=1)).isoformat(),
                "description": "Заказ подтверждён"
            })

        if order["status"] in ["SHIPPED", "DELIVERED"]:
            status_history.append({
                "status": "SHIPPED",
                "timestamp": (created + timedelta(days=1)).isoformat(),
                "description": "Заказ отправлен"
            })

        if order["status"] == "DELIVERED":
            status_history.append({
                "status": "DELIVERED",
                "timestamp": order.get("delivered_at", (created + timedelta(days=3)).isoformat()),
                "description": "Заказ доставлен"
            })

        if order["status"] == "CANCELLED":
            status_history.append({
                "status": "CANCELLED",
                "timestamp": datetime.now().isoformat(),
                "description": "Заказ отменён"
            })

        return {
            "success": True,
            "order_id": order_id,
            "current_status": order["status"],
            "estimated_delivery": order.get("estimated_delivery"),
            "delivery_address": order["delivery_address"],
            "status_history": status_history
        }


# Глобальный экземпляр
oms_client = OMSClientMock()


# --------------------------------------------------------------
# INTENT DETECTION
# --------------------------------------------------------------

class UserIntent:
    """Типы намерений пользователя"""
    QUESTION = "question"
    CANCEL_ORDER = "cancel_order"
    CHANGE_ADDRESS = "change_address"
    RETURN_ITEM = "return_item"
    TRACK_ORDER = "track_order"


def detect_intent(query: str) -> tuple[str, dict]:
    """
    Определяет намерение пользователя

    Returns:
        (intent, params)
    """
    query_lower = query.lower()

    # Извлекаем номер заказа
    import re
    order_match = re.search(r'(?:заказ|order|sifariş)[\s#:]*(\d+)', query_lower)
    order_id = order_match.group(1) if order_match else None

    # Паттерны для каждого намерения
    patterns = {
        UserIntent.CANCEL_ORDER: [
            'отменить заказ', 'отмени заказ', 'отмена заказа',
            'cancel order', 'sifarişi ləğv et'
        ],
        UserIntent.CHANGE_ADDRESS: [
            'изменить адрес', 'поменять адрес', 'сменить адрес',
            'change address', 'ünvanı dəyiş'
        ],
        UserIntent.RETURN_ITEM: [
            'вернуть товар', 'возврат товара', 'вернуть заказ',
            'return item', 'return order', 'məhsulu qaytarmaq'
        ],
        UserIntent.TRACK_ORDER: [
            'где мой заказ', 'отследить заказ', 'статус заказа',
            'track order', 'order status', 'sifarişimi izlə'
        ]
    }

    # Определяем намерение
    for intent, words in patterns.items():
        if any(word in query_lower for word in words):
            return intent, {"order_id": order_id}

    return UserIntent.QUESTION, {}


def is_confirmation_response(query: str) -> tuple[bool, bool]:
    """
    Проверяет является ли сообщение подтверждением

    Returns:
        (is_confirmation, is_positive)
    """
    query_lower = query.lower().strip()

    positive = ['да', 'yes', 'bəli', 'подтверждаю', 'confirm', 'ok', 'ок', 'давай']
    negative = ['нет', 'no', 'xeyr', 'отмена', 'cancel', 'назад']

    is_positive = any(word in query_lower for word in positive)
    is_negative = any(word in query_lower for word in negative)

    return (is_positive or is_negative), is_positive


# --------------------------------------------------------------
# ACTION HANDLERS
# --------------------------------------------------------------

def handle_cancel_order_request(user_id: str, order_id: str) -> str:
    """Обработка запроса на отмену заказа"""

    order = oms_client.get_order(order_id)

    if not order:
        return f"❌ Заказ #{order_id} не найден.\n\nПроверьте номер заказа и попробуйте снова."

    if not order["can_cancel"]:
        return (
            f"❌ К сожалению, заказ #{order_id} нельзя отменить.\n\n"
            f"📦 Статус: {order['status']}\n"
            f"Создан: {order['created_at'][:10]}\n\n"
            f"Для помощи обратитесь в поддержку."
        )

    # Сохраняем ожидающее действие
    conversation_state.set_pending_action(
        user_id=user_id,
        action_type="cancel_order",
        action_params={"order_id": order_id}
    )

    items_text = "\n".join([
        f"  • {item['name']} x{item['quantity']} - {item['price']} AZN"
        for item in order["items"]
    ])

    return (
        f"📦 Информация о заказе #{order_id}:\n\n"
        f"Статус: {order['status']}\n"
        f"Сумма: {order['total']} AZN\n"
        f"Создан: {order['created_at'][:10]}\n\n"
        f"Товары:\n{items_text}\n\n"
        f"⚠️ Вы уверены, что хотите отменить этот заказ?\n\n"
        f"Напишите 'Да' для подтверждения или 'Нет' для отмены."
    )


def handle_change_address_request(user_id: str, order_id: str, query: str) -> str:
    """Обработка запроса на изменение адреса"""

    order = oms_client.get_order(order_id)

    if not order:
        return f"❌ Заказ #{order_id} не найден."

    if not order["can_change_address"]:
        return (
            f"❌ Нельзя изменить адрес для заказа #{order_id}\n\n"
            f"Статус: {order['status']}\n\n"
            f"Изменение адреса возможно только для заказов в статусе PENDING или CONFIRMED."
        )

    # Пытаемся извлечь новый адрес из запроса
    import re
    address_match = re.search(r'на\s+(.+?)(?:\.|$)', query, re.IGNORECASE)
    new_address = address_match.group(1).strip() if address_match else None

    if not new_address:
        return (
            f"📦 Заказ #{order_id}\n"
            f"Текущий адрес: {order['delivery_address']}\n\n"
            f"📝 Пожалуйста, укажите новый адрес доставки.\n\n"
            f"Например: 'Изменить адрес заказа {order_id} на ул. Низами 25, Баку'"
        )

    # Сохраняем ожидающее действие
    conversation_state.set_pending_action(
        user_id=user_id,
        action_type="change_address",
        action_params={
            "order_id": order_id,
            "new_address": new_address
        }
    )

    return (
        f"📍 Изменение адреса доставки для заказа #{order_id}\n\n"
        f"Старый адрес: {order['delivery_address']}\n"
        f"Новый адрес: {new_address}\n\n"
        f"⚠️ Подтвердите изменение адреса?\n\n"
        f"Напишите 'Да' для подтверждения или 'Нет' для отмены."
    )


def handle_return_request(user_id: str, order_id: str) -> str:
    """Обработка запроса на возврат товара"""

    order = oms_client.get_order(order_id)

    if not order:
        return f"❌ Заказ #{order_id} не найден."

    if order["status"] != "DELIVERED":
        return (
            f"❌ Возврат возможен только для доставленных заказов.\n\n"
            f"Статус заказа #{order_id}: {order['status']}"
        )

    # Проверяем какие товары можно вернуть
    returnable_items = [item for item in order["items"] if item.get("can_return", False)]

    if not returnable_items:
        return (
            f"❌ К сожалению, товары из заказа #{order_id} не подлежат возврату.\n\n"
            f"Для уточнения обратитесь в поддержку."
        )

    items_text = "\n".join([
        f"  • {item['name']} - {item['price']} AZN"
        for item in returnable_items
    ])

    # Сохраняем ожидающее действие
    item_ids = [item["product_id"] for item in returnable_items]
    conversation_state.set_pending_action(
        user_id=user_id,
        action_type="return_item",
        action_params={
            "order_id": order_id,
            "item_ids": item_ids,
            "reason": "Customer request"
        }
    )

    return (
        f"↩️ Возврат товара из заказа #{order_id}\n\n"
        f"Товары доступные для возврата:\n{items_text}\n\n"
        f"⚠️ Подтвердите создание запроса на возврат?\n\n"
        f"Напишите 'Да' для подтверждения или 'Нет' для отмены."
    )


def handle_track_order_request(order_id: str) -> str:
    """Обработка запроса на отслеживание заказа"""

    result = oms_client.track_order(order_id)

    if not result["success"]:
        return f"❌ {result['message']}"

    # Форматируем историю статусов
    history_text = "\n".join([
        f"  ✓ {status['description']} - {status['timestamp'][:10]}"
        for status in result["status_history"]
    ])

    response = (
        f"📦 Отслеживание заказа #{order_id}\n\n"
        f"Текущий статус: {result['current_status']}\n"
        f"Адрес доставки: {result['delivery_address']}\n"
    )

    if result.get("estimated_delivery"):
        response += f"Ожидаемая доставка: {result['estimated_delivery'][:10]}\n"

    response += f"\nИстория:\n{history_text}"

    return response


def handle_confirmation(user_id: str, is_positive: bool) -> str:
    """Обработка подтверждения действия"""

    pending = conversation_state.get_pending_action(user_id)

    if not pending:
        return "❌ Нет ожидающих действий для подтверждения."

    if not is_positive:
        conversation_state.clear_pending_action(user_id)
        return "✅ Действие отменено."

    action_type = pending['action_type']
    params = pending['action_params']

    # Выполняем действие в зависимости от типа
    if action_type == "cancel_order":
        result = oms_client.cancel_order(params['order_id'])
        conversation_state.clear_pending_action(user_id)

        if result['success']:
            return (
                f"✅ Заказ #{params['order_id']} успешно отменён!\n\n"
                f"💰 Сумма возврата: {result['refund_info']['amount']} AZN\n"
                f"⏱️ Средства вернутся в течение {result['refund_info']['estimated_days']} рабочих дней\n\n"
                f"📧 Вы получите подтверждение на email."
            )
        else:
            return f"❌ Ошибка: {result['message']}"

    elif action_type == "change_address":
        result = oms_client.change_delivery_address(
            params['order_id'],
            params['new_address']
        )
        conversation_state.clear_pending_action(user_id)

        if result['success']:
            return (
                f"✅ Адрес доставки успешно изменён!\n\n"
                f"Заказ: #{params['order_id']}\n"
                f"Новый адрес: {result['new_address']}\n\n"
                f"📧 Вы получите подтверждение на email."
            )
        else:
            return f"❌ Ошибка: {result['message']}"

    elif action_type == "return_item":
        result = oms_client.create_return(
            params['order_id'],
            params['item_ids'],
            params['reason']
        )
        conversation_state.clear_pending_action(user_id)

        if result['success']:
            items_text = "\n".join([
                f"  • {item['name']} - {item['price']} AZN"
                for item in result['items']
            ])

            return (
                f"✅ Запрос на возврат создан!\n\n"
                f"ID возврата: {result['return_id']}\n"
                f"Заказ: #{params['order_id']}\n"
                f"Сумма возврата: {result['refund_amount']} AZN\n\n"
                f"Товары:\n{items_text}\n\n"
                f"📦 {result['instructions']}\n\n"
                f"📧 Вы получите подтверждение на email."
            )
        else:
            return f"❌ Ошибка: {result['message']}"

    return "❌ Неизвестное действие."
