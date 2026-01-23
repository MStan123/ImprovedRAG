import re
from llm import llm, fallback_llm
from logger_setup import setup_logger
from chat_history_manager import chat_history

logger = setup_logger()

BUSINESS_ENTITIES = {
    "birbonus": [
        "бирбонус", "бир бонус", "birbonus", "bir bomus",
        "bir-bonus", "birbonuz",
    ],
    "birmarket": [
        "бирмаркет", "бир маркет", "birmarket", "bir market",
        "birmarkat", "bir-market",
    ],
}

def detect_lang(query: str) -> str:
    """
    Простая эвристическая детекция языка: азербайджанский или русский.
    """
    query_lower = query.lower()

    az_letters = "qüertuiöğəılşçmnvbzsdfghjkl"
    ru_letters = "йцукенгшщзхъфыжвэадлпорячстмьибю"

    if any(letter in query_lower for letter in az_letters):
        return 'az'
    if any(letter in query_lower for letter in ru_letters):
        return 'ru'

    return 'en'


def normalize_query(query: str) -> str:
    """Нормализация запроса с заменой вариантов названий"""
    q = query.lower()
    for canonical, variants in BUSINESS_ENTITIES.items():
        for v in variants:
            q = re.sub(rf"\b{re.escape(v)}\b", canonical, q)
    return q


def contextualize_query(query: str, user_id: str | None) -> str:
    """
    Переформулирует запрос в самостоятельный, если он ссылается на историю.
    """
    if not user_id:
        return query

    try:
        history = chat_history.get_history(user_id)
        recent_messages = history[-6:]  # последние 6 сообщений (user + assistant)

        if not recent_messages:
            return query

        # Формируем историю с ролями
        history_text = []
        for msg in recent_messages:
            role = "User" if msg.role == "user" else "Assistant"
            history_text.append(f"{role}: {msg.content}")
        history_text = "\n".join(history_text)

        system_prompt = (
            "You are a helpful assistant that reformulates the latest user question "
            "into a standalone question that can be fully understood without the chat history. "
            "You work in customer support for Birmarket (online marketplace in Azerbaijan). "
            "INSTRUCTIONS:\n"
            "- Reformulate ONLY if the question refers to previous context "
            "(e.g. 'это', 'он', 'та товар', 'bu', 'o', 'həmin', 'сколько стоит?', etc.).\n"
            "- If it's a continuation ('да', 'расскажи подробнее', 'yes, more') — keep reference to previous topic.\n"
            "- Keep the SAME language as the latest question.\n"
            "- DO NOT answer the question — only return the standalone version.\n"
            "- If the question is already independent — return it AS IS.\n"
            "- Do not add any explanations."
        )

        user_prompt = f"""Chat history (user questions only):
{history_text}

Latest user question: {query}

Standalone question:"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        try:
            response = llm.invoke(messages)
        except Exception:
            response = fallback_llm.invoke(messages)

        contextualized = response.content.strip().strip('"\'')

        if not contextualized:
            logger.warning("Contextualization returned empty, using original query")
            return query

        logger.info(f"Query contextualized:\nOriginal: {query}\nStandalone: {contextualized}")
        return contextualized

    except Exception as e:
        logger.error(f"Error during contextualization: {e}")
        return query


# ==================== SMART ROUTING SYSTEM ====================

def classify_query_with_llm(query: str) -> str:
    """
    Классификация намерения через LLM (для сложных случаев)
    """
    classification_prompt = f"""Classify the user's query intent for Birmarket marketplace support.

Choose ONE category:
- PRODUCT_SEARCH: questions about products, prices, availability, characteristics
- KNOWLEDGE_BASE: questions about delivery, returns, payment, policies, general info
- ORDER_STATUS: questions about order status, tracking
- GENERAL: greetings, thanks, casual conversation

User query: "{query}"

Response (ONE WORD ONLY):"""

    messages = [
        {"role": "system", "content": "You are a query classifier. Reply with only ONE word."},
        {"role": "user", "content": classification_prompt}
    ]

    try:
        response = llm.invoke(messages)
        intent = response.content.strip().upper()

        # Валидация ответа
        valid_intents = ['PRODUCT_SEARCH', 'KNOWLEDGE_BASE', 'ORDER_STATUS', 'GENERAL']
        if intent in valid_intents:
            logger.info(f"LLM classified intent: {intent}")
            return intent
        else:
            logger.warning(f"LLM returned invalid intent: {intent}, defaulting to GENERAL")
            return 'GENERAL'

    except Exception as e:
        logger.error(f"LLM classification failed: {e}")
        return 'GENERAL'


def extract_search_params(query: str) -> dict:
    """
    Извлекает параметры поиска из текста запроса
    """
    params = {}

    # Цена максимальная
    price_patterns = [
        r'до\s+(\d+)',
        r'maksimum\s+(\d+)',
        r'up to\s+(\d+)',
        r'под\s+(\d+)',
        r'дешевле\s+(\d+)',
        r'ucuz\s+(\d+)'
    ]

    for pattern in price_patterns:
        match = re.search(pattern, query, re.IGNORECASE)
        if match:
            params['max_price'] = float(match.group(1))
            break

    # Цена минимальная
    min_patterns = [
        r'от\s+(\d+)',
        r'minimum\s+(\d+)',
        r'from\s+(\d+)',
        r'dən\s+(\d+)'
    ]

    for pattern in min_patterns:
        match = re.search(pattern, query, re.IGNORECASE)
        if match:
            params['min_price'] = float(match.group(1))
            break

    # Наличие
    stock_keywords = ['в наличии', 'stokda', 'in stock', 'available', 'mövcud']
    if any(kw in query.lower() for kw in stock_keywords):
        params['only_in_stock'] = True
    else:
        params['only_in_stock'] = False

    logger.info(f"📊 Extracted params: {params}")
    return params


# ==================== END SMART ROUTING ====================


def needs_human_handoff(response: str, context: str, query: str) -> bool:
    """
    Определяет, нужно ли ПРЕДЛОЖИТЬ handoff (не создавать сразу!)

    Returns:
        True если нужно предложить, False если всё ОК
    """
    no_info_phrases = [
        "don't have exact information", "нет точной информации", "не имею точной информации",
        "dəqiq məlumat yoxdur", "recommend contacting", "рекомендую связаться",
        "unfortunately", "к сожалению", "təəssüf ki", "not found", "не найдено"
    ]

    # Прямой запрос на handoff - пропускаем подтверждение
    handoff_direct_requests = [
        "хочу поговорить с человеком", "соедините с оператором",
        "нужен оператор", "связаться с поддержкой",
        "поговорить с менеджером", "живой человек",
        "operator", "human", "support", "manager",
        "operatorla danış", "insan lazım"
    ]

    response_lower = response.lower()
    query_lower = query.lower()

    # 1. ПРЯМОЙ запрос на handoff - НЕ предлагаем, а сразу соединяем
    if any(trigger in query_lower for trigger in handoff_direct_requests):
        return "direct"  # Специальный маркер

    # 2. Контекст слишком короткий
    if len(context.strip()) < 50:
        return "offer"  # Предложить

    # 3. В ответе есть фразы "не знаю"
    if any(phrase in response_lower for phrase in no_info_phrases):
        return "offer"

    # 4. Ответ слишком короткий
    if len(response.strip()) < 30:
        return "offer"

    return False  # Всё ОК, handoff не нужен


def add_handoff_offer_to_response(response: str, user_lang: str) -> str:
    """
    Добавляет предложение о handoff к ответу AI

    Args:
        response: Оригинальный ответ AI
        user_lang: Язык пользователя (ru, az, en)

    Returns:
        Обновленный ответ с предложением
    """
    offers = {
        'ru': "\n\n❓ Хотите, соединю вас с оператором для более детальной помощи?",
        'az': "\n\n❓ Daha ətraflı kömək üçün sizi operatorla əlaqələndirməyimi istəyirsiniz?",
    }

    offer_text = offers.get(user_lang, offers['az'])
    return response + offer_text