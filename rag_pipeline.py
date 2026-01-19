from pathlib import Path
import uuid
import re
from langchain_core.documents import Document
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_core.retrievers import BaseRetriever
from retrieval import hybrid_summary_search
from reranker import summary_compressor
from llm import llm, fallback_llm
from semantic_cache import semantic_cache
from stats import stats
from langdetect import detect
from logger_setup import setup_logger
from support_handoff import handoff
from langdetect import detect
from conversation_manager import conversation_state
from chat_history_manager import chat_history
from feedback_manager import feedback_manager
from dwh_product_search import dwh_search

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
        previous_queries = [
            msg.content for msg in history
            if msg.role == "user"
        ][:10]

        if not previous_queries:
            return query

        history_text = "\n".join([f"User: {q}" for q in previous_queries])

        system_prompt = (
            "You are a helpful assistant that reformulates the latest user question "
            "into a standalone question that can be fully understood without the chat history. "
            "You work in customer support for Birmarket (online marketplace in Azerbaijan). "
            "Users communicate in Russian, Azerbaijani or English.\n\n"
            "INSTRUCTIONS:\n"
            "- Reformulate ONLY if the question refers to previous context "
            "(e.g. 'это', 'он', 'та товар', 'bu', 'o', 'həmin', 'сколько стоит?', etc.).\n"
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

def detect_intent_by_keywords(query: str) -> str:
    """
    Быстрая классификация по ключевым словам
    Returns: PRODUCT_SEARCH, KNOWLEDGE_BASE, ORDER_STATUS, GENERAL
    """
    query_lower = query.lower()

    # Паттерны для поиска товаров
    product_keywords = [
        # Русский
        'цена', 'стоимость', 'сколько стоит', 'купить', 'в наличии',
        'есть ли', 'найди', 'покажи товар', 'характеристики', 'модель',
        'бренд', 'производитель', 'посоветуй товар', 'хочу купить',
        'какой телефон', 'какой смартфон', 'ноутбук', 'планшет',
        # Азербайджанский
        'qiymət', 'dəyər', 'neçəyədir', 'almaq', 'stokda',
        'tap', 'məhsul göstər', 'xüsusiyyətlər', 'model',
        'brend', 'istehsalçı', 'məsləhət ver', 'hansı telefon',
        # Английский
        'price', 'cost', 'how much', 'buy', 'in stock',
        'find product', 'show me', 'features', 'brand', 'which phone'
    ]

    # Паттерны для Knowledge Base
    kb_keywords = [
        'как', 'где', 'когда', 'почему', 'что такое',
        'доставка', 'возврат', 'оплата', 'гарантия',
        'çatdırılma', 'qaytarma', 'ödəniş', 'zəmanət',
        'delivery', 'return', 'payment', 'warranty',
        'политика', 'правила', 'условия',
        'siyasət', 'qaydalar', 'şərtlər',
        'policy', 'rules', 'terms'
    ]

    # Паттерны для заказов
    order_keywords = [
        'заказ', 'sifariş', 'order',
        'статус', 'status', 'vəziyyət',
        'где мой заказ', 'track', 'izləmək',
        'номер заказа', 'order number'
    ]

    # Подсчет совпадений
    product_score = sum(1 for kw in product_keywords if kw in query_lower)
    kb_score = sum(1 for kw in kb_keywords if kw in query_lower)
    order_score = sum(1 for kw in order_keywords if kw in query_lower)

    # Проверяем упоминания брендов и моделей (усиливаем product_score)
    product_patterns = [
        r'\b(iphone|айфон|ayfon)\b',
        r'\b(samsung|самсунг)\b',
        r'\b(xiaomi|сяоми|şaomi)\b',
        r'\b(huawei|хуавей)\b',
        r'\b(apple|эппл)\b',
        r'\b\d+gb\b',  # память
        r'\b(смартфон|smartphone|telefon)\b',
        r'\b(ноутбук|laptop|noutbuk)\b',
        r'\b(планшет|tablet|planşet)\b',
        r'\b(наушники|headphones|qulaqlıq)\b',
        r'\b(часы|watch|saat)\b',  # ДОБАВЛЕНО
        r'\b(смарт.?часы|smart.?watch|ağıllı saat)\b',  # ДОБАВЛЕНО
        r'\b(фитнес.?браслет|fitness.?band|fitness.?tracker)\b',  # ДОБАВЛЕНО
    ]

    for pattern in product_patterns:
        if re.search(pattern, query_lower):
            product_score += 2

    # Определяем намерение по максимальному счету
    scores = {
        'PRODUCT_SEARCH': product_score,
        'KNOWLEDGE_BASE': kb_score,
        'ORDER_STATUS': order_score,
        'GENERAL': 0
    }

    max_score = max(scores.values())

    if max_score == 0:
        return 'GENERAL'

    return max(scores, key=scores.get)


def calculate_keyword_confidence(query: str) -> float:
    """
    Рассчитывает уверенность в классификации по ключевым словам
    Returns: 0.0 - 1.0
    """
    query_lower = query.lower()

    # Очень явные паттерны (confidence = 1.0)
    high_confidence_patterns = [
        r'сколько стоит',
        r'neçəyədir',
        r'how much',
        r'в наличии',
        r'stokda var',
        r'in stock',
        r'где мой заказ',
        r'sifarişim harada',
        r'where is my order'
    ]

    for pattern in high_confidence_patterns:
        if re.search(pattern, query_lower):
            return 1.0

    # Подсчет общей силы сигналов
    intent = detect_intent_by_keywords(query)

    # Простая эвристика: чем больше ключевых слов, тем выше уверенность
    all_keywords = [
        'цена', 'qiymət', 'price', 'купить', 'almaq', 'buy',
        'доставка', 'çatdırılma', 'delivery', 'возврат', 'qaytarma',
        'заказ', 'sifariş', 'order', 'бонус', 'bonus'
    ]

    matches = sum(1 for kw in all_keywords if kw in query_lower)

    if matches >= 3:
        return 0.9
    elif matches >= 2:
        return 0.7
    elif matches >= 1:
        return 0.5
    else:
        return 0.3


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


def smart_routing(user_query: str) -> str:
    """
    Умная маршрутизация с комбинацией ключевых слов и LLM
    """
    # 1. Быстрая проверка ключевых слов
    keyword_intent = detect_intent_by_keywords(user_query)
    keyword_confidence = calculate_keyword_confidence(user_query)

    logger.info(f"🔍 Keyword intent: {keyword_intent} (confidence: {keyword_confidence:.2f})")

    # Если уверенность высокая (>0.8) - сразу роутим
    if keyword_confidence > 0.8:
        logger.info(f"✅ High confidence, routing to: {keyword_intent}")
        return keyword_intent

    # 2. Если не уверены - используем LLM
    logger.info("🤔 Low confidence, consulting LLM...")
    llm_intent = classify_query_with_llm(user_query)

    # 3. Если намерения совпадают - точно правильно
    if keyword_intent == llm_intent:
        logger.info(f"✅ Keyword and LLM agree: {llm_intent}")
        return llm_intent

    # 4. Если разные - доверяем LLM
    logger.info(f"⚠️ Mismatch! Keyword: {keyword_intent}, LLM: {llm_intent}. Trusting LLM.")
    return llm_intent


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
        'en': "\n\n❓ Would you like me to connect you with an operator for more detailed assistance?",
        'tr': "\n\n❓ Daha detaylı yardım için sizi bir operatörle bağlamamı ister misiniz?"
    }

    offer_text = offers.get(user_lang, offers['en'])
    return response + offer_text


def answer_query(
        query: str,
        user_id: str | None = None,
        session_id: str | None = None,
        history_last_n: int = 3
) -> tuple[str, list[Document], list[str], str]:
    """
    Основной RAG с диалогом, поддержкой feedback и интеграцией DWH.

    Returns:
        (ответ, reranked_docs, selected_files, feedback_id)
    """
    message_id = str(uuid.uuid4())
    normalized_query = normalize_query(query)

    user_lang = detect(query)
    logger.info(f"🌍 Detected language: {user_lang}")

    # 1. Контекстуализация запроса
    contextualized_query = contextualize_query(normalized_query, user_id)
    logger.info(f"Processing query (user_id: {user_id or 'no-user'}, session: {session_id or 'no-session'}): {query}")
    if contextualized_query != normalized_query:
        logger.info(f"→ Contextualized to: {contextualized_query}")

    # ========== ПРОВЕРКА: Ожидает ли пользователь подтверждения handoff? ==========
    # ВАЖНО: Это должно быть ПЕРЕД любой другой логикой!
    pending_action = conversation_state.get_pending_action(user_id)

    if pending_action and pending_action["action_type"] == "handoff_confirmation":
        logger.info(f"🔔 User {user_id} has pending handoff confirmation")

        # Используем улучшенный метод парсинга
        user_response = conversation_state.parse_user_response(query)

        # ========== ПОЛЬЗОВАТЕЛЬ СОГЛАСИЛСЯ ==========
        if user_response == "yes":
            logger.info("✅ User confirmed handoff, creating session...")

            # Получаем данные из pending action
            action_data = pending_action["data"]

            # Формируем расширенный контекст для оператора
            extended_context = f"""═══════════════════════════════════════════════════════
🤖 АВТОМАТИЧЕСКИЙ ПЕРЕВОД: Пользователь подтвердил необходимость помощи оператора
═══════════════════════════════════════════════════════

📝 ОРИГИНАЛЬНЫЙ ВОПРОС:
{action_data["original_query"]}

🤖 ОТВЕТ AI (НЕ ПОМОГ):
{action_data["ai_response"][:500]}{'...' if len(action_data["ai_response"]) > 500 else ''}

📚 КОНТЕКСТ ИЗ БАЗЫ ЗНАНИЙ:
{action_data["context"][:800]}{'...' if len(action_data["context"]) > 800 else ''}

💡 ПРИЧИНА ПЕРЕВОДА: Пользователь подтвердил запрос на помощь оператора
═══════════════════════════════════════════════════════
"""

            # Создаём handoff сессию
            session_handoff_id = handoff.create_session(
                query=action_data["contextualized_query"],
                context=extended_context,
                user_id=user_id,
                user_phone=None,
                user_name=None,
                user_email=None
            )

            # Очищаем pending action
            conversation_state.clear_pending_action(user_id)

            # Формируем ответ
            response_lang = user_lang if user_lang else 'en'
            chat_url = f"http://localhost:8001/chat?session={session_handoff_id}"

            handoff_messages = {
                'ru': f"✅ Отлично! Соединяю вас с оператором...\n\n🎫 Номер обращения: #{session_handoff_id[:8].upper()}\n⏱️ Среднее время ожидания: ~2-3 минуты\n\n🔗 Чат откроется автоматически:\n{chat_url}",
                'az': f"✅ Əla! Sizi operatorla əlaqələndirirəm...\n\n🎫 Müraciət nömrəsi: #{session_handoff_id[:8].upper()}\n⏱️ Orta gözləmə vaxtı: ~2-3 dəqiqə\n\n🔗 Çat avtomatik açılacaq:\n{chat_url}",
                'en': f"✅ Great! Connecting you with an operator...\n\n🎫 Ticket number: #{session_handoff_id[:8].upper()}\n⏱️ Average wait time: ~2-3 minutes\n\n🔗 Chat will open automatically:\n{chat_url}",
                'tr': f"✅ Harika! Sizi operatörle bağlıyorum...\n\n🎫 Başvuru numarası: #{session_handoff_id[:8].upper()}\n⏱️ Ortalama bekleme süresi: ~2-3 dakika\n\n🔗 Sohbet otomatik açılacak:\n{chat_url}"
            }

            final_response = handoff_messages.get(response_lang, handoff_messages['en'])

            stats.handoff_count += 1
            logger.info(f"✅ HANDOFF CONFIRMED - Session: {session_handoff_id}")

            # Создаем feedback с пометкой handoff_triggered
            feedback_id = feedback_manager.create_pending_feedback(
                ticket_id=message_id,
                user_id=user_id,
                session_id=session_id,
                original_query=action_data["original_query"],
                contextualized_query=action_data["contextualized_query"],
                ai_response=final_response,
                category="handoff_confirmed",
                selected_files=[],
                from_cache=False,
                handoff_triggered=True
            )

            return final_response, [], [], feedback_id

        # ========== ПОЛЬЗОВАТЕЛЬ ОТКАЗАЛСЯ ==========
        elif user_response == "no":
            logger.info("❌ User declined handoff, continuing conversation...")

            # Очищаем pending action
            conversation_state.clear_pending_action(user_id)

            # Формируем ответ
            response_lang = user_lang if user_lang else 'en'

            decline_messages = {
                'ru': "Хорошо, я постараюсь помочь вам дальше. Что бы вы хотели узнать?",
                'az': "Yaxşı, sizə kömək etməyə davam edəcəyəm. Nə öyrənmək istərdiniz?",
                'en': "Alright, I'll continue helping you. What would you like to know?",
                'tr': "Tamam, size yardımcı olmaya devam edeceğim. Ne öğrenmek istersiniz?"
            }

            final_response = decline_messages.get(response_lang, decline_messages['en'])

            # НЕ создаем feedback для отказа (это служебное сообщение)
            return final_response, [], [], None

        # ========== НЕПОНЯТНЫЙ ОТВЕТ ==========
        else:  # user_response == "unclear"
            logger.warning("⚠️ User response unclear, asking again...")

            response_lang = user_lang if user_lang else 'en'

            clarification_messages = {
                'ru': "Извините, я не понял ваш ответ. Пожалуйста, ответьте 'Да' или 'Нет':\n\nХотите соединиться с оператором?",
                'az': "Bağışlayın, cavabınızı başa düşmədim. Zəhmət olmasa 'Bəli' və ya 'Xeyr' cavabı verin:\n\nOperatorla əlaqə saxlamaq istəyirsiniz?",
                'en': "Sorry, I didn't understand your answer. Please reply 'Yes' or 'No':\n\nWould you like to connect with an operator?",
                'tr': "Üzgünüm, cevabınızı anlayamadım. Lütfen 'Evet' veya 'Hayır' yanıtı verin:\n\nOperatörle bağlantı kurmak ister misiniz?"
            }

            final_response = clarification_messages.get(response_lang, clarification_messages['en'])

            return final_response, [], [], None

    # ========== ОБЫЧНАЯ ОБРАБОТКА ЗАПРОСА (если нет pending action) ==========

    # ==================== SMART ROUTING ====================
    intent = smart_routing(contextualized_query)
    logger.info(f"🎯 Final routing decision: {intent}")

    # ==================== PRODUCT SEARCH (DWH) ====================
    if intent == "PRODUCT_SEARCH":
        logger.info("🛍️ Routing to DWH product search")

        # Извлекаем параметры поиска
        search_params = extract_search_params(contextualized_query)

        # Поиск товаров в DWH
        products = dwh_search.search_products(
            query=contextualized_query,
            only_in_stock=search_params.get('only_in_stock', True),
            min_price=search_params.get('min_price'),
            max_price=search_params.get('max_price'),
        )

        if products:
            logger.info(f"✅ Found {len(products)} products in DWH")
            products_context = dwh_search.format_products_for_llm(products, user_lang.lower())

            # Генерация ответа LLM с товарами
            messages = [
                {
                    "role": "system",
                    "content": (
                        f"You are a helpful shopping assistant for Birmarket marketplace.\n"
                        f"CRITICAL: You MUST respond ONLY in user's language.\n"
                        f"Answer questions about products based on the provided catalog data.\n"
                        f"Be friendly, concise, and helpful.\n"
                        f"ALWAYS mention: price, availability, seller name.\n"
                        f"If multiple products match - show up to {len(products)} products, but keep descriptions concise.\n"
                        f"Do not offer something extra at the end of the answer."
                    )
                },
                {
                    "role": "user",
                    "content": (
                        f"Available products:\n{products_context}\n\n"
                        f"User question: {query}\n\n"
                    )
                }
            ]

            try:
                response = llm.invoke(messages)
            except Exception:
                response = fallback_llm.invoke(messages)

            final_response = response.content

            # Подсчет токенов
            usage = response.response_metadata.get("usage", {})
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
            total_tokens = prompt_tokens + completion_tokens
            if total_tokens == 0:
                total_tokens = int(len((products_context + query + final_response).split()) * 1.3)

            stats.spent_tokens += total_tokens
            stats.llm_calls += 1

            if user_id:
                # Сохраняем вопрос пользователя
                chat_history.add_message(user_id, "user", query, metadata={"contextualized": contextualized_query})

                # Сохраняем ответ ассистента
                chat_history.add_message(user_id, "assistant", final_response,
                                         metadata={"tokens": total_tokens, "message_id": message_id})

            # Создаем feedback
            feedback_id = feedback_manager.create_pending_feedback(
                ticket_id=message_id,
                user_id=user_id,
                session_id=session_id,
                original_query=query,
                contextualized_query=contextualized_query,
                ai_response=final_response,
                category="product_search",
                selected_files=[],
                from_cache=False,
                handoff_triggered=False
            )

            logger.info("✅ Product search completed successfully")
            # Возвращаем пустые docs т.к. данные из DWH, не из файлов
            return final_response, [], [], feedback_id
        else:
            # Товары не найдены - переключаемся на обычный RAG (может быть общий вопрос)
            logger.warning("⚠️ No products found in DWH, falling back to Knowledge Base")
            intent = "KNOWLEDGE_BASE"

    # ==================== ORDER STATUS ====================
    if intent == "ORDER_STATUS":
        logger.info("📦 Routing to ORDER_STATUS handler")
        # TODO: Здесь должна быть интеграция с системой заказов
        # Пока отвечаем через RAG или handoff
        response = "Для проверки статуса заказа, пожалуйста, укажите номер вашего заказа или соединю вас со специалистом."

        return response, [], [], None

    # ==================== GENERAL (casual chat) ====================
    if intent == "GENERAL":
        logger.info("💬 General conversation detected")

        # Простой ответ без retrieval для приветствий/благодарностей
        casual_patterns = [
            r'\b(привет|здравствуй|hi|hello|salam)\b',
            r'\b(Как дела?|Как ваши дела?|how are you?|necəsiz|necesiz)\b',
            r'\b(спасибо|thanks|təşəkkür|мерси)\b',
            r'\b(пока|bye|goodbye|sağol)\b'
        ]

        is_casual = any(re.search(p, query.lower()) for p in casual_patterns)

        if is_casual:
            messages = [{
                "role": "system",
                "content": (
                    f"You are a friendly Birmarket support assistant.\n"
                    f'If there is a choise between turkish or Azerbaijani - use JUST Azerbaijani.\n'
                    f"CRITICAL: Respond ONLY in user's language.\n"
                    f"Be warm and brief."
                )
            }, {
                "role": "user",
                "content": f"User message: {query}"
            }]

            try:
                response = llm.invoke(messages)
            except Exception:
                response = fallback_llm.invoke(messages)

            final_response = response.content

            return final_response, [], [], None

        # Если не совсем casual - ищем в Knowledge Base
        intent = "KNOWLEDGE_BASE"

    # ==================== KNOWLEDGE BASE (RAG) ====================
    if intent == "KNOWLEDGE_BASE":
        logger.info("📚 Routing to Knowledge Base (RAG)")

        # 2. Гибридный retrieval
        summary_docs = hybrid_summary_search(contextualized_query, top_k=30)
        selected_files = [doc.metadata["file"] for doc in summary_docs]

        # 3. Загрузка детальных чанков
        chunks_dir = Path("/home/user/PyCharmMiscProject/RAG/chunks")
        detailed_docs = []
        for file_name in selected_files:
            file_path = chunks_dir / file_name
            if file_path.exists():
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    detailed_docs.append(
                        Document(
                            page_content=content,
                            metadata={"source": file_name, "type": "detailed_chunk"}
                        )
                    )

        if not detailed_docs:
            response = "К сожалению, не нашёл релевантной информации по вашему вопросу."

            if user_id:
                chat_history.add_message(user_id, "user", query, metadata={"contextualized": contextualized_query})
                chat_history.add_message(user_id, "assistant", response, metadata={"no_relevant_docs": True})

            feedback_id = feedback_manager.create_pending_feedback(
                ticket_id=message_id,
                user_id=user_id,
                session_id=session_id,
                original_query=query,
                contextualized_query=contextualized_query,
                ai_response=response,
                category="knowledge_base",
                selected_files=[],
                from_cache=False,
                handoff_triggered=False
            )

            return response, [], selected_files, feedback_id

        # 4. Rerank через FlashRank
        class SimpleRetriever(BaseRetriever):
            docs: list

            def _get_relevant_documents(self, query: str, **kwargs):
                return self.docs

        temp_retriever = SimpleRetriever(docs=detailed_docs)
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=summary_compressor,
            base_retriever=temp_retriever
        )
        reranked_docs = compression_retriever.invoke(contextualized_query)

        # 5. Формируем контекст
        context = "\n\n".join(doc.page_content for doc in reranked_docs)

        # 6. Семантический кэш
        from_cache = False
        cached_doc = semantic_cache.retrieve_cached_response(contextualized_query)
        if cached_doc:
            stats.cache_hits += 1
            cached_tokens = cached_doc.metadata.get("tokens", 0)
            stats.saved_tokens += cached_tokens
            logger.info("✅ From Semantic Cache")

            final_response = cached_doc.metadata["response"]
            from_cache = True

            if user_id:
                chat_history.add_message(user_id, "user", query, metadata={"contextualized": contextualized_query})
                chat_history.add_message(user_id, "assistant", cached_doc.metadata["response"],
                                         metadata={"from_cache": True})

            feedback_id = feedback_manager.create_pending_feedback(
                ticket_id=message_id,
                user_id=user_id,
                session_id=session_id,
                original_query=query,
                contextualized_query=contextualized_query,
                ai_response=final_response,
                category="knowledge_base",
                selected_files=selected_files,
                from_cache=True,
                handoff_triggered=False
            )

            return final_response, reranked_docs, selected_files, feedback_id

        # 7. Запрос к LLM
        logger.info("🤖 Request to Azure OpenAI")
        stats.llm_calls += 1

        messages = [
            {
                "role": "system",
                "content": "You are a friendly, warm, and professional AI assistant for Birmarket customer support — an online marketplace operating in Azerbaijan.\n\n" +
                           "CORE PRINCIPLES:\n" +
                           "• Answer strictly based on the provided context.\n" +
                           "• If the needed information is missing from the context — honestly say: 'Unfortunately, I don't have the exact information on this.'\n" +
                           "• NEVER invent, guess, or assume facts about Birmarket, prices, delivery times, product availability, policies, or any other details.\n\n" +
                           "ALLOWED SMALL TALK (answer naturally and kindly):\n" +
                           "• Greetings (salam, hi, hello, good day, etc.)\n" +
                           "• Questions about language ('Can I speak Russian?', 'English?', 'Azərbaycanca?')\n" +
                           "• Thanks and polite phrases ('təşəkkür', 'thank you')\n" +
                           "• Casual questions like 'How are you?', 'What can you do?'\n" +
                           "• Requests to repeat or explain something again\n\n" +
                           "LANGUAGE & TONE RULES:\n" +
                           "• Be ready to answer ONLY in Russian, Azerbaijani, English, or Turkish — whichever the user chooses.\n" +
                           "• If the context contains useful information in Azerbaijani — use it and translate the relevant parts accurately into the user's language.\n" +
                           "• Speak politely, warmly, and concisely — like a friendly and competent shop assistant.\n" +
                           "• Use respectful 'You' / 'siz' form unless the customer clearly switches to informal 'you' / 'sən' first.\n" +
                           "• If client talk you to connect him with operator - do not say something extra, just 'Of course.'.\n" +
                           "• Keep answers short but complete — enough for the customer to clearly understand.\n\n" +
                           "STRICT BOUNDARIES:\n" +
                           "• You are NOT a general-purpose AI. If the question is clearly outside Birmarket support (politics, weather, programming, personal advice, religion, relationships, etc.) — politely redirect: 'Sorry, I specialize only in helping with purchases and Birmarket services. For other topics I recommend using other resources.'\n" +
                           "• Never discuss your own nature, model, training, xAI, Grok, or any internal details.\n" +
                           "• Avoid excessive apologies or self-deprecating phrases unless there is a real reason.\n\n" +
                           "• Your main goal: Help customers quickly, clearly, and pleasantly with anything related to shopping on Birmarket."
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {query}"
            }
        ]

        try:
            response = llm.invoke(messages)
        except Exception:
            response = fallback_llm.invoke(messages)

        # Подсчёт токенов
        usage = response.response_metadata.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        total_tokens = prompt_tokens + completion_tokens
        if total_tokens == 0:
            total_tokens = int(len((context + contextualized_query + response.content).split()) * 1.3)
        stats.spent_tokens += total_tokens

        final_response = response.content
        handoff_triggered = False

        # 8. Сохранение истории (ДО handoff логики)
        if user_id:
            # Сохраняем вопрос пользователя
            chat_history.add_message(
                user_id,
                "user",
                query,
                metadata={"contextualized": contextualized_query}
            )

            # Сохраняем ответ ассистента
            chat_history.add_message(
                user_id,
                "assistant",
                final_response,
                metadata={
                    "tokens": total_tokens,
                    "message_id": message_id
                }
            )

        # ========== ПРОВЕРКА НУЖЕН ЛИ HANDOFF ==========
        handoff_decision = needs_human_handoff(final_response, context, query)

        if handoff_decision == "direct":
            # ПРЯМОЙ запрос - соединяем сразу без подтверждения
            logger.info("🔴 DIRECT handoff request detected")

            extended_context = f"""═══════════════════════════════════════════════════════
🤖 ПРЯМОЙ ЗАПРОС НА ОПЕРАТОРА
═══════════════════════════════════════════════════════

📝 ЗАПРОС ПОЛЬЗОВАТЕЛЯ:
{query}

📚 КОНТЕКСТ:
{context[:800]}{'...' if len(context) > 800 else ''}

💡 ПРИЧИНА: Пользователь напрямую запросил связь с оператором
═══════════════════════════════════════════════════════
"""

            session_handoff_id = handoff.create_session(
                query=contextualized_query,
                context=extended_context,
                user_id=user_id,
                user_phone=None,
                user_name=None,
                user_email=None
            )

            response_lang = user_lang if user_lang else 'en'
            chat_url = f"http://localhost:8001/chat?session={session_handoff_id}"

            handoff_messages = {
                'ru': f"✅ Конечно! Соединяю вас с оператором...\n\n🎫 Номер обращения: #{session_handoff_id[:8].upper()}\n⏱️ Среднее время ожидания: ~2-3 минуты\n\n🔗 Чат:\n{chat_url}",
                'az': f"✅ Əlbəttə! Sizi operatorla əlaqələndirirəm...\n\n🎫 Müraciət nömrəsi: #{session_handoff_id[:8].upper()}\n⏱️ Orta gözləmə vaxtı: ~2-3 dəqiqə\n\n🔗 Çat:\n{chat_url}",
                'en': f"✅ Of course! Connecting you with an operator...\n\n🎫 Ticket: #{session_handoff_id[:8].upper()}\n⏱️ Wait time: ~2-3 min\n\n🔗 Chat:\n{chat_url}"
            }

            final_response = handoff_messages.get(response_lang, handoff_messages['en'])
            stats.handoff_count += 1
            handoff_triggered = True

        elif handoff_decision == "offer":
            # ПРЕДЛОЖИТЬ handoff - создаем pending action
            logger.info("🟡 Handoff needed, creating confirmation request")

            # Добавляем предложение к ответу
            final_response = add_handoff_offer_to_response(final_response, user_lang)

            # Создаем pending action
            conversation_state.create_handoff_confirmation(
                user_id=user_id,
                original_query=query,
                contextualized_query=contextualized_query,
                ai_response=final_response,
                context=context,
                ttl_minutes=10
            )

            logger.info(f"⏳ Waiting for user confirmation (user_id: {user_id})")

        else:
            # Всё ОК, кэшируем ответ
            semantic_cache.store_response(contextualized_query, final_response, total_tokens)
            stats.cached_responses += 1
            logger.info("✅ Response cached")

        # Создаем feedback
        feedback_id = feedback_manager.create_pending_feedback(
            ticket_id=message_id,
            user_id=user_id,
            session_id=session_id,
            original_query=query,
            contextualized_query=contextualized_query,
            ai_response=final_response,
            category="knowledge_base",
            selected_files=selected_files,
            from_cache=from_cache,
            handoff_triggered=handoff_triggered
        )

        return final_response, reranked_docs, selected_files, feedback_id

    # Если попали сюда - что-то пошло не так, возвращаем дефолт
    return "Извините, произошла ошибка. Попробуйте еще раз.", [], [], None

__all__ = ["answer_query"]