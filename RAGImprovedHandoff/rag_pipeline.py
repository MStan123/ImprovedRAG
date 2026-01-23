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
from logger_setup import setup_logger
from support_handoff import handoff
from conversation_manager import conversation_state
from chat_history_manager import chat_history
from feedback_manager import feedback_manager
from dwh_product_search import dwh_search
from useful_func import normalize_query, detect_lang, needs_human_handoff, contextualize_query, classify_query_with_llm,add_handoff_offer_to_response

logger = setup_logger()

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

    user_lang = detect_lang(query)
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
            response_lang = user_lang if user_lang else 'az'
            chat_url = f"http://localhost:8001/chat?session={session_handoff_id}"

            handoff_messages = {
                'ru': f"✅ Отлично! Соединяю вас с оператором...\n\n🎫 Номер обращения: #{session_handoff_id[:8].upper()}\n⏱️ Среднее время ожидания: ~2-3 минуты\n\n🔗 Чат откроется автоматически:\n{chat_url}",
                'az': f"✅ Əla! Sizi operatorla əlaqələndirirəm...\n\n🎫 Müraciət nömrəsi: #{session_handoff_id[:8].upper()}\n⏱️ Orta gözləmə vaxtı: ~2-3 dəqiqə\n\n🔗 Çat avtomatik açılacaq:\n{chat_url}",
            }

            final_response = handoff_messages.get(response_lang, handoff_messages['az'])

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

            return final_response, [], [], None

        # ========== ПОЛЬЗОВАТЕЛЬ ОТКАЗАЛСЯ ==========
        elif user_response == "no":
            logger.info("❌ User declined handoff, continuing conversation...")

            # Очищаем pending action
            conversation_state.clear_pending_action(user_id)

            # Формируем ответ
            response_lang = user_lang if user_lang else 'az'

            decline_messages = {
                'ru': "Хорошо, я постараюсь помочь вам дальше. Что бы вы хотели узнать?",
                'az': "Yaxşı, sizə kömək etməyə davam edəcəyəm. Nə öyrənmək istərdiniz?",
            }

            final_response = decline_messages.get(response_lang, decline_messages['az'])

            # НЕ создаем feedback для отказа (это служебное сообщение)
            return final_response, [], [], None

        # ========== НЕПОНЯТНЫЙ ОТВЕТ ==========
        else:  # user_response == "unclear"
            logger.warning("⚠️ User response unclear, asking again...")

            response_lang = user_lang if user_lang else 'az'

            clarification_messages = {
                'ru': "Извините, я не понял ваш ответ. Пожалуйста, ответьте 'Да' или 'Нет':\n\nХотите соединиться с оператором?",
                'az': "Bağışlayın, cavabınızı başa düşmədim. Zəhmət olmasa 'Bəli' və ya 'Xeyr' cavabı verin:\n\nOperatorla əlaqə saxlamaq istəyirsiniz?",
            }

            final_response = clarification_messages.get(response_lang, clarification_messages['az'])

            return final_response, [], [], None

    # ========== ОБЫЧНАЯ ОБРАБОТКА ЗАПРОСА (если нет pending action) ==========

    # ==================== SMART ROUTING ====================
    intent = classify_query_with_llm(contextualized_query)
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

        dialog_history = []
        if user_id:
            recent = chat_history.get_history(user_id)[-6:]  # последние 6 сообщений
            for msg in recent:
                role = "User" if msg.role == "user" else "Assistant"
                dialog_history.append(f"{role}: {msg.content}")

        history_text = "\n\n".join(dialog_history) if dialog_history else "(no previous messages)"

        # Проверяем, продолжение ли разговора
        is_followup = len(dialog_history) >= 2

        # Базовый системный промпт
        base_system = (
            "You are a friendly, warm, and professional AI assistant for Birmarket customer support — an online marketplace operating in Azerbaijan.\n\n"
            "CORE PRINCIPLES:\n"
            "• Answer strictly based on the provided context.\n"
            "• If the needed information is missing from the context — honestly say: 'Unfortunately, I don't have the exact information on this.'\n"
            "• NEVER invent, guess, or assume facts about Birmarket, prices, delivery times, product availability, policies, or any other details.\n\n"
            "ALLOWED SMALL TALK (answer naturally and kindly):\n"
            "• Greetings (salam, hi, hello, good day, etc.)\n"
            "• Questions about language ('Can I speak Russian?', 'Azərbaycanca?')\n"
            "• Thanks and polite phrases ('təşəkkür', 'thank you')\n"
            "• Casual questions like 'How are you?', 'What can you do?'\n"
            "• Requests to repeat or explain something again\n\n"
            "LANGUAGE & TONE RULES:\n"
            "• Be ready to answer ONLY in Russian, Azerbaijani — whichever the user chooses.\n"
            "• Speak politely, warmly, and concisely — like a friendly and competent shop assistant.\n"
            "• Keep answers short but complete.\n\n"
            "STRICT BOUNDARIES:\n"
            "• You are NOT a general-purpose AI. If the question is clearly outside Birmarket support — politely redirect.\n"
            "• Never discuss your own nature, model, training, xAI, Grok, etc."
        )

        # Добавляем инструкцию для продолжения разговора, если это follow-up
        if is_followup:
            continuation_instruction = (
                "\n\nIMPORTANT — THIS IS A CONTINUING CONVERSATION:\n"
                "- ALWAYS maintain context from previous messages.\n"
                "- If user says 'да', 'расскажи подробнее', 'ещё', 'more', 'tell me more', 'yes', etc. — "
                "continue and expand on the LAST topic you discussed.\n"
                "- Refer to your previous response when continuing.\n"
                "- Never forget what you just told the user.\n"
            )
            system_content = base_system + continuation_instruction
        else:
            system_content = base_system

        # Формируем сообщения
        messages = [
            {"role": "system", "content": system_content},
            {
                "role": "user",
                "content": f"""Recent chat history (for context):
                {history_text}

                Knowledge context:
                {context}

                Current question: {query}"""
            }
        ]

        # Вызов LLM
        logger.info("🤖 Request to Azure OpenAI")
        stats.llm_calls += 1

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

            response_lang = user_lang if user_lang else 'az'
            chat_url = f"http://localhost:8001/chat?session={session_handoff_id}"

            handoff_messages = {
                'ru': f"✅ Конечно! Соединяю вас с оператором...\n\n🎫 Номер обращения: #{session_handoff_id[:8].upper()}\n⏱️ Среднее время ожидания: ~2-3 минуты\n\n🔗 Чат:\n{chat_url}",
                'az': f"✅ Əlbəttə! Sizi operatorla əlaqələndirirəm...\n\n🎫 Müraciət nömrəsi: #{session_handoff_id[:8].upper()}\n⏱️ Orta gözləmə vaxtı: ~2-3 dəqiqə\n\n🔗 Çat:\n{chat_url}",
            }

            final_response = handoff_messages.get(response_lang, handoff_messages['az'])
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