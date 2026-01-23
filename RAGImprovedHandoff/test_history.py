# test_with_history.py
from rag_pipeline import answer_query
from chat_history_manager import chat_history

user_id = "test_user_123"
chat_history.clear_history(user_id)

print("=== Тест с историей ===\n")

# Диалог 1
print("1️⃣ User: Привет")
resp, _, _ = answer_query("Привет", user_id=user_id)
print(f"Bot: {resp[:50]}...\n")

# Диалог 2
print("2️⃣ User: Как работает доставка?")
resp, _, _ = answer_query("Как работает доставка?", user_id=user_id)
print(f"Bot: {resp[:80]}...\n")

# Диалог 3 - контекстный вопрос
print("3️⃣ User: А сколько это стоит?")  # Бот должен понять что речь о доставке!
resp, _, _ = answer_query("А сколько это стоит?", user_id=user_id)
print(f"Bot: {resp[:80]}...\n")

# Проверяем историю
print("📊 История:")
history = chat_history.get_history(user_id)
for msg in history:
    print(f"  {msg.role}: {msg.content[:40]}...")

# Проверяем summary
print("\n📜 Summary для оператора:")
summary = chat_history.get_summary_for_agent(user_id)
print(summary)