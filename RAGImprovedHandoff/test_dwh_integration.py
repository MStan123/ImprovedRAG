"""
Быстрая диагностика проблем с поиском
"""
from dwh_product_search import dwh_search
from rag_pipeline import smart_routing
import pandas as pd

print("="*60)
print("БЫСТРАЯ ДИАГНОСТИКА")
print("="*60)

# 1. Проверка загрузки данных
print("\n1️⃣ Проверка данных:")
stats = dwh_search.get_statistics()
print(f"   Всего товаров: {stats.get('total_products', 0)}")
print(f"   В наличии: {stats.get('in_stock_products', 0)}")
print(f"   Последнее обновление: {stats.get('last_updated', 'N/A')}")

# 2. Проверка названий товаров
print("\n2️⃣ Примеры названий товаров в базе:")
df = dwh_search.loader.get_dataframe()
if df is not None and 'Name' in df.columns:
    for i, name in enumerate(df['Name'].head(10), 1):
        print(f"   {i}. {name}")

# 3. Тестовые запросы
print("\n3️⃣ Тестирование поиска:")

test_queries = [
    "айфон",
    "iPhone",
    "Samsung",
    "смарт часы",
    "какова цена айфона в каталоге?",  # Ваш запрос
    "сколько стоят смарт часы?"          # Ваш запрос
]

for query in test_queries:
    # Debug info
    debug_info = dwh_search.debug_search(query)

    # Actual search
    products = dwh_search.search_products(query=query, only_in_stock=False, top_n=5)

    print(f"\n   Запрос: '{query}'")
    print(f"   Ключевые слова: {debug_info['extracted_keywords']}")
    print(f"   Совпадения по словам: {debug_info['word_matches']}")
    print(f"   Найдено товаров: {len(products)}")

    if products:
        print(f"   ✅ Первый товар: {products[0]['Name']}")
    else:
        print(f"   ❌ Товары не найдены")

# 4. Проверка Smart Routing
print("\n4️⃣ Проверка Smart Routing:")
test_routing = [
    "сколько стоят смарт часы?",
    "какова цена айфона?"
]

for q in test_routing:
    intent = smart_routing(q)
    print(f"   '{q}' → {intent}")

print("\n" + "="*60)
print("ДИАГНОСТИКА ЗАВЕРШЕНА")
print("="*60)