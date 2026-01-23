#!/usr/bin/env python3
"""
Тест дедупликации товаров и форматирования с описаниями
"""

import pandas as pd
from dwh_product_search import dwh_search


def test_deduplication():
    """Тестирует дедупликацию товаров"""

    print("=" * 70)
    print("TEST: Deduplication & Formatting")
    print("=" * 70)

    # Получаем данные
    df = dwh_search.loader.get_dataframe()

    if df is None or df.empty:
        print("❌ DataFrame is empty")
        return

    print(f"\n✅ Total rows in DataFrame: {len(df)}")
    print(f"📊 Columns: {list(df.columns)}")

    # Проверяем дублирование
    unique_products = df['MpProductID'].nunique()
    print(f"\n🔢 Unique MpProductID: {unique_products}")
    print(f"📋 Total rows: {len(df)}")
    print(f"🔄 Duplication ratio: {len(df) / unique_products:.2f}x")

    # Пример дубликатов
    sample_id = df['MpProductID'].iloc[0]
    duplicates = df[df['MpProductID'] == sample_id]

    print(f"\n📦 Example: MpProductID = {sample_id}")
    print(f"   Variants: {len(duplicates)}")

    for idx, row in duplicates.iterrows():
        print(f"   • {row['Name'][:60]}")
        if pd.notna(row.get('Description')) and row['Description']:
            print(f"     Desc: {row['Description'][:80]}...")

    # Тестируем дедупликацию
    print("\n" + "=" * 70)
    print("Testing deduplicate_products()")
    print("=" * 70)

    sample_products = duplicates.to_dict('records')
    deduplicated = dwh_search.deduplicate_products(sample_products)

    print(f"\n✅ Input: {len(sample_products)} products")
    print(f"✅ Output: {len(deduplicated)} products")

    if deduplicated:
        product = deduplicated[0]
        print(f"\n📦 Deduplicated product:")
        print(f"   ID: {product['MpProductID']}")
        print(f"   Price: {product['RetailPrice']} AZN")
        print(f"   Stock: {product['Qty']}")
        print(f"   Names count: {len(product.get('AllNames', []))}")

        for name in product.get('AllNames', []):
            print(f"      • {name}")

        print(f"   Descriptions count: {len(product.get('AllDescriptions', []))}")

        for desc in product.get('AllDescriptions', []):
            if desc:
                print(f"      • {desc[:100]}...")

    # Тестируем форматирование
    print("\n" + "=" * 70)
    print("Testing format_products_for_llm_v2()")
    print("=" * 70)

    for lang in ['ru', 'az']:
        print(f"\n📝 Language: {lang}")
        print("-" * 70)

        formatted = dwh_search.format_products_for_llm_v2(
            products=sample_products,
            language=lang,
            include_descriptions=True,
            max_description_length=150
        )

        print(formatted)
        print(f"\n📏 Total length: {len(formatted)} characters")


def test_full_search_flow():
    """Тестирует полный flow: поиск -> дедупликация -> форматирование"""

    print("\n\n")
    print("=" * 70)
    print("TEST: Full Search Flow")
    print("=" * 70)

    test_queries = [
        "smart watch",
        "часы",
        "saat"
    ]

    for query in test_queries:
        print(f"\n🔍 Query: '{query}'")
        print("-" * 70)

        # Поиск
        products = dwh_search.search_products(
            query=query,
            only_in_stock=True,
            only_active=True,
            top_n=5
        )

        print(f"✅ Found: {len(products)} products (before dedup)")

        if not products:
            print("❌ No products found")
            continue

        # Умный выбор
        best = dwh_search.select_best_products_to_show(
            products=products,
            query=query,
            max_products=3
        )

        print(f"🎯 Selected: {len(best)} best products")

        # Форматирование
        formatted_ru = dwh_search.format_products_for_llm_v2(
            products=best,
            language='ru',
            include_descriptions=True,
            max_description_length=150
        )

        print("\n📝 Formatted (RU):")
        print(formatted_ru)

        print(f"\n📏 Context size: {len(formatted_ru)} chars")


def test_description_extraction():
    """Тестирует извлечение описаний на разных языках"""

    print("\n\n")
    print("=" * 70)
    print("TEST: Description Language Detection")
    print("=" * 70)

    df = dwh_search.loader.get_dataframe()

    # Берем первый товар с описанием
    sample = df[df['Description'].notna() & (df['Description'] != '')].head(2)

    for idx, row in sample.iterrows():
        print(f"\n📦 Product: {row['Name']}")
        print(f"   Description: {row['Description'][:200]}...")

        # Определяем язык
        desc = row['Description']

        # Простая эвристика
        has_cyrillic = any('\u0400' <= c <= '\u04FF' for c in desc)
        has_az_chars = any(c in 'əüöşıçğ' for c in desc.lower())

        if has_az_chars:
            detected = 'Azerbaijani'
        elif has_cyrillic:
            detected = 'Russian'
        else:
            detected = 'English/Other'

        print(f"   Detected language: {detected}")


if __name__ == "__main__":
    print("\n🧪 Starting DWH Product Search Tests\n")

    # 1. Дедупликация
    test_deduplication()

    # 2. Полный flow
    test_full_search_flow()

    # 3. Извлечение описаний
    test_description_extraction()

    print("\n\n✅ All tests completed!")