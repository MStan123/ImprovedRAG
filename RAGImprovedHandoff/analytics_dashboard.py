"""
Простой дашборд для аналитики feedback
Можно запустить как отдельное приложение или интегрировать в админ-панель
"""

from feedback_manager import feedback_manager
from datetime import datetime, timedelta
import json


def print_analytics_report(days_back: int = 7):
    """Выводит отчёт по аналитике за последние N дней"""

    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)

    analytics = feedback_manager.get_analytics(
        start_date=start_date,
        end_date=end_date
    )

    print("\n" + "=" * 60)
    print(f"📊 FEEDBACK ANALYTICS REPORT")
    print(f"Period: {start_date.date()} — {end_date.date()}")
    print("=" * 60 + "\n")

    # Общая статистика
    print("📈 OVERALL STATISTICS")
    print(f"  Total responses evaluated: {analytics['total']}")
    print(f"  👍 Positive (Да): {analytics['yes_count']} ({analytics['yes_percentage']}%)")
    print(f"  👎 Negative (Нет): {analytics['no_count']} ({analytics['no_percentage']}%)")
    print()

    # По категориям
    if analytics.get('by_category'):
        print("📂 BY CATEGORY")
        for category, counts in analytics['by_category'].items():
            total_cat = counts['yes'] + counts['no']
            yes_pct = round((counts['yes'] / total_cat) * 100, 1) if total_cat > 0 else 0
            print(f"  {category}:")
            print(f"    Total: {total_cat} | Yes: {counts['yes']} ({yes_pct}%) | No: {counts['no']}")
        print()

    # По дням
    if analytics.get('by_date'):
        print("📅 DAILY BREAKDOWN (last 7 days)")
        for date, counts in sorted(analytics['by_date'].items(), reverse=True)[:7]:
            total_day = counts['yes'] + counts['no']
            yes_pct = round((counts['yes'] / total_day) * 100, 1) if total_day > 0 else 0
            print(f"  {date}: {total_day} total | {counts['yes']} yes ({yes_pct}%) | {counts['no']} no")
        print()

    # По версиям БЗ
    if analytics.get('by_kb_version'):
        print("📚 BY KNOWLEDGE BASE VERSION")
        for kb_ver, counts in analytics['by_kb_version'].items():
            total_kb = counts['yes'] + counts['no']
            yes_pct = round((counts['yes'] / total_kb) * 100, 1) if total_kb > 0 else 0
            print(f"  Version {kb_ver}:")
            print(f"    Total: {total_kb} | Yes: {counts['yes']} ({yes_pct}%) | No: {counts['no']}")
        print()

    # Рекомендации
    print("💡 RECOMMENDATIONS")
    if analytics['no_percentage'] > 30:
        print("  ⚠️  High negative feedback rate (>30%). Consider:")
        print("     - Reviewing knowledge base quality")
        print("     - Analyzing common 'No' responses for patterns")
        print("     - Updating outdated information")
    elif analytics['yes_percentage'] > 80:
        print("  ✅ Excellent performance! Keep up the good work.")
    else:
        print("  📊 Moderate performance. Room for improvement.")

    print("\n" + "=" * 60 + "\n")


def export_analytics_to_json(filepath: str = "./data/analytics_report.json"):
    """Экспортирует аналитику в JSON"""
    analytics = feedback_manager.get_analytics()

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(analytics, f, ensure_ascii=False, indent=2)

    print(f"✅ Analytics exported to: {filepath}")


def get_low_rated_responses(min_rating_threshold: float = 0.5, limit: int = 10):
    """
    Находит самые плохо оценённые ответы для ручного анализа

    Returns:
        list of dict с информацией о плохих ответах
    """
    import json

    feedback_file = feedback_manager.feedback_file
    if not feedback_file.exists():
        return []

    no_responses = []

    with open(feedback_file, 'r', encoding='utf-8') as f:
        for line in f:
            record = json.loads(line)
            if record['rating'] == 'no':
                no_responses.append({
                    'date': record['created_at'],
                    'query': record['original_query'],
                    'response': record['ai_response'][:200] + '...',  # первые 200 символов
                    'category': record.get('category', 'uncategorized'),
                    'from_cache': record['from_cache'],
                    'files_used': record['selected_files']
                })

    return no_responses[:limit]


# Пример использования
if __name__ == "__main__":
    print_analytics_report(days_back=7)

    # Экспорт в JSON
    export_analytics_to_json()

    # Показать плохие ответы
    print("\n" + "=" * 60)
    print("🔍 LOW-RATED RESPONSES FOR REVIEW")
    print("=" * 60 + "\n")

    bad_responses = get_low_rated_responses(limit=5)
    for i, resp in enumerate(bad_responses, 1):
        print(f"{i}. [{resp['date'][:10]}] Category: {resp['category']}")
        print(f"   Query: {resp['query']}")
        print(f"   Response preview: {resp['response']}")
        print(f"   From cache: {resp['from_cache']}")
        print()