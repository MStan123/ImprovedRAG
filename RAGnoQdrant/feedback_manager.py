from datetime import datetime
from pathlib import Path
import json
import uuid
from typing import Optional, Literal
from dataclasses import dataclass, asdict
from logger_setup import setup_logger

logger = setup_logger()


@dataclass
class FeedbackRecord:
    """Структура для хранения оценки ответа"""
    feedback_id: str
    ticket_id: str  # ID тикета/сообщения
    user_id: Optional[str]  # ID пользователя
    session_id: Optional[str]  # ID сессии (fallback если нет user_id)

    # Контекст запроса
    original_query: str  # Оригинальный вопрос
    contextualized_query: str  # Контекстуализированный вопрос
    ai_response: str  # Ответ ИИ

    # Оценка
    rating: Literal["yes", "no"]  # Помог ли ответ

    # Метаданные
    category: Optional[str]  # Категория вопроса (опционально)
    knowledge_base_version: str  # Версия базы знаний
    selected_files: list[str]  # Файлы, использованные для ответа

    # Временные метки
    created_at: str  # ISO timestamp

    # Дополнительные данные
    from_cache: bool = False  # Был ли ответ из кэша
    handoff_triggered: bool = False  # Был ли переход к человеку

    def to_dict(self):
        return asdict(self)


class FeedbackManager:
    """Менеджер для управления оценками ответов"""

    def __init__(self, storage_path: str = "./data/feedback"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # Файлы для хранения
        self.feedback_file = self.storage_path / "feedback_records.jsonl"
        self.pending_file = self.storage_path / "pending_feedback.json"

        # Версия базы знаний (можно читать из конфига)
        self.kb_version = self._get_kb_version()

        # Хранилище ожидающих оценки записей
        self.pending_feedback = self._load_pending()

    def _get_kb_version(self) -> str:
        """Получить версию базы знаний"""
        version_file = Path("./data/kb_version.txt")
        if version_file.exists():
            return version_file.read_text().strip()

        # Fallback: дата последнего обновления чанков
        chunks_dir = Path("/home/user/PyCharmMiscProject/RAG/chunks")
        if chunks_dir.exists():
            latest = max(chunks_dir.glob("*.txt"),
                         key=lambda p: p.stat().st_mtime,
                         default=None)
            if latest:
                mtime = datetime.fromtimestamp(latest.stat().st_mtime)
                return mtime.strftime("%Y-%m-%d")

        return datetime.now().strftime("%Y-%m-%d")

    def _load_pending(self) -> dict:
        """Загрузить ожидающие оценки записи"""
        if self.pending_file.exists():
            with open(self.pending_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}

    def _save_pending(self):
        """Сохранить ожидающие оценки"""
        with open(self.pending_file, 'w', encoding='utf-8') as f:
            json.dump(self.pending_feedback, f, ensure_ascii=False, indent=2)

    def create_pending_feedback(
            self,
            ticket_id: str,
            user_id: Optional[str],
            session_id: Optional[str],
            original_query: str,
            contextualized_query: str,
            ai_response: str,
            category: Optional[str] = None,
            selected_files: list[str] = None,
            from_cache: bool = False,
            handoff_triggered: bool = False
    ) -> str:
        """
        Создать запись, ожидающую оценки от пользователя

        Returns:
            feedback_id для отслеживания
        """
        feedback_id = str(uuid.uuid4())

        pending_record = {
            "feedback_id": feedback_id,
            "ticket_id": ticket_id,
            "user_id": user_id,
            "session_id": session_id,
            "original_query": original_query,
            "contextualized_query": contextualized_query,
            "ai_response": ai_response,
            "category": category,
            "knowledge_base_version": self.kb_version,
            "selected_files": selected_files or [],
            "created_at": datetime.now().isoformat(),
            "from_cache": from_cache,
            "handoff_triggered": handoff_triggered
        }

        self.pending_feedback[feedback_id] = pending_record
        self._save_pending()

        logger.info(f"Created pending feedback record: {feedback_id}")
        return feedback_id

    def submit_feedback(
            self,
            feedback_id: str,
            rating: Literal["yes", "no"]
    ) -> bool:
        """
        Сохранить оценку пользователя

        Args:
            feedback_id: ID записи обратной связи
            rating: "yes" или "no"

        Returns:
            True если успешно сохранено
        """
        if feedback_id not in self.pending_feedback:
            logger.warning(f"Feedback ID not found: {feedback_id}")
            return False

        # Получаем pending запись
        pending = self.pending_feedback[feedback_id]

        # Создаём финальную запись
        feedback_record = FeedbackRecord(
            feedback_id=feedback_id,
            ticket_id=pending["ticket_id"],
            user_id=pending["user_id"],
            session_id=pending["session_id"],
            original_query=pending["original_query"],
            contextualized_query=pending["contextualized_query"],
            ai_response=pending["ai_response"],
            rating=rating,
            category=pending["category"],
            knowledge_base_version=pending["knowledge_base_version"],
            selected_files=pending["selected_files"],
            created_at=pending["created_at"],
            from_cache=pending["from_cache"],
            handoff_triggered=pending["handoff_triggered"]
        )

        # Сохраняем в JSONL (append-only для аналитики)
        with open(self.feedback_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(feedback_record.to_dict(), ensure_ascii=False) + '\n')

        # Удаляем из pending
        del self.pending_feedback[feedback_id]
        self._save_pending()

        logger.info(f"Feedback submitted: {feedback_id} -> {rating}")
        return True

    def get_analytics(
            self,
            start_date: Optional[datetime] = None,
            end_date: Optional[datetime] = None,
            category: Optional[str] = None
    ) -> dict:
        """
        Получить аналитику по оценкам

        Returns:
            {
                'total': int,
                'yes_count': int,
                'no_count': int,
                'yes_percentage': float,
                'no_percentage': float,
                'by_category': {...},
                'by_date': {...},
                'by_kb_version': {...}
            }
        """
        if not self.feedback_file.exists():
            return {
                'total': 0,
                'yes_count': 0,
                'no_count': 0,
                'yes_percentage': 0,
                'no_percentage': 0
            }

        records = []
        with open(self.feedback_file, 'r', encoding='utf-8') as f:
            for line in f:
                record = json.loads(line)

                # Фильтрация по датам
                record_date = datetime.fromisoformat(record['created_at'])
                if start_date and record_date < start_date:
                    continue
                if end_date and record_date > end_date:
                    continue

                # Фильтрация по категории
                if category and record.get('category') != category:
                    continue

                records.append(record)

        if not records:
            return {
                'total': 0,
                'yes_count': 0,
                'no_count': 0,
                'yes_percentage': 0,
                'no_percentage': 0
            }

        # Базовая статистика
        yes_count = sum(1 for r in records if r['rating'] == 'yes')
        no_count = sum(1 for r in records if r['rating'] == 'no')
        total = len(records)

        analytics = {
            'total': total,
            'yes_count': yes_count,
            'no_count': no_count,
            'yes_percentage': round((yes_count / total) * 100, 2),
            'no_percentage': round((no_count / total) * 100, 2),
        }

        # Аналитика по категориям
        by_category = {}
        for record in records:
            cat = record.get('category', 'uncategorized')
            if cat not in by_category:
                by_category[cat] = {'yes': 0, 'no': 0}
            by_category[cat][record['rating']] += 1
        analytics['by_category'] = by_category

        # Аналитика по датам (по дням)
        by_date = {}
        for record in records:
            date_str = record['created_at'].split('T')[0]
            if date_str not in by_date:
                by_date[date_str] = {'yes': 0, 'no': 0}
            by_date[date_str][record['rating']] += 1
        analytics['by_date'] = by_date

        # Аналитика по версиям БЗ
        by_kb = {}
        for record in records:
            kb_ver = record.get('knowledge_base_version', 'unknown')
            if kb_ver not in by_kb:
                by_kb[kb_ver] = {'yes': 0, 'no': 0}
            by_kb[kb_ver][record['rating']] += 1
        analytics['by_kb_version'] = by_kb

        return analytics


# Глобальный экземпляр
feedback_manager = FeedbackManager()