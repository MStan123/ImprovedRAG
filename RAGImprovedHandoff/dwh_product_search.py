import pandas as pd
import pyarrow.parquet as pq
import redis
import json
import hashlib
from typing import List, Dict, Optional
from pathlib import Path
from datetime import datetime
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient
from logger_setup import setup_logger

logger = setup_logger()


class DWHCache:
    """Кэширование результатов поиска в Redis"""

    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self.ttl = 1800  # 30 минут
        self.prefix = "dwh_product_cache:"

    def get_cache_key(self, query: str, filters: dict) -> str:
        """Генерирует ключ кэша на основе запроса и фильтров"""
        data = f"{query}_{json.dumps(filters, sort_keys=True)}"
        hash_key = hashlib.md5(data.encode()).hexdigest()
        return f"{self.prefix}{hash_key}"

    def get(self, query: str, filters: dict) -> Optional[List[Dict]]:
        """Получает закэшированные результаты"""
        key = self.get_cache_key(query, filters)
        try:
            cached = self.redis.get(key)
            if cached:
                logger.info(f"✅ DWH Cache HIT for query: {query[:50]}")
                return json.loads(cached)
        except Exception as e:
            logger.error(f"Cache get error: {e}")
        return None

    def set(self, query: str, filters: dict, results: List[Dict]):
        """Кэширует результаты"""
        key = self.get_cache_key(query, filters)
        try:
            self.redis.setex(key, self.ttl, json.dumps(results, default=str))
            logger.info(f"💾 Results cached for query: {query[:50]}")
        except Exception as e:
            logger.error(f"Cache set error: {e}")


class AzureParquetLoader:
    """Загрузчик Parquet файлов из Azure Storage"""

    def __init__(
            self,
            storage_account_name: str,
            container_name: str,
            blob_path: str,
            use_managed_identity: bool = True,
            connection_string: Optional[str] = None
    ):
        self.storage_account_name = storage_account_name
        self.container_name = container_name
        self.blob_path = blob_path
        self.local_cache_path = Path("/tmp/dwh_products_cache.parquet")
        self.df = None
        self.last_loaded = None

        # Инициализация Azure Blob Client
        if use_managed_identity:
            # Managed Identity (для production в Azure)
            credential = DefaultAzureCredential()
            account_url = f"https://{storage_account_name}.blob.core.windows.net"
            self.blob_service_client = BlobServiceClient(account_url, credential=credential)
        elif connection_string:
            # Connection String (для локальной разработки)
            self.blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        else:
            raise ValueError("Укажите use_managed_identity=True или connection_string")

        logger.info(f"🔗 Azure Blob initialized: {storage_account_name}/{container_name}")

    def download_parquet(self, force_reload: bool = False) -> pd.DataFrame:
        """
        Скачивает Parquet файл из Azure Storage

        Args:
            force_reload: принудительно перезагрузить файл

        Returns:
            DataFrame с данными товаров
        """
        # Проверяем нужно ли перезагружать
        if not force_reload and self.df is not None:
            # Если данные загружены меньше 10 минут назад - используем кэш
            if self.last_loaded and (datetime.now() - self.last_loaded).seconds < 600:
                logger.info("📦 Using in-memory DataFrame cache")
                return self.df

        try:
            logger.info(f"📥 Downloading Parquet from Azure: {self.blob_path}")

            # Получаем blob client
            blob_client = self.blob_service_client.get_blob_client(
                container=self.container_name,
                blob=self.blob_path
            )

            # Скачиваем во временный файл
            with open(self.local_cache_path, "wb") as download_file:
                download_stream = blob_client.download_blob()
                download_file.write(download_stream.readall())

            # Читаем Parquet в DataFrame
            self.df = pd.read_parquet(self.local_cache_path)
            self.last_loaded = datetime.now()

            logger.info(f"✅ Loaded {len(self.df)} products from Parquet")
            logger.info(f"📊 Columns: {list(self.df.columns)}")

            return self.df

        except Exception as e:
            logger.error(f"❌ Failed to download Parquet: {e}")
            # Если есть старый локальный файл - используем его
            if self.local_cache_path.exists():
                logger.warning("⚠️ Using stale local cache")
                self.df = pd.read_parquet(self.local_cache_path)
                return self.df
            raise

    def get_dataframe(self) -> pd.DataFrame:
        """Возвращает актуальный DataFrame"""
        if self.df is None:
            return self.download_parquet()
        return self.df


class DWHProductSearch:
    """Класс для поиска товаров в Parquet данных из ADF"""

    def __init__(
            self,
            storage_account_name: str,
            container_name: str,
            blob_path: str,
            use_managed_identity: bool = True,
            connection_string: Optional[str] = None,
            redis_client: Optional[redis.Redis] = None
    ):
        self.loader = AzureParquetLoader(
            storage_account_name=storage_account_name,
            container_name=container_name,
            blob_path=blob_path,
            use_managed_identity=use_managed_identity,
            connection_string=connection_string
        )
        self.cache = DWHCache(redis_client) if redis_client else None

        # Загружаем данные при инициализации
        try:
            self.loader.download_parquet()
        except Exception as e:
            logger.warning(f"⚠️ Initial load failed: {e}. Will retry on first search.")

    def reload_data(self):
        """Принудительно перезагружает данные из Azure"""
        logger.info("🔄 Force reloading data from Azure...")
        self.loader.download_parquet(force_reload=True)

    def search_products(
            self,
            query: Optional[str] = None,
            product_id: Optional[int] = None,
            gtin: Optional[str] = None,
            only_in_stock: bool = True,
            min_price: Optional[float] = None,
            max_price: Optional[float] = None
    ) -> List[Dict]:
        """
        Поиск товаров в DataFrame с фильтрацией

        Args:
            query: текст для поиска в названии товара
            product_id: конкретный ID товара
            gtin: штрих-код товара
            only_in_stock: только товары в наличии
            min_price: минимальная цена
            max_price: максимальная цена
            top_n: максимальное количество результатов

        Returns:
            Список словарей с информацией о товарах
        """
        # Проверяем кэш
        if self.cache and query:
            filters = {
                'only_in_stock': only_in_stock,
                'min_price': min_price,
                'max_price': max_price,
            }
            cached_results = self.cache.get(query, filters)
            if cached_results:
                return cached_results

        try:
            # Получаем DataFrame
            df = self.loader.get_dataframe()

            if df is None or df.empty:
                logger.warning("⚠️ DataFrame is empty")
                return []

            # Создаем копию для фильтрации
            filtered_df = df.copy()

            # Фильтр по ID товара
            if product_id is not None:
                filtered_df = filtered_df[filtered_df['MpProductID'] == product_id]

            # Фильтр по GTIN
            if gtin is not None:
                # Предполагаем что есть колонка Gtin
                if 'Gtin' in filtered_df.columns:
                    filtered_df = filtered_df[filtered_df['Gtin'] == gtin]

            # Фильтр по наличию
            if only_in_stock:
                filtered_df = filtered_df[filtered_df['Qty'] > 0]

            # Фильтр по цене
            if min_price is not None:
                filtered_df = filtered_df[filtered_df['RetailPrice'] >= min_price]

            if max_price is not None:
                filtered_df = filtered_df[filtered_df['RetailPrice'] <= max_price]

            # Поиск по тексту в названии
            if query is not None and query.strip():
                query_lower = query.lower()

                # УЛУЧШЕННЫЙ ПОИСК: Извлекаем ключевые слова из запроса
                # Убираем стоп-слова и вопросительные конструкции
                stop_words = [
                    'какова', 'какая', 'какой', 'какие', 'сколько', 'стоит', 'стоят',
                    'цена', 'qiymət', 'price', 'cost', 'в', 'на', 'из', 'для',
                    'каталог', 'каталоге', 'catalog', 'есть', 'ли', 'var', 'mi',
                    'neçə', 'nə', 'qədər', 'qədərdir', 'how', 'much', 'what', 'is'
                ]

                # Разбиваем запрос на слова и убираем стоп-слова
                query_words = [
                    word for word in query_lower.split()
                    if word not in stop_words and len(word) > 2
                ]

                # Если остались ключевые слова - ищем их
                if query_words:
                    # Создаем маску: товар должен содержать хотя бы одно ключевое слово
                    mask = pd.Series([False] * len(filtered_df))

                    for word in query_words:
                        # Ищем каждое слово в названии
                        word_mask = filtered_df['Name'].str.lower().str.contains(word, na=False, regex=False)
                        mask = mask | word_mask

                        # Также ищем в Description если есть
                        if 'Description' in filtered_df.columns:
                            desc_mask = filtered_df['Description'].str.lower().str.contains(word, na=False, regex=False)
                            mask = mask | desc_mask

                    filtered_df = filtered_df[mask]
                else:
                    # Если после фильтрации не осталось слов - ищем по исходному запросу
                    mask = filtered_df['Name'].str.lower().str.contains(query_lower, na=False)

                    if 'Description' in filtered_df.columns:
                        mask |= filtered_df['Description'].str.lower().str.contains(query_lower, na=False)

                    filtered_df = filtered_df[mask]

            # Сортировка: сначала по количеству (больше = лучше), потом по цене (меньше = лучше)
            filtered_df = filtered_df.sort_values(
                by=['Qty', 'RetailPrice'],
                ascending=[False, True]
            )

            # FALLBACK: Если ничего не найдено, попробуем более агрессивный поиск
            if len(filtered_df) == 0 and query is not None:
                logger.info(f"⚠️ No results with keyword search, trying fallback...")

                # Попробуем искать по частичным совпадениям (первые 3 буквы каждого слова)
                df_original = self.loader.get_dataframe()

                # Применяем только фильтры (без текстового поиска)
                if only_in_stock:
                    df_original = df_original[df_original['Qty'] > 0]
                if min_price is not None:
                    df_original = df_original[df_original['RetailPrice'] >= min_price]
                if max_price is not None:
                    df_original = df_original[df_original['RetailPrice'] <= max_price]

                # Ищем по первым буквам (для опечаток типа "айфон" вместо "iPhone")
                query_words = query.lower().split()
                mask = pd.Series([False] * len(df_original))

                for word in query_words:
                    if len(word) >= 3:
                        # Ищем слова начинающиеся с этих букв
                        pattern = word[:3]
                        word_mask = df_original['Name'].str.lower().str.contains(pattern, na=False, regex=False)
                        mask = mask | word_mask

                if mask.any():
                    filtered_df = df_original[mask].sort_values(
                        by=['Qty', 'RetailPrice'],
                        ascending=[False, True]
                    )
                    logger.info(f"✅ Fallback found {len(filtered_df)} results")

            # Конвертируем в список словарей
            results = filtered_df.to_dict('records')

            logger.info(f"🔍 Search for '{query}': found {len(results)} products")

            # Кэшируем результаты
            if self.cache and query:
                self.cache.set(query, filters, results)

            return results

        except Exception as e:
            logger.error(f"❌ Search error: {e}")
            return []

    def get_product_by_id(self, product_id: int) -> Optional[Dict]:
        """Получить конкретный товар по ID"""
        results = self.search_products(product_id=product_id, only_in_stock=False)
        return results[0] if results else None

    def format_products_for_llm(self, products: List[Dict], language: str = 'ru', max_display: int = None) -> str:
        """
        Форматирует результаты поиска для передачи в LLM

        Args:
            products: список товаров
            language: язык форматирования (ru, az, en)
            max_display: максимальное количество товаров для отображения (None = все)

        Returns:
            Отформатированная строка с информацией о товарах
        """
        if not products:
            return ""

        # Заголовки и шаблоны для разных языков
        templates = {
            'ru': {
                'header': "📦 Найденные товары из каталога:\n\n",
                'template': """• {Name}
  💰 Цена: {RetailPrice} AZN
  🏪 Продавец: {MerchantMarketingName}
  {installment}
  ID: {MpProductID}
""",
                'installment_yes': "💳 Рассрочка: до {months} месяцев",
                'installment_no': "💳 Рассрочка: не доступна",
                'more': "...и ещё {count} товаров\n"
            },
            'az': {
                'header': "📦 Kataloqdan tapılan məhsullar:\n\n",
                'template': """• {Name}
  💰 Qiymət: {RetailPrice} AZN
  🏪 Satıcı: {MerchantMarketingName}
  {installment}
  ID: {MpProductID}
""",
                'installment_yes': "💳 Taksit: {months} aya qədər",
                'installment_no': "💳 Taksit: mövcud deyil",
                'more': "...və daha {count} məhsul\n"
            },
            'en': {
                'header': "📦 Products found in catalog:\n\n",
                'template': """• {Name}
  💰 Price: {RetailPrice} AZN
  🏪 Seller: {MerchantMarketingName}
  {installment}
  ID: {MpProductID}
""",
                'installment_yes': "💳 Installment: up to {months} months",
                'installment_no': "💳 Installment: not available",
                'more': "...and {count} more products\n"
            }
        }

        # Выбираем шаблон или используем русский по умолчанию
        lang_template = templates.get(language, templates['ru'])

        formatted = lang_template['header']

        # Определяем сколько товаров показывать
        if max_display is None:
            # Умная логика: показываем больше если нашлось мало
            if len(products) <= 3:
                display_count = len(products)
            elif len(products) <= 10:
                display_count = len(products)  # Показываем все до 10
            else:
                display_count = 10  # Максимум 10
        else:
            display_count = min(max_display, len(products))

        for product in products[:display_count]:
            # Форматирование информации о рассрочке
            installment = (
                lang_template['installment_yes'].format(
                    months=product.get('MaxInstallmentMonths', 0)
                )
                if product.get('InstallmentEnabled')
                else lang_template['installment_no']
            )

            # Добавляем товар
            formatted += lang_template['template'].format(
                Name=product.get('Name', 'N/A'),
                RetailPrice=product.get('RetailPrice', 'N/A'),
                MerchantMarketingName=product.get('MerchantMarketingName', 'N/A'),
                installment=installment,
                MpProductID=product.get('MpProductID', 'N/A')
            )
            formatted += "\n"

        # Если товаров больше чем показали, указываем сколько ещё
        if len(products) > display_count:
            formatted += lang_template['more'].format(count=len(products) - display_count)

        return formatted

    def search_by_brand(self, brand: str, **kwargs) -> List[Dict]:
        """Поиск товаров конкретного бренда"""
        return self.search_products(query=brand, **kwargs)

    def search_in_price_range(
            self,
            min_price: float,
            max_price: float,
            category_query: Optional[str] = None,
            **kwargs
    ) -> List[Dict]:
        """Поиск товаров в ценовом диапазоне"""
        return self.search_products(
            query=category_query,
            min_price=min_price,
            max_price=max_price,
            **kwargs
        )

    def debug_search(self, query: str) -> Dict:
        """
        Отладочная функция для диагностики поиска
        """
        df = self.loader.get_dataframe()

        if df is None or df.empty:
            return {"error": "DataFrame is empty"}

        query_lower = query.lower()

        # Проверяем что есть в базе
        sample_names = df['Name'].head(10).tolist() if 'Name' in df.columns else []

        # Извлекаем ключевые слова
        stop_words = [
            'какова', 'какая', 'какой', 'какие', 'сколько', 'стоит', 'стоят',
            'цена', 'qiymət', 'price', 'cost', 'в', 'на', 'из', 'для',
            'каталог', 'каталоге', 'catalog', 'есть', 'ли', 'var', 'mi'
        ]
        query_words = [
            word for word in query_lower.split()
            if word not in stop_words and len(word) > 2
        ]

        # Проверяем сколько товаров содержат каждое слово
        word_matches = {}
        for word in query_words:
            count = df['Name'].str.lower().str.contains(word, na=False).sum()
            word_matches[word] = count

        return {
            "query": query,
            "extracted_keywords": query_words,
            "word_matches": word_matches,
            "total_products": len(df),
            "sample_names": sample_names
        }

    def get_statistics(self) -> Dict:
        """Получить статистику по данным"""
        try:
            df = self.loader.get_dataframe()
            if df is None or df.empty:
                return {}

            return {
                'total_products': len(df),
                'in_stock_products': len(df[df['Qty'] > 0]),
                'avg_price': float(df['RetailPrice'].mean()),
                'min_price': float(df['RetailPrice'].min()),
                'max_price': float(df['RetailPrice'].max()),
                'last_updated': self.loader.last_loaded.isoformat() if self.loader.last_loaded else None
            }
        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            return {}


# ==================== ГЛОБАЛЬНЫЙ ЭКЗЕМПЛЯР ====================

# Загрузка конфигурации из переменных окружения
import os
from dotenv import load_dotenv

load_dotenv()

# Azure Storage настройки
STORAGE_ACCOUNT_NAME = os.getenv('AZURE_STORAGE_ACCOUNT_NAME', 'your_storage_account')
CONTAINER_NAME = os.getenv('AZURE_CONTAINER_NAME', 'birmarket-data')
BLOB_PATH = os.getenv('AZURE_BLOB_PATH', 'dwh/products/latest.parquet')

# Для локальной разработки можно использовать connection string
STORAGE_CONNECTION_STRING = os.getenv('AZURE_STORAGE_CONNECTION_STRING')

# Используем Managed Identity в production, connection string в dev
USE_MANAGED_IDENTITY = os.getenv('USE_MANAGED_IDENTITY', 'true').lower() == 'true'

# Инициализация Redis для кэширования (опционально)
try:
    redis_client = redis.Redis(
        host=os.getenv('REDIS_HOST', 'localhost'),
        port=int(os.getenv('REDIS_PORT', 6379)),
        db=int(os.getenv('REDIS_DB', 0)),
        decode_responses=True
    )
    redis_client.ping()
    logger.info("✅ Redis connected for caching")
except Exception as e:
    logger.warning(f"⚠️ Redis not available: {e}")
    redis_client = None

# Создаем глобальный экземпляр
dwh_search = DWHProductSearch(
    storage_account_name=STORAGE_ACCOUNT_NAME,
    container_name=CONTAINER_NAME,
    blob_path=BLOB_PATH,
    use_managed_identity=USE_MANAGED_IDENTITY,
    connection_string=STORAGE_CONNECTION_STRING if not USE_MANAGED_IDENTITY else None,
    redis_client=redis_client
)

__all__ = ['dwh_search', 'DWHProductSearch']