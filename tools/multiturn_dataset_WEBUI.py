from ragas.dataset_schema import MultiTurnSample
from ragas.messages import HumanMessage

def get_evaluation_samples():
    return [
        # 1. Java задача с оптимизацией
        MultiTurnSample(
            user_input=[
                HumanMessage(content="Напиши Java-метод для проверки палиндрома с использованием рекурсии"),
                HumanMessage(content="Как можно оптимизировать этот алгоритм по памяти?")
            ]
        ),
        
        # 2. Генерация BRD для системы аутентификации
        MultiTurnSample(
            user_input=[
                HumanMessage(content="Создай структуру BRD для системы двухфакторной аутентификации"),
                HumanMessage(content="Добавь раздел с требованиями к аудиту безопасности")
            ]
        ),
        
        # 3. PlantUML диаграмма классов
        MultiTurnSample(
            user_input=[
                HumanMessage(content="Нарисуй в PlantUML диаграмму классов для паттерна Наблюдатель"),
                HumanMessage(content="Добавь методы для управления подписками")
            ]
        ),
        
        # 4. SQL-оптимизация
        MultiTurnSample(
            user_input=[
                HumanMessage(content="Напиши SQL-запрос для поиска дубликатов в таблице users"),
                HumanMessage(content="Как улучшить производительность этого запроса на больших данных?")
            ]
        ),
        
        # 5. Docker конфигурация
        MultiTurnSample(
            user_input=[
                HumanMessage(content="Создай Dockerfile для Spring Boot приложения с Java 17"),
                HumanMessage(content="Добавь конфигурацию для multi-stage build")
            ]
        ),
        
        # 6. REST API документация
        MultiTurnSample(
            user_input=[
                HumanMessage(content="Опиши REST API для сервиса управления задачами в формате OpenAPI"),
                HumanMessage(content="Добавь примеры ответов с ошибками валидации")
            ]
        ),
        
        # 7. Тест кейсы для платежного модуля
        MultiTurnSample(
            user_input=[
                HumanMessage(content="Сгенерируй тест-кейсы для проверки обработки платежей"),
                HumanMessage(content="Добавь негативные сценарии для ошибочных CVV")
            ]
        ),
        
        # 8. Рефакторинг Python кода
        MultiTurnSample(
            user_input=[
                HumanMessage(content="Предложи рефакторинг этого Python-кода с заменой циклов на списковые включения"),
                HumanMessage(content="Как улучшить обработку исключений в этом коде?")
            ]
        ),
        
        # 9. CI/CD конвейер
        MultiTurnSample(
            user_input=[
                HumanMessage(content="Опиши этапы CI/CD пайплайна для микросервиса на Kotlin"),
                HumanMessage(content="Добавь шаг для проверки уязвимостей в зависимостях")
            ]
        ),
        
        # 10. Архитектурная диаграмма
        MultiTurnSample(
            user_input=[
                HumanMessage(content="Нарисуй в PlantUML диаграмму компонентов для e-commerce системы"),
                HumanMessage(content="Добавь сервис кэширования Redis между компонентами")
            ]
        ),
        
#         # 11. Шаблон политики безопасности
#         MultiTurnSample(
#             user_input=[
#                 HumanMessage(content="Создай шаблон политики безопасности для API"),
#                 HumanMessage(content="Добавь раздел по обработке PII данных")
#             ]
#         ),
        
#         # 12. JavaScript оптимизация
#         MultiTurnSample(
#             user_input=[
#                 HumanMessage(content="Напиши функцию дебаунсинга на TypeScript"),
#                 HumanMessage(content="Как изменить её для поддержки leading edge выполнения?")
#             ]
#         ),
        
#         # 13. Миграция базы данных
#         MultiTurnSample(
#             user_input=[
#                 HumanMessage(content="Составь план миграции PostgreSQL с 12 на 15 версию"),
#                 HumanMessage(content="Добавь шаги для отката изменений при ошибке")
#             ]
#         ),
        
#         # 14. Код ревью
#         MultiTurnSample(
#             user_input=[
#                 HumanMessage(content="Проанализируй этот Java-код на предмет проблем с многопоточностью"),
#                 HumanMessage(content="Предложи альтернативу использованию synchronized блоков")
#             ]
#         ),
#         # 15. UML последовательности
#         MultiTurnSample(
#             user_input=[
#                 HumanMessage(content="Нарисуй диаграмму последовательности для процесса оплаты через Google Pay"),
#                 HumanMessage(content="Добавь обработку ошибки недостаточного баланса")
#             ]
#         ),
#                 # 1. Оптимизация рекурсивного палиндрома (Java)
#         MultiTurnSample(
#             user_input=[
#                 HumanMessage(content="""
# Нужно реализовать проверку палиндрома рекурсивно на Java. 
# Требования:
# - Игнорировать регистр и небуквенные символы
# - Сигнатура: public static boolean isPalindrome(String s)
#                 """),
#                 HumanMessage(content="""
# Для строк длиной 10^5 символов получаем StackOverflowError. 
# Как переделать алгоритм с использованием оптимизации хвостовой рекурсии 
# или другим способом избежать переполнения стека?
#                 """)
#             ]
#         ),
#         # 2. BRD для 2FA (системный анализ)
#         MultiTurnSample(
#             user_input=[
#                 HumanMessage(content="""
# Создай структуру BRD для системы двухфакторной аутентификации в банковском приложении.
# Особое внимание:
# - Поддержка SMS, TOTP и push-уведомлений
# - Совместимость с PCI DSS
# - Лимит попыток ввода кода
#                 """),
#                 HumanMessage(content="""
# Добавь нефункциональные требования:
# - Производительность: обработка 1000 запросов/сек
# - Аудит: логирование всех попыток аутентификации 
# - Совместимость с существующей AD-инфраструктурой
#                 """)
#             ]
#         ),
#         # 3. PlantUML для паттерна Наблюдатель (системный дизайн)
#         MultiTurnSample(
#             user_input=[
#                 HumanMessage(content="""
# Создай диаграмму классов для реализации паттерна Observer в системе нотификаций:
# - Интерфейсы Subject и Observer
# - Конкретные классы: EmailNotifier, SMSNotifier
# - Метод sendNotification(String message)
#                 """),
#                 HumanMessage(content="""
# Добавь механизм подписки на конкретные типы событий:
# - Регистрация метода subscribe(EventType type, Observer obs)
# - Фильтрация уведомлений по типу события
#                 """)
#             ]
#         ),
#         # 4. Оптимизация SQL для дубликатов (DBA)
#         MultiTurnSample(
#             user_input=[
#                 HumanMessage(content="""
# Найди дубликаты в таблице users:
# - Поля: email (unique), phone, created_at
# - Критерий: дубль при совпадении email или phone
# - Учитывать мягкое удаление (is_deleted = false)
#                 """),
#                 HumanMessage(content="""
# На продакшене 50M записей. Запрос выполняется 12 минут. 
# Предложи оптимизацию:
# - Создание покрывающих индексов
# - Партиционирование
# - Возможность использования MATERIALIZED VIEW
#                 """)
#             ]
#         ),
#         # 5. Docker для Spring Boot (DevOps)
#         MultiTurnSample(
#             user_input=[
#                 HumanMessage(content="""
# Напиши Dockerfile для Spring Boot 3.x приложения с:
# - Использованием Gradle
# - Java 17
# - Экспозицией порта 8080
# - JVM параметрами: -Xmx512m -XX:+UseZGC
#                 """),
#                 HumanMessage(content="""
# Добавь multi-stage build:
# - Этап сборки с кэшированием зависимостей
# - Итоговый образ на основе amazoncorretto:17-alpine
# - Удаление ненужных файлов (например, документации JDK)
#                 """)
#             ]
#         ),
#         # 6. OpenAPI для задач (API Design)
#         MultiTurnSample(
#             user_input=[
#                 HumanMessage(content="""
# Опиши в OpenAPI 3.0 эндпоинты для управления задачами:
# - CRUD операций
# - Фильтрация по статусу (OPEN, IN_PROGRESS, DONE)
# - Пагинация с page/size параметрами
#                 """),
#                 HumanMessage(content="""
# Добавь примеры ответов для:
# - 400: Невалидные значения статуса
# - 429: Превышен лимит запросов
# - 500: Общая схема ошибки с traceId
#                 """)
    # ])
]
