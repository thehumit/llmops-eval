from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings

from datasets import load_dataset

from ragas import SingleTurnSample
from ragas.metrics import AspectCritic

from ragas.metrics import AspectCritic, SimpleCriteriaScore
from ragas.dataset_schema import MultiTurnSample, EvaluationDataset
from ragas.messages import HumanMessage, AIMessage
from ragas import evaluate


import asyncio  # Import asyncio for asynchronous operations


import logging
# Set up detailed logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configure logging for ragas and related libraries
logging.getLogger("ragas").setLevel(logging.DEBUG)
logging.getLogger("langchain").setLevel(logging.DEBUG)

async def evaluate_metric(test_data, evaluator_llm):
    metric = AspectCritic(name="summary_accuracy", llm=evaluator_llm, definition="Verify if the summary is accurate.")
    test_data = SingleTurnSample(**test_data)
    result = metric.single_turn_ascore(test_data)  # Await the coroutine
    return result

# Call the asynchronous function
if __name__ == "__main__":
    evaluator_llm = LangchainLLMWrapper(ChatOpenAI(api_key="None", base_url=f"http://10.148.80.14:30001/v1", streaming=False))
    
    sample1 = MultiTurnSample(
        user_input=[
            HumanMessage(
                content="Мне нужно увеличить кредитный лимит и проверить, почему отклонена последняя транзакция в Ленте."
            ),
            AIMessage(
                content="Давайте сначала проверим отклонённую транзакцию. Покупка на 5,842 рубля в Ленте 15 марта была отклонена из-за недостатка средств."
            ),
            HumanMessage(content="Это невозможно, у меня было достаточно денег на счете."),
            AIMessage(
                content="Понимаю ваше недоумение. Проверя детали... Вижу, что при текущем балансе 25,000 рублей было заблокировано 20,000 рублей по предварительному бронированию авиабилетов, что оставило вам только 5,000 рублей доступных средств."
            ),
            HumanMessage(content="Ах да, я же бронировал билеты!"),
            AIMessage(content="Могу ли я помочь вам ещё чем-то?"),
        ]
    )
    
    sample2 = MultiTurnSample(
        user_input=[
            HumanMessage(
                content="Почему моя транзакция в Азбуке Вкуса была отклонена?"
            ),
            AIMessage(
                content="Какую сумму вы пытались оплатить? Прошу уточнить для проверки."  # Модель "забыла" и переспрашивает
            ),
            HumanMessage(content="3,450 рублей, 10 апреля."),
            AIMessage(
                content="Спасибо за уточнение. Эта транзакция была отклонена системой безопасности, так как она отличалась от вашего обычного паттерна покупок."
            ),
            HumanMessage(content="Но это был подарок на день рождения!"),
            AIMessage(
                content="Поняла. Разблокирую карту для будущих транзакций. Рекомендую заранее уведомлять банк о нестандартных платежах."
            ),
        ]
    )
    
    sample3 = MultiTurnSample(
        user_input=[
            HumanMessage(
                content="Хочу разобраться с комиссией за международный перевод и проверить баланс."
            ),
            AIMessage(
                content="Ваш текущий баланс составляет 78,300 рублей. Международный перевод в размере 50,000 рублей будет облагаться комиссией 1.5%."
            ),
            HumanMessage(content="Почему так много? В прошлый раз было 1%."),
            AIMessage(
                content="Извините, не могли бы вы повторить сумму перевода? Я что-то запуталась."  # Модель "забыла"
            ),
            HumanMessage(content="50,000 рублей, как я уже сказал."),
            AIMessage(
                content="Благодарю за терпение. Да, комиссия действительно увеличилась с 1% до 1.5% с 1 марта согласно новому тарифу."
            ),
        ]
    )
    
    sample4 = MultiTurnSample(
        user_input=[
            HumanMessage(
                content="Моя карта была заблокирована после оплаты в Steam, помогите разобраться."
            ),
            AIMessage(
                content="Да, вижу блокировку 12 апреля после платежа в Steam на 2,399 рублей. Наша система безопасности сработала из-за подозрительной активности."
            ),
            HumanMessage(content="Но это был мой обычный платёж за игру!"),
            AIMessage(
                content="Понимаю. Просто в этот раз платеж был сделан через VPN из Германии, хотя обычно вы платите из России. Это вызвало автоматическую блокировку."
            ),
            HumanMessage(content="Да, был в командировке и использовал VPN."),
            AIMessage(content="Разблокирую вашу карту. В будущем советую уведомлять банк о поездках."),
        ]
    )
    
    sample5 = MultiTurnSample(
        user_input=[
            HumanMessage(
                content="Почему я не могу оплатить подписку на Яндекс Плюс?"
            ),
            AIMessage(
                content="Какую карту вы используете? У меня нет данных о предыдущих попытках."  # Модель "забыла"
            ),
            HumanMessage(content="Виртуальную карту *6437, как всегда."),
            AIMessage(
                content="Спасибо. Вижу проблему - на этой карте установлен лимит 1,000 рублей для онлайн-платежей, а подписка стоит 1,199 рублей."
            ),
            HumanMessage(content="Точно! Спасибо, что напомнили про лимит."),
        ]
    )
    
    sample6 = MultiTurnSample(
        user_input=[
            HumanMessage(
                content="Не могу понять, почему не проходит оплата в такси через приложение."
            ),
            AIMessage(
                content="Проверя... Ваш последний платеж в 450 рублей был отклонен из-за истекшего срока действия карты."
            ),
            HumanMessage(content="Но я же обновил карту месяц назад!"),
            AIMessage(
                content="Извините, вы говорите про мобильное приложение или веб-версию? Я потеряла контекст."  # Модель "забыла"
            ),
            HumanMessage(content="Приложение Яндекс.Такси на iPhone."),
            AIMessage(
                content="Спасибо. Да, вижу проблему - в приложении сохранены данные старой карты. Нужно удалить её и добавить новую в настройках платежей."
            ),
        ]
    )
    
    sample7 = MultiTurnSample(
        user_input=[
            HumanMessage(
                content="Почему сняли 500 рублей за обслуживание счета? В тарифе такого нет."
            ),
            AIMessage(
                content="Это комиссия за SMS-уведомления. С 1 апреля изменились условия вашего тарифного плана."
            ),
            HumanMessage(content="Но я же отключил SMS месяц назад!"),
            AIMessage(
                content="Проверяю... Действительно, моя ошибка. Ваши SMS-уведомления были отключены 15 марта. Верну 500 рублей в течение 3 дней."
            ),
            HumanMessage(content="Спасибо, жду возврата."),
        ]
    )
    
    sample8 = MultiTurnSample(
        user_input=[
            HumanMessage(
                content="Мне нужно подключить автоплатеж за коммуналку и увеличить лимит на снятие наличных."
            ),
            AIMessage(
                content="Давайте начнем с автоплатежа. За какое именно ЖКУ вы хотите настроить оплату?"  # Частичное "забывание"
            ),
            HumanMessage(content="За электричество, компания МосЭнергоСбыт."),
            AIMessage(
                content="Хорошо. А по поводу лимита - текущий дневной лимит 50,000 рублей можно увеличить до 100,000. Нужно ли это сделать?"
            ),
            HumanMessage(content="Да, увеличьте до 100,000 и настройте автоплатеж."),
            AIMessage(content="Готово! Оба изменения вступят в силу с завтрашнего дня."),
        ]
    )
    
    sample9 = MultiTurnSample(
        user_input=[
            HumanMessage(
                content="Почему я не могу перевести деньги своему брату через СБП?"
            ),
            AIMessage(
                content="Переводы через СБП ограничены 100,000 рублей в сутки. Какую сумму вы пытались перевести?"
            ),
            HumanMessage(content="Всего 15,000 рублей, но пишет ошибку."),
            AIMessage(
                content="Прошу прощения, вы упомянули СБП или другой сервис? Я запуталась в деталях."  # Модель "забыла"
            ),
            HumanMessage(content="Система быстрых платежей, через мобильное приложение."),
            AIMessage(
                content="Спасибо. Вижу проблему - получатель не подключил СБП в своём банке. Попробуйте обычный банковский перевод."
            ),
        ]
    )
    
    sample10 = MultiTurnSample(
        user_input=[
            HumanMessage(
                content="Моя карта не работает за границей, хотя я предупреждал банк о поездке."
            ),
            AIMessage(
                content="Да, вижу ваше уведомление о поездке в Турцию с 1 по 15 апреля. Где именно не работает карта?"
            ),
            HumanMessage(content="В ресторане в Стамбуле, платеж на 1,200 лир."),
            AIMessage(
                content="Проблема в том, что у вас установлен лимит 5,000 рублей на платежи за границей, а 1,200 лир это около 6,000 рублей по текущему курсу."
            ),
            HumanMessage(content="Ах вот в чем дело! Можно увеличить лимит?"),
            AIMessage(content="Да, временно увеличу до 15,000 рублей на время поездки."),
        ]
    )
    
    sample11 = MultiTurnSample(
        user_input=[
            HumanMessage(
                content="Почему с моего счета списали 3,000 рублей с пометкой 'страховка'?"
            ),
            AIMessage(
                content="Это автоматическое продление страховки путешественника, которую вы подключили в прошлом году."
            ),
            HumanMessage(content="Но я же отменял её перед поездкой в Сочи!"),
            AIMessage(
                content="Извините, не могли бы вы уточнить дату поездки? Я не вижу полной истории."  # Модель "забыла"
            ),
            HumanMessage(content="Поездка была с 10 по 20 августа 2023 года."),
            AIMessage(
                content="Спасибо. Действительно, страховка была отменена 9 августа. Это ошибочное списание - верну деньги в течение 5 дней."
            ),
        ]
    )
    
    sample12 = MultiTurnSample(
        user_input=[
            HumanMessage(
                content="Как мне получить выписку за прошлый год и увеличить лимит по кредитной карте?"
            ),
            AIMessage(
                content="Выписку за 2023 год я могу отправить на вашу почту. А по кредитной карте - текущий лимит 150,000 можно увеличить до 200,000."
            ),
            HumanMessage(content="Отправьте выписку и увеличьте лимит, пожалуйста."),
            AIMessage(
                content="Простите, вы хотели выписку по какому именно счету? У вас их несколько."  # Модель "забыла"
            ),
            HumanMessage(content="По основному рублевому счету, который заканчивается на 4572."),
            AIMessage(content="Готово! Выписка отправлена, лимит по карте увеличен."),
        ]
    )
    
    sample13 = MultiTurnSample(
        user_input=[
            HumanMessage(
                content="Мне не приходит код подтверждения для входа в мобильный банк."
            ),
            AIMessage(
                content="Проблема может быть в перебоях у оператора. Какой у вас телефонный номер привязан к счету?"
            ),
            HumanMessage(content="+7 916 123-45-67, но проблема не в этом."),
            AIMessage(
                content="Проверяю... Вижу, что вы отключили SMS-коды и подключили push-уведомления. Проверьте настройки уведомлений в приложении."
            ),
            HumanMessage(content="Действительно, спасибо!"),
        ]
    )
    
    sample14 = MultiTurnSample(
        user_input=[
            HumanMessage(
                content="Почему у меня два списания по 5,000 рублей с пометкой 'Apple'?"
            ),
            AIMessage(
                content="Это подписки на Apple Music и iCloud+, которые продлились автоматически."
            ),
            HumanMessage(content="Но я отписывался от них в прошлом месяце!"),
            AIMessage(
                content="Извините, вы могли бы сказать, когда именно вы отменяли подписки? Я не вижу этой информации."  # Модель "забыла"
            ),
            HumanMessage(content="Примерно 15 марта, через настройки iPhone."),
            AIMessage(
                content="Спасибо. Вижу ошибку - отмена не дошла до платежной системы. Верну деньги и помогу правильно отменить подписки."
            ),
        ]
    )
    
    sample15 = MultiTurnSample(
        user_input=[
            HumanMessage(
                content="Не могу оплатить авиабилеты, пишет 'превышен лимит'."
            ),
            AIMessage(
                content="Да, ваш дневной лимит на онлайн-платежи составляет 30,000 рублей. Сколько стоят билеты?"
            ),
            HumanMessage(content="42,500 рублей, но это же не ежедневная покупка!"),
            AIMessage(
                content="Понимаю. Временное увеличение лимита возможно. Нужно ли вам разовое повышение до 50,000 рублей?"
            ),
            HumanMessage(content="Да, именно это мне и нужно."),
            AIMessage(content="Лимит увеличен на 24 часа. Удачной поездки!"),
        ]
    )
    
    sample16 = MultiTurnSample(
        user_input=[
            HumanMessage(
                content="Почему я не могу снять деньги в вашем же банкомате?"
            ),
            AIMessage(
                content="Какую сумму вы пытались снять? И какой у вас текущий лимит?"  # Модель переспрашивает
            ),
            HumanMessage(content="40,000 рублей, лимит 50,000 в сутки."),
            AIMessage(
                content="Странно. Проверяю... Оказывается, в этом банкомате установлен собственный лимит 30,000 рублей на одну операцию. Попробуйте снять в два приема или найдите другой банкомат."
            ),
            HumanMessage(content="Спасибо, так и сделаю."),
        ]
    )
    
    sample17 = MultiTurnSample(
        user_input=[
            HumanMessage(
                content="Мне нужно заменить карту и узнать про кэшбэк за последний месяц."
            ),
            AIMessage(
                content="Карту можно заменить в отделении или заказать доставку. Ваш кэшбэк за март составил 1,234 рубля."
            ),
            HumanMessage(content="Почему так мало? В феврале было 3,567!"),
            AIMessage(
                content="Простите, вы спрашивали про дебетовую или кредитную карту? Я перепутала данные."  # Модель "забыла"
            ),
            HumanMessage(content="Дебетовую, которая заканчивается на 5683."),
            AIMessage(
                content="Спасибо. Да, в марте вы меньше тратили в категориях с повышенным кэшбэком - всего 12,000 против 35,000 в феврале."
            ),
        ]
    )
    
    sample18 = MultiTurnSample(
        user_input=[
            HumanMessage(
                content="Почему с моего счета списали 2,000 рублей за перевод?"
            ),
            AIMessage(
                content="Это комиссия за срочный межбанковский перевод в другой регион. Обычный перевод стоил бы 50 рублей, но вы выбрали опцию 'моментальный'."
            ),
            HumanMessage(content="Я такого не выбирал!"),
            AIMessage(
                content="Проверяю историю операций... Действительно, при подтверждении перевода была нажата кнопка 'срочно'. Возможно, это было случайно."
            ),
            HumanMessage(content="Верните деньги, это ошибка!"),
            AIMessage(content="Оформим возврат комиссии в течение 3 рабочих дней."),
        ]
    )
    
    sample19 = MultiTurnSample(
        user_input=[
            HumanMessage(
                content="Как мне отменить овердрафт и проверить остаток по вкладу?"
            ),
            AIMessage(
                content="Овердрафт можно отключить в настройках карты. Остаток по вашему вкладу 'Накопительный' составляет 456,789 рублей."
            ),
            HumanMessage(content="Отлично, а проценты начисляются исправно?"),
            AIMessage(
                content="Простите, я забыла уточнить - вы спрашиваете про рублевый или валютный вклад?"  # Модель "забыла"
            ),
            HumanMessage(content="Рублевый, тот самый 'Накопительный'."),
            AIMessage(
                content="Спасибо. Да, проценты начисляются ежемесячно 5-го числа. В марте начислено 3,456 рублей."
            ),
        ]
    )
    
    sample20 = MultiTurnSample(
        user_input=[
            HumanMessage(
                content="Почему не приходит смс с кодом для подтверждения перевода?"
            ),
            AIMessage(
                content="Проблема может быть на стороне оператора. Какой номер телефона привязан к вашему счету?"
            ),
            HumanMessage(content="+7 925 111-22-33, но проблема не в этом."),
            AIMessage(
                content="Проверяю... Вижу, что вы недавно сменили SIM-карту. Для безопасности SMS-подтверждения временно заблокированы. Нужно подтвердить новый номер в отделении."
            ),
            HumanMessage(content="Понял, зайду в банк завтра."),
        ]
    )
    sample21 = MultiTurnSample(
        user_input=[
            HumanMessage(content="Я хочу оплатить штраф ГИБДД по номеру постановления 782ХТ125."),
            AIMessage(content="Для оплаты штрафа 782ХТ125 потребуется 1,500 рублей. Подтверждаете оплату?"),
            HumanMessage(content="Да, но сначала скажите за что этот штраф."),
            AIMessage(content="Извините, а какое именно постановление вы спрашивали? Я потеряла номер."),  # Забыл
            HumanMessage(content="782ХТ125, как я уже сказал!"),
            AIMessage(content="Спасибо! Штраф выписан 15.04 за превышение скорости на 20-40 км/ч в Москве.")
        ]
    )

    sample22 = MultiTurnSample(
        user_input=[
            HumanMessage(content="Почему с моего счета списали 3,900 рублей с пометкой 'МТС'?"),
            AIMessage(content="Это ежемесячный платеж за ваш тарифный план 'Безлимит+' в МТС."),
            HumanMessage(content="Но я перешёл на тариф за 2,900 рублей месяц назад!"),
            AIMessage(content="Действительно, вижу смену тарифа. Верну излишне списанные 1,000 рублей.")
        ]
    )

    sample23 = MultiTurnSample(
        user_input=[
            HumanMessage(content="Мне нужно заблокировать карту *5678 и заказать новую."),
            AIMessage(content="Карта *5678 заблокирована. Новую доставим за 3 дня. Куда привезти?"),
            HumanMessage(content="По тому же адресу: Москва, ул. Ленина 15, кв 42."),
            AIMessage(content="Простите, вы могли бы повторить номер карты? Я отвлеклась."),  # Забыл
            HumanMessage(content="*5678, как в первом сообщении!"),
            AIMessage(content="Принято! Новая карта приедет по адресу: Ленина 15, кв 42.")
        ]
    )

    sample24 = MultiTurnSample(
        user_input=[
            HumanMessage(content="Почему не работает перевод на карту Сбербанка 2202****3456?"),
            AIMessage(content="Этот номер карты принадлежит Марии Ивановой. Это верный получатель?"),
            HumanMessage(content="Да, это моя жена. Но почему перевод не проходит?"),
            AIMessage(content="Проблема в том, что у вас установлен лимит 15,000 рублей на переводы между банками.")
        ]
    )

    sample25 = MultiTurnSample(
        user_input=[
            HumanMessage(content="Какой у меня остаток по кредиту и когда следующий платеж?"),
            AIMessage(content="Остаток по кредиту: 125,600 рублей. Следующий платеж 25.04 на 12,340 рублей."),
            HumanMessage(content="А сколько будет следующий платеж после этого?"),
            AIMessage(content="Извините, вы спрашиваете про ипотеку или потребительский кредит?"),  # Забыл
            HumanMessage(content="Потребительский, о котором мы только что говорили!"),
            AIMessage(content="Понял! Платеж 25.05 будет 12,200 рублей (уменьшается аннуитет).")
        ]
    )

    sample26 = MultiTurnSample(
        user_input=[
            HumanMessage(content="Почему я не могу войти в мобильный банк с нового телефона?"),
            AIMessage(content="Нужно подтвердить новое устройство. Код отправлен на +7***123-45-67."),
            HumanMessage(content="Это старый номер! Новый +7***987-65-43."),
            AIMessage(content="Обновил номер в системе. Новый код отправлен на +7***987-65-43.")
        ]
    )

    sample27 = MultiTurnSample(
        user_input=[
            HumanMessage(content="Мне нужно подключить автоплатеж за интернет Ростелеком на сумму 890 рублей."),
            AIMessage(content="Автоплатеж на 890 рублей будет списываться 10-го числа. Подтверждаете?"),
            HumanMessage(content="Да, но только с 1 мая, сейчас не нужно."),
            AIMessage(content="Простите, за какой именно сервис вы подключаете автоплатеж?"),  # Забыл
            HumanMessage(content="Ростелеком, как я сказал в первом сообщении!"),
            AIMessage(content="Записал: автоплатеж за Ростелеком 890 рублей с 1 мая.")
        ]
    )

    sample28 = MultiTurnSample(
        user_input=[
            HumanMessage(content="Почему сняли 2,500 рублей за страховку кредитной карты?"),
            AIMessage(content="Это плата за программу 'Защита покупок', подключенную 15.03.2023."),
            HumanMessage(content="Но я отключал её в феврале через приложение!"),
            AIMessage(content="Проверил. Действительно, отмена не прошла. Верну деньги и отключу страховку.")
        ]
    )

    sample29 = MultiTurnSample(
        user_input=[
            HumanMessage(content="Мне нужно заказать выписку по счету *4321 за март 2024."),
            AIMessage(content="Выписка по *4321 за март будет отправлена на email user@example.com."),
            HumanMessage(content="Нет, отправьте на work@mail.ru, это мой новый email."),
            AIMessage(content="Обновил email. Выписка уже летит на work@mail.ru.")
        ]
    )

    sample30 = MultiTurnSample(
        user_input=[
            HumanMessage(content="Почему не приходит код подтверждения для смены пароля?"),
            AIMessage(content="Коды идут на email или SMS? У вас в настройках выбран email."),
            HumanMessage(content="Я жду SMS! Смените способ подтверждения."),
            AIMessage(content="Прошу прощения, вы могли бы назвать последние 4 цифры карты?"),  # Забыл
            HumanMessage(content="*4321, как в моём профиле! Меняйте способ подтверждения!"),
            AIMessage(content="Сделал! Теперь коды будут приходить по SMS на +7***555-44-33.")
        ]
    )

    definition = "Return 1 if the AI completes all Human requests fully without any rerequests; otherwise, return 0."

    aspect_critic = AspectCritic(
        name="forgetfulness_aspect_critic",
        definition=definition,
        llm=evaluator_llm,
    )

    result = evaluate(
        dataset=EvaluationDataset(samples = [
        sample1, 
        sample2, sample3, sample4, sample5,
        sample6, sample7, sample8, sample9, sample10,
        sample11, sample12, sample13, sample14, sample15,
        sample16, sample17, sample18, sample19, sample20,
        sample21, sample22, sample23, sample24, sample25,
        sample26, sample27, sample28, sample29, sample30
    ]),
        metrics=[aspect_critic],
        
    )

    print(result.to_pandas())

    forgetfulness_result = result[aspect_critic.name]  # Assuming the result contains the aspect critic's name
    print(f"\nForgetfulness Aspect Critic Result:\n {sum(forgetfulness_result) / len(forgetfulness_result)}\n")
    # print(asyncio.run(evaluate_metric(test_data, evaluator_llm=evaluator_llm)))  # Run the async function
