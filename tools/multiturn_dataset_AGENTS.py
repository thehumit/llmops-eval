from ragas.dataset_schema import MultiTurnSample
from ragas.messages import HumanMessage

def get_evaluation_samples():
    # Each sample.user_input is a list of HumanMessage only.
    return [
        MultiTurnSample(
            user_input=[
                HumanMessage(content="Почему моя транзакция в Азбуке Вкуса была отклонена?"),
                HumanMessage(content="Почему не приходит смс с кодом для смены пароля?"),
            ],

        ),
        MultiTurnSample(
            user_input=[
                HumanMessage(content="Почему не приходит смс с кодом для смены пароля?")
            ]
        ),
        # … your 30+ samples …
    ]