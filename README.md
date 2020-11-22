# GPT3-example
Скрипт примера запуска GPT3 от Сбера

Внимание! Скрипт скачивает модели в несколько гигабайт - это может занять много времени. Единожды скачавшись модель кэшируется и далее скрипт запускает её из кэша.

Необходимые модули:

nltk>=3.4

numpy>=1.15.4

pandas>=0.24.0

sentencepiece>=0.1.8

tensorflow>=1.12.0

boto3==1.11.11

regex==2020.1.8

transformers==2.8.0



Для запуска перейти в консоли в папку со скриптом и дать команду


Для запуска средней модели (требует минимум 8 гб ОЗУ):

python generate_transformers.py --model_type=gpt2 --model_name_or_path=sberbank-ai/rugpt3medium_based_on_gpt2 --temperature=0.9 --k=0 --p=0.95 --length=100


Для запуска большой модели (требует минимум 16 гб ОЗУ):

python generate_transformers.py --model_type=gpt2 --model_name_or_path=sberbank-ai/rugpt3large_based_on_gpt2 --temperature=0.9 --k=0 --p=0.95 --length=100


Дождаться загрузки и появления надписи Context>>


Начать писать начало предложения и нажать Enter - GPT3 продолжит его за вас

