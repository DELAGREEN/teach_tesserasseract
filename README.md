# Клонирование репозитория 
- что бы клонировать репозиторий с сылками на чужой репозиторий необходимо выполнить команду 
*`git clone --recurse-submodules https://github.com/DELAGREEN/teach_tesserasseract.git`*

# Сборка репозитория
- Первым этапом является добавления сторонних репозиториев в свой проект с помощью команды  
    - добавляем tesstrain *`git submodule add https://github.com/tesseract-ocr/tesstrain.git`*

 # Сборка Docker контейнера
**ВНИМАНИЕ**
<br>
**Бывает так что проект собирается не с первого раза, нужно просто запустить сборку повторно.**
- Что бы собрать проект нужно запустить `docker-compose` с файлом `docker-compose.yml` проект сам соберется
- В файле .env нужно указать путь монтирования к хост машине
- **Если хотите взаимодействовать с файлами в хостмашине от обычного пользователя, раскоментируйте строки `UID` и `GID`в .env файле**

# Чел на Youtube 
https://www.youtube.com/watch?v=KE4xEzFGSU8&ab_channel=GabrielGarcia

# Обучение

`cd sub_modules/tesstrain/` 
`TESSDATA_PREFIX=../tesseract/tessdata make training MODEL_NAME=okbm_dwg_gostw2_304 START_MODEL=rus TESSDATA=../tesseract/tessdata MAX_ITERATIONS=10000`
`mkdir langdata`
`cd langdata`
`git clone https://github.com/tesseract-ocr/langdata_lstm.git/`

`make unicharset lists proto-model tesseract-langdata training MODEL_NAME=rus MAX_ITERATIONS=100000`

распаковать файлы из папки языка в root-langdata