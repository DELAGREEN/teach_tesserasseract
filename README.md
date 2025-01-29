# Сборка репозитория
- Первым этапом является добавления сторонних репозиториев в свой проект с помощью команды  
    - добавляем tesstrain *git submodule add https://github.com/tesseract-ocr/tesstrain.git*

 # Сборка Docker контейнера
 - команда для сборки *sudo docker build -t "tesseract_teatch:0.1 ."*
    - где tesseract_teatch - наименование docker image
    - где 0.1 версия контейнера опционально
    - . - обозначает с какой директории начинато производить сборку, в моём случае это текущая директория 
- что бы запустить контейнер и проникнуть внутрь него нужно выполнить команду *sudo docker run -it teatch_tesseract:0.1*

# Клонирование репозитория в контейнер 
- что бы клонировать репозиторий с сылками на чужой репозиторий необходимо выполнить команду 
*git clone --recurse-submodules https://github.com/DELAGREEN/teach_tesserasseract.git*





# Чел на Youtube 
https://www.youtube.com/watch?v=KE4xEzFGSU8&ab_channel=GabrielGarcia