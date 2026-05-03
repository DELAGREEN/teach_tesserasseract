### 1.Виртуальное окружение 
### Устанавливаем
`python3 -m venv venv`
#### Активируем
`source venv/bin/activate` 
#### Устанавливаем зависимости
`pip install -r requirements.txt `
### Устанавливаем Tesseract
`sudo dnf install tesseract`


### Фикс ошибок
При ошибке: ERROR: Unknown compiler(s): [['c++'], ['g++'], ...]
`sudo dnf install gcc-c++ python3-devel`