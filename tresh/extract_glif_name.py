from fontTools.ttLib import TTFont

# Загрузка TTF файла
font = TTFont('/home/nzxt/rep/teach_tesserasseract/fonts/gost_2.304_italic.ttf')

# Получение символов из шрифта
cmap = font['cmap']
table = cmap.getcmap(3, 1).cmap

# Вывод всех символов
for char_code, glyph_name in table.items():
    try:
        print(f"Символ: {chr(char_code)}, Код: {char_code}, Имя глифа: {glyph_name}")
    except UnicodeEncodeError:
        print(f"Символ не может быть отображен, Код: {char_code}, Имя глифа: {glyph_name}")

# Закрытие файла шрифта
font.close()
