from fontTools.ttLib import TTFont
from PIL import Image, ImageDraw, ImageFont
import math

# Загрузка TTF файла
font_path = '/home/nzxt/rep/teach_tesserasseract/fonts/gost_2.304_italic.ttf'
font = TTFont(font_path)

# Получение символов из шрифта
cmap = font['cmap']
table = cmap.getcmap(3, 1).cmap

# Параметры изображения
image_size = (1000, 1000)
background_color = (255, 255, 255)  # Белый фон
text_color = (0, 0, 0)  # Черный текст
font_size = 20
chars_per_row = 20
padding = 5

# Создание изображения и объекта для рисования
image = Image.new('RGB', image_size, background_color)
draw = ImageDraw.Draw(image)
pil_font = ImageFont.truetype(font_path, font_size)

# Функция для создания нового изображения
def create_new_image():
    return Image.new('RGB', image_size, background_color), ImageDraw.Draw(Image.new('RGB', image_size, background_color))

# Отрисовка символов
x, y = padding, padding
image_count = 1
for char_code, glyph_name in table.items():
    if y + font_size > image_size[1] - padding:
        # Сохранение текущего изображения и создание нового
        image.save(f'font_chars_{image_count}.png')
        image_count += 1
        image, draw = create_new_image()
        x, y = padding, padding

    try:
        char = chr(char_code)
        draw.text((x, y), char, font=pil_font, fill=text_color)
        x += font_size + padding
        if x > image_size[0] - font_size - padding:
            x = padding
            y += font_size + padding
    except UnicodeEncodeError:
        print(f"Символ не может быть отображен, Код: {char_code}, Имя глифа: {glyph_name}")

# Сохранение последнего изображения
image.save(f'font_chars_{image_count}.png')

print(f"Создано {image_count} изображений с символами шрифта.")

# Закрытие файла шрифта
font.close()
