import os
import random
import pathlib
from PIL import Image, ImageDraw, ImageFont
import icecream

def text_to_image(text, font_path, font_size, image_width, image_height, background_color, text_color, path_to_save_image):
    # Создание изображения с текстом
    img = Image.new('RGB', (image_width, image_height), background_color)
    d = ImageDraw.Draw(img)
    font = ImageFont.truetype(font_path, font_size)

    # Определение начальной позиции текста
    text_x, text_y = 10, 10

    # Рендер текста на изображение
    d.text((text_x, text_y), text, fill=text_color, font=font)

    # Сохранение изображения
    icecream.ic(f'{path_to_save_image}.tif')
    img.save(f'{path_to_save_image}.tif')

    # Возвращаем размеры текста для создания box файлов
    text_width, text_height = d.textsize(text, font=font)
    return text_x, text_y, text_width, text_height

def create_box_file(text, font_path, font_size, text_x, text_y, text_width, text_height, path_to_save_box):
    # Создание файла .box на основе текста и его координат
    with open(path_to_save_box, 'w') as box_file:
        x_cursor = text_x  # Начальная позиция для каждого символа

        for char in text:
            char_width, char_height = ImageFont.truetype(font_path, font_size).getsize(char)

            # Запись символа и его координат в формате .box
            box_file.write(f"{char} {x_cursor} {text_y} {x_cursor + char_width} {text_y + text_height} 0\n")

            # Сдвиг курсора на ширину символа + отступ
            x_cursor += char_width

training_text_file = r'/home/nzxt/rep/teach_tesserasseract/sub_modules/langdata/rus/rus.training_text'
lines = []

# Чтение строк из файла текста для обучения
with open(training_text_file, 'r') as input_file:
    for line in input_file.readlines():
        lines.append(line.strip())

output_directory = r'/home/nzxt/rep/teach_tesserasseract/test'

# Создание выходной директории, если её нет
if not os.path.exists(output_directory):
    os.mkdir(output_directory)

random.shuffle(lines)
line_count = 0

# Параметры шрифта
font_path = r'/home/nzxt/rep/teach_tesserasseract/fonts/gost_2.304_italic.ttf'
font_size = 45
image_width = 980
image_height = 200
background_color = (255, 255, 255)
text_color = (0, 0, 0)

# Генерация файлов
for line in lines:
    training_text_file_name = pathlib.Path(training_text_file).stem

    # Создание файла .gt
    line_training_text = os.path.join(output_directory, f'{training_text_file_name}_{line_count}.gt.txt')
    with open(line_training_text, 'w') as output_file:
        output_file.writelines([line])

    # Генерация изображения и получение размеров текста
    file_base_name = f'rus_{line_count}'
    path = os.path.join(output_directory, file_base_name)

    text_x, text_y, text_width, text_height = text_to_image(
        line, font_path, font_size, image_width, image_height, background_color, text_color, path
    )

    # Создание файла .box
    box_file_path = os.path.join(output_directory, f'{file_base_name}.box')
    create_box_file(line, font_path, font_size, text_x, text_y, text_width, text_height, box_file_path)

    line_count += 1
