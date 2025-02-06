import os
import random
import pathlib
from PIL import Image, ImageDraw, ImageFont
import icecream

def text_to_image(text, font_path, font_size, image_width, image_height, background_color, text_color, path_to_save_image):
    img = Image.new('RGB', (image_width, image_height), background_color)
    d = ImageDraw.Draw(img)
    font = ImageFont.truetype(font_path, font_size)
    d.text((10, 10), text, fill=text_color, font=font)
    icecream.ic(f'{path_to_save_image}.tif')
    img.save(f'{path_to_save_image}.tif')

def text_to_image_with_boxes(text, font_path, font_size, image_width, image_height, background_color, text_color, path_to_save_image):
    img = Image.new('RGB', (image_width, image_height), background_color)
    d = ImageDraw.Draw(img)
    font = ImageFont.truetype(font_path, font_size)
    
    x, y = 10, 10  # Начальная позиция текста

    # Рисуем текст и боксы вокруг символов
    for char in text:
        # Используем textbbox для получения координат bounding box
        bbox = d.textbbox((x, y), char, font=font)  # Получаем координаты bounding box
        width = bbox[2] - bbox[0]  # Ширина (x2 - x1)
        height = bbox[3] - bbox[1]  # Высота (y2 - y1)
        d.text((x, y), char, fill=text_color, font=font)
        
        # Рисуем бокс вокруг символа
        d.rectangle([bbox[0], bbox[1], bbox[2], bbox[3]], outline="red", width=2)

        # Смещаем x на ширину символа с учетом небольшого отступа
        x += width + 2  # Отступ между символами

    icecream.ic(f'{path_to_save_image}_with_boxes.tif')
    img.save(f'{path_to_save_image}_with_boxes.tif')

def create_box_file(text, path_to_save_box, font_path, font_size):
    # Создание файла .box на основе текста
    with open(path_to_save_box, 'w') as box_file:
        x, y = 10, 10  # Начальная позиция текста
        font = ImageFont.truetype(font_path, font_size)
        img = Image.new('RGB', (1, 1))  # Создаем пустое изображение для вычислений
        d = ImageDraw.Draw(img)
        for char in text:
            # Используем textbbox для получения координат bounding box
            bbox = d.textbbox((x, y), char, font=font)
            width = bbox[2] - bbox[0]  # Ширина (x2 - x1)
            height = bbox[3] - bbox[1]  # Высота (y2 - y1)
            box_file.write(f"{char} 0 {x} {y} {x + width} {y + height}\n")
            x += width + 2  # Смещаем x на ширину символа с учетом отступа

training_text_file = r'/home/user/rep/teach_tesserasseract/sub_modules/langdata_lstm/rus/rus.training_text'

lines = []

with open(training_text_file, 'r') as input_file:
    for line in input_file.readlines():
        lines.append(line.strip())

output_directory = r'/home/user/rep/teach_tesserasseract/sub_modules/tesstrain/data/rus-ground-truth'

if not os.path.exists(output_directory):
    os.mkdir(output_directory)

random.shuffle(lines)

line_count = 0
for line in lines:
    if line_count < 20000:
        training_text_file_name = pathlib.Path(training_text_file).stem
        line_training_text = os.path.join(output_directory, f'{training_text_file_name}_{line_count}.gt.txt')

        # Создание файла .gt
        with open(line_training_text, 'w') as output_file:
            output_file.writelines([line])

        # Создание файла .box
        box_file_path = os.path.join(output_directory, f'{training_text_file_name}_{line_count}.box')
        create_box_file(line, box_file_path, r'/home/user/rep/teach_tesserasseract/fonts/gost_2.304.ttf', 45)

        file_base_name = fr'rus_{line_count}'

        path = fr'{output_directory}/{file_base_name}'

        # Создание обычного изображения
        text_to_image(line, r'/home/user/rep/teach_tesserasseract/fonts/gost_2.304.ttf', 45, 980, 200, (255, 255, 255), (0, 0, 0), path)

        # Создание изображения с боксовыми линиями
        text_to_image_with_boxes(line, r'/home/user/rep/teach_tesserasseract/fonts/gost_2.304.ttf', 45, 980, 200, (255, 255, 255), (0, 0, 0), path)

        line_count += 1
