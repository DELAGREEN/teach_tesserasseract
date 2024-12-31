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
    img.save(f'{path_to_save_image}_1.tif')

def create_box_file(text, path_to_save_box):
    # Создание файла .box на основе текста
    with open(path_to_save_box, 'w') as box_file:
        for i, char in enumerate(text):
            # Запись символа и его координат (x1, y1, x2, y2) в формате .box
            # Здесь предполагается фиксированное значение для координат (10, 10) и размера символа.
            # Вы можете изменить это в зависимости от ваших требований.
            box_file.write(f"{char} 0 10 10 50 50\n")  # Примерные координаты

training_text_file = r'/home/nzxt/rep/teach_tesserasseract/tesstrain/langdata/rus/rus.training_text'

lines = []

with open(training_text_file, 'r') as input_file:
    for line in input_file.readlines():
        lines.append(line.strip())

output_directory = r'/home/nzxt/rep/teach_tesserasseract/test'

if not os.path.exists(output_directory):
    os.mkdir(output_directory)

random.shuffle(lines)

line_count = 0
for line in lines:
    training_text_file_name = pathlib.Path(training_text_file).stem
    line_training_text = os.path.join(output_directory, fr'{training_text_file_name}_{line_count}.gt.txt')

    # Создание файла .gt
    with open(line_training_text, 'w') as output_file:
        output_file.writelines([line])

    # Создание файла .box
    box_file_path = os.path.join(output_directory, f'{training_text_file_name}_{line_count}.box')
    create_box_file(line, box_file_path)

    file_base_name = fr'rus_{line_count}'

    path = fr'{output_directory}/{file_base_name}'

    text_to_image(line, r'/home/nzxt/rep/teach_tesserasseract/fonts/gost_2.304_italic.ttf', 45, 980, 200, (255, 255, 255), (0, 0, 0), path)

    line_count += 1