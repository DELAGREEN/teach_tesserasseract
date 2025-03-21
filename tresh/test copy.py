import os
import random
import pathlib
from PIL import Image, ImageDraw, ImageFont
import icecream

def text_to_image(text, font_path, font_size, image_width, image_height, background_color, text_color, path_to_save_image):
    img = Image.new('RGB', (image_width, image_height), background_color)
    d = ImageDraw.Draw(img)
    font = ImageFont.truetype(font_path, font_size)

    # Получаем общую высоту текста
    _1, _2, _3, text_height = d.textbbox((0, 0), text, font=font)
    icecream.ic(f'{_1} {_2} {_3}')
    x = 10  # Начальная x-координата
    y = (image_height - text_height) // 2  # Центрирование текста по вертикали
    boxes = []

    for char in text:
        # Получаем размеры символа
        char_bbox = font.getbbox(char)
        char_width = char_bbox[2] - char_bbox[0]
        char_height = char_bbox[3] - char_bbox[1]

        # Рисуем символ
        d.text((x, y), char, fill=text_color, font=font)

        # Сохраняем координаты box
        boxes.append((char, x, y, x + char_width, y + char_height))

        # Двигаем курсор вправо с учетом межсимвольного интервала
        x += char_width + 1  # +1 для небольшого отступа между символами

    img.save(f'{path_to_save_image}_1.tif')
    return boxes


def create_box_file(boxes, path_to_save_box):
    with open(path_to_save_box, 'w') as box_file:
        for char, x_min, y_min, x_max, y_max in boxes:
            box_file.write(f"{char} {x_min} {y_min} {x_max} {y_max} 0\n")

def draw_boxes_on_image(image_path, boxes, output_path):
    img = Image.open(image_path)
    d = ImageDraw.Draw(img)

    for _, x_min, y_min, x_max, y_max in boxes:
        d.rectangle([x_min, y_min, x_max, y_max], outline="green", width=2)

    img.save(output_path)

# Параметры
training_text_file = r'/home/nzxt/rep/teach_tesserasseract/sub_modules/langdata/rus/rus.training_text'
output_directory = r'/home/nzxt/rep/teach_tesserasseract/test'
font_path = r'/home/nzxt/rep/teach_tesserasseract/fonts/gost_2.304_italic.ttf'
font_size = 45
image_width = 1150
image_height = 150
background_color = (255, 255, 255)
text_color = (0, 0, 0)

# Чтение текста
lines = []
with open(training_text_file, 'r') as input_file:
    for line in input_file.readlines():
        lines.append(line.strip())

if not os.path.exists(output_directory):
    os.mkdir(output_directory)

random.shuffle(lines)

# Генерация данных
line_count = 0
for line in lines:
    training_text_file_name = pathlib.Path(training_text_file).stem
    line_training_text = os.path.join(output_directory, fr'{training_text_file_name}_{line_count}.gt.txt')

    # Создание файла .gt
    with open(line_training_text, 'w') as output_file:
        output_file.writelines([line])

    # Создание изображения и боксов
    file_base_name = fr'rus_{line_count}'
    path = os.path.join(output_directory, file_base_name)

    boxes = text_to_image(line, font_path, font_size, image_width, image_height, background_color, text_color, path)

    # Создание файла .box
    box_file_path = f'{path}.box'
    create_box_file(boxes, box_file_path)

    # Создание изображения с обведёнными символами
    boxed_image_path = f'{path}_boxed.tif'
    draw_boxes_on_image(f'{path}_1.tif', boxes, boxed_image_path)

    line_count += 1
