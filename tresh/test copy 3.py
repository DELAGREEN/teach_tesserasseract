import os
import random
import pathlib
from PIL import Image, ImageDraw, ImageFont
from fontTools.ttLib import TTFont
from fontTools.pens.boundsPen import BoundsPen

def get_char_bounds(font_path, char):
    font = TTFont(font_path)
    glyph_set = font.getGlyphSet()
    cmap = font.getBestCmap()
    if ord(char) not in cmap:
        return None
    glyph_name = cmap[ord(char)]
    
    pen = BoundsPen(glyph_set)
    glyph_set[glyph_name].draw(pen)
    
    return pen.bounds

def get_font_metrics(font_path):
    font = TTFont(font_path)
    hhea = font['hhea']
    return {
        'ascent': hhea.ascent,
        'descent': hhea.descent,
        'line_gap': hhea.lineGap
    }

def text_to_image(text, font_path, font_size, image_width, image_height, background_color, text_color, path_to_save_image):
    img = Image.new('RGB', (image_width, image_height), background_color)
    d = ImageDraw.Draw(img)
    font = ImageFont.truetype(font_path, font_size)

    metrics = get_font_metrics(font_path)
    ascent = metrics['ascent'] * font_size / 1000
    descent = -metrics['descent'] * font_size / 1000

    x = 10
    y = (image_height - (ascent + descent)) // 2 + ascent
    boxes = []

    for char in text:
        char_bounds = get_char_bounds(font_path, char)
        if char_bounds:
            char_width = char_bounds[2] - char_bounds[0]
            char_height = char_bounds[3] - char_bounds[1]
            
            scaled_width = char_width * font_size / 1000
            scaled_height = char_height * font_size / 1000
            
            # Используем textbbox для получения фактических размеров отрисованного символа
            bbox = d.textbbox((x, y - ascent), char, font=font)
            actual_width = bbox[2] - bbox[0]
            actual_height = bbox[3] - bbox[1]
            
            d.text((x, y - ascent), char, fill=text_color, font=font)
            
            # Используем фактические размеры для box
            boxes.append((char, int(x), int(bbox[1]), int(x + actual_width), int(bbox[3])))
            
            # Уменьшаем отступ между символами
            x += actual_width
        else:
            # Если символ не найден в шрифте, используем стандартный метод
            bbox = d.textbbox((x, y - ascent), char, font=font)
            d.text((x, y - ascent), char, fill=text_color, font=font)
            boxes.append((char, bbox[0], bbox[1], bbox[2], bbox[3]))
            x += bbox[2] - bbox[0]

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
    if line_count >= 20:  # Ограничение количества генерируемых изображений для теста
        break

print(f"Сгенерировано {line_count} изображений и box-файлов.")
