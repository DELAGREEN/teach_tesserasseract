import os
import random
import pathlib
from PIL import Image, ImageDraw, ImageFont

import subprocess


def text_to_image(text, font_path, font_size, image_width, image_height, background_color, text_color, path_to_save_image):
    img = Image.new('RGB', (image_width, image_height), background_color)
    d = ImageDraw.Draw(img)
    font = ImageFont.truetype(font_path, font_size)
    d.text((10, 10), text, fill=text_color, font=font)
    #icecream.ic(f'{path_to_save_image}.tif')
    img.save(f'{path_to_save_image}_1.tif')


def genarate_box():

    subprocess.run([
        'text2image',
        '--font=gost_2.304',
        #'--font=Symbol type A',
        f'--text={line_training_text}',
        f'--outputbase={output_directory}/{file_base_name}',
        '--max_pages=1',
        '--strip_unrenderable_words',
        '--leading=48',
        '--xsize=3600',
        '--ysize=480',
        '--char_spacing=1.0',
        '--exposure=0',
        '--unicharset_file=/langdata/rus/rus.unicharset'
    ])

training_text_file = '/home/astraadmin/Desktop/rep/drawing_text_recognition/new/tesseract-tutorial/tesstrain/langdata/rus/rus.training_text'

lines = ['Это пример текста для обучения и пример разметки.']

#with open(training_text_file, 'r') as input_file:
#    for line in input_file.readlines():
#        lines.append(line.strip())

output_directory = '/home/nzxt/rep/teach_tesserasseract/test'

if not os.path.exists(output_directory):
    os.mkdir(output_directory)

random.shuffle(lines)

#count = 100

#lines = lines[:count]

line_count = 0
for line in lines:
    training_text_file_name = pathlib.Path(training_text_file).stem
    line_training_text = os.path.join(output_directory, f'{training_text_file_name}_{line_count}.gt.txt')
    with open(line_training_text, 'w') as output_file:
        output_file.writelines([line])

    file_base_name = f'rus_{line_count}'

    path = f'{output_directory}/{file_base_name}'
    
    genarate_box()
    text_to_image(line, '/home/nzxt/rep/teach_tesserasseract/fonts/gost_2.304.ttf', 45, 980, 200, (0, 0, 0), (255, 255, 255), path)

    line_count += 1
