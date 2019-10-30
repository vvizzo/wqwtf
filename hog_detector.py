#!/usr/bin/python3
# The contents of this file are in the public domain.
#
#   Usage:
#       ./hog_detector.py files/*.jpg
#
#   To install dlib
#       pip install dlib
#   Remember that dlib requires numpy for proper working
#       pip install numpy
#   To draw on images you need at least PIL/Pillow module
#   It is available in all pip repositories contrary to more robust OpenCV
#   But for simple marking of faces it is more that enough (ver. >= 5.3.0)
#       pip install Pillow

import sys
from os.path import splitext
import dlib
from PIL import Image, ImageDraw

if len(sys.argv) == 1:
    print("Usage: ./hog_detector.py files/*.jpg")

# Load classic Histogram Oriented Gradients face detector
detector = dlib.get_frontal_face_detector()

for f in sys.argv[1:]:
    # Real work is done in those two lines
    img = dlib.load_rgb_image(f)
    coords = detector(img)

    # Simple reporting to know what is going on
    print("File: {}; faces detected: {}".format(f, len(coords)))

    # Annotate files with red rectangles
    if len(coords) > 0:

        fm = Image.open(f)

        for i, d in enumerate(coords):
            draw_rect = ImageDraw.Draw(fm)
            draw_rect.rectangle([(d.left(), d.top()), (d.right(), d.bottom())],
                                outline="red", width=5)

        outfile = splitext(f)[0]+'-hog-mark.jpg'
        fm.save(outfile, 'JPEG')
