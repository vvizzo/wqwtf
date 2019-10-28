#!/usr/bin/python3
"""
Create medalions in JPEG format with faces from image.
Accepts one and only one file name as argument.
"""

import sys
import os
import dlib
from PIL import Image, ImageDraw

DETECTOR = dlib.get_frontal_face_detector()

def create_meds(img):
    """
    Real face processing, we get clipped face area
    and save as medalion.
    Argument: Image object
    Output: save medalions as JPEG files
    """

    print("Processing file: {}".format(img.filename))
    # Real face detection
    iobj = dlib.load_rgb_image(img.filename)
    dets = DETECTOR(iobj)
    # Processing all faces in image
    for i, d in enumerate(dets, start=1):

        coords = face_area(d.left(), d.top(), d.right(), d.bottom())

        imcrop = img.crop(box=coords)

        immed = medalion(imcrop)

        outfname = os.path.splitext(img.filename)[0]+f'-med-{i}.jpg'

        immed.save(outfname, 'JPEG')


def face_area(x_1, y_1, x_2, y_2):
    """
    Increase face area, to square format, face detectors are very close
    clipping useless when you want to get whole head.
    Arguments: 4 ints, coordinates of bounding box
    Output: 4 ints, expanded bounding box
    """

    x_center = int(x_1 + (x_2 - x_1) / 2)
    y_center = int(y_1 + (y_2 - y_1) / 2)

    factor = 2
    square_factor = int(max(x_2 - x_1, y_2 - y_1) * factor / 2)

    x_1p = x_center - square_factor
    y_1p = y_center - square_factor
    x_2p = x_1p + square_factor * 2
    y_2p = y_1p + square_factor * 2

    return (x_1p, y_1p, x_2p, y_2p)


def medalion(square):
    """
    Create circle on white background with head.
    Argument: Image object
    Output: masked Image object
    """

    # Create white background for our medalion
    masked = Image.new('RGB', (square.width, square.height), 'white')
    # Create mask, in this case I want it black
    maskbg = Image.new('L', (square.width, square.height), 'black')
    # draw white circle to create medalion
    draw = ImageDraw.Draw(maskbg)
    draw.ellipse([0, 0, square.width, square.height], 'white')
    masked.paste(square, mask=maskbg)

    return masked

def main():
    """
    Process first file name in command line.
    """

    # Take only first file
    if len(sys.argv) < 2:
        print("Usage: we need file to process")
        raise SystemExit

    im = Image.open(sys.argv[1])

    # Create medalions
    create_meds(im)


if __name__ == "__main__":
    main()
