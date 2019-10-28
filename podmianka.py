#!/usr/bin/python3
"""
Swap faces between two photos.
"""

import sys
import os
import dlib
from PIL import Image

DETECTOR = dlib.get_frontal_face_detector()

def get_crop(image):
    """
    Real face processing, we get clipped face area
    and its coordinates
    """

    print("Processing file: {}".format(image.filename))
    # Real face detection
    img = dlib.load_rgb_image(image.filename)
    dets = DETECTOR(img)
    # Processing only first face on image
    (x_1, y_1, x_2, y_2) = (dets[0].left(), dets[0].top(),
                            dets[0].right(), dets[0].bottom())

    (x_1f, y_1f, x_2f, y_2f) = face_chip(x_1, y_1, x_2, y_2)

    imcrop = image.crop(box=(x_1f, y_1f, x_2f, y_2f))

    return (imcrop, x_1f, y_1f, x_2f, y_2f)


def face_chip(x_1, y_1, x_2, y_2):
    """
    Increase face area, face detectors are very close clipping
    useless when you want to get whole head
    """

    height = x_2 - x_1
    width = y_2 - y_1
    factor = 0.5
    x_1p = int(x_1 - (factor * width))
    x_2p = int(x_2 + (factor * width))
    y_1p = int(y_1 - (factor * height))
    y_2p = int(y_2 + (factor * height))

    return (x_1p, y_1p, x_2p, y_2p)


def adj_crop(crop, x_1, y_1, x_2, y_2):
    """
    Resize crop to fit into other image.
    BICUBIC does the best job when resizing
    """

    new_height = x_2 - x_1
    new_width = y_2 - y_1
    resized = crop.resize((new_width, new_height), resample=Image.LANCZOS)
    return resized


def main():
    """
    Process first two images by swapping faces (one face per file)
    """

    # Take only first two images
    if len(sys.argv) < 3:
        print("Usage: we need two files with one face per image as arguments to swap faces")
        raise SystemExit

    fname1 = sys.argv[1]
    fname2 = sys.argv[2]

    im1 = Image.open(fname1)
    im2 = Image.open(fname2)

    # Get crop from image and its coordinates
    (crop1, c1x1, c1y1, c1x2, c1y2) = get_crop(im1)
    (crop2, c2x1, c2y1, c2x2, c2y2) = get_crop(im2)

    # Adjust crop size to fit on second image
    # Since faces are recognized as very close to square we don't have to
    # play with proportions
    crop1adj = adj_crop(crop1, c2x1, c2y1, c2x2, c2y2)
    crop2adj = adj_crop(crop2, c1x1, c1y1, c1x2, c1y2)

    im1.paste(crop2adj, box=(c1x1, c1y1))
    im2.paste(crop1adj, box=(c2x1, c2y1))

    out1 = os.path.splitext(fname1)[0]+'-swapped.jpg'
    out2 = os.path.splitext(fname2)[0]+'-swapped.jpg'

    im1.save(out1, 'JPEG')
    im2.save(out2, 'JPEG')

if __name__ == "__main__":
    main()
