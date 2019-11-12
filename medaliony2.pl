#!/usr/bin/python3
"""
Medalions with 5 point face predictor and head position correction
"""

import sys
import os
import dlib
from PIL import Image, ImageDraw


def get_correction(shape):
    """
    Get correction value depending on position of head.
    Argument: shape - dlib.predictor values
    Returns: corr - int
    """

    # Get corners of eyes
    x_A1 = shape.part(0).x # Right corner of right eye
    y_A1 = shape.part(0).y
    x_A2 = shape.part(1).x # Left corner of right eye
    y_A2 = shape.part(1).y
    x_B1 = shape.part(2).x # Right corner of left eye
    y_B1 = shape.part(2).y
    x_B2 = shape.part(3).x # Left corner of left eye
    y_B2 = shape.part(3).y

    # Get center of eyes for more precision than from corners
    x_A = (x_A1+x_A2)/2
    y_A = (y_A1+y_A2)/2
    x_B = (x_B1+x_B2)/2
    y_B = (y_B1+y_B2)/2


    x_C = shape.part(4).x # Base of nose
    y_C = shape.part(4).y

    # Center point between eyes
    x_Ce = (x_B+x_A)/2
    # yCe = (yB+yA)/2

    # Compute orientation point
    # Note: everything is in float at the moment
    # If we want to use it for drawing remember to int them
    # Leaving as floats for better precision when computing, especially
    # important for smaller images
    ad = (y_B-y_A)/(x_B-x_A) # this is slope of line which is connecting eyes
    x_L = (y_C + (1/ad)*x_C - y_A + ad*x_A)/(ad + 1/ad)
    # At the moment we don't need yL but leave it for future use
    # y_L = (-1/ad)*x_L + y_C + (1/ad)*x_C

    # Correction for bounding box position
    # Defaults for 2, probably could (should?) be gradiented
    # More significant correction when difference is bigger
    # to account for emerging shape of head
    return int((x_Ce-x_L)*2)


def face_area(bounding_box, correction):
    """
    Increase face area, to square format, face detectors are very close
    clipping useless when you want to get whole head
    Arguments: bounding box original, correction value
    Returns: 4-element list - bounding box for expanded area (ints)
    """

    x_1, y_1, x_2, y_2 = bounding_box

    x_1 = x_1 + correction
    x_2 = x_2 + correction

    x_center = int(x_1 + (x_2 - x_1) / 2)
    y_center = int(y_1 + (y_2 - y_1) / 2)

    factor = 2
    square_factor = int(max(x_2 - x_1, y_2 - y_1) * factor / 2)

    x_1p = x_center - square_factor
    y_1p = y_center - square_factor
    x_2p = x_1p + square_factor * 2
    y_2p = y_1p + square_factor * 2

    return [x_1p, y_1p, x_2p, y_2p]


def medalion(square):
    """
    Create circle on white background with head
    Argument: Image - whole
    Returns: Image - circle on white background
    """

    # Create white background for our medalion
    masked = Image.new('RGB', square.size, 'white')
    # Create mask, in this case I want it black
    maskbg = Image.new('L', square.size, 'black')
    # draw white circle to create medalion
    draw = ImageDraw.Draw(maskbg)
    draw.ellipse([0, 0, square.width, square.height], 'white')
    masked.paste(square, mask=maskbg)

    return masked


def main():
    """
    Main processing function.
    """

    if len(sys.argv) < 2:
        raise SystemExit('Usage: Not enough arguments')

    # Traditional HOG detection
    detector = dlib.get_frontal_face_detector()
    # Using 5 points have advantage not only of simplicity but it will also
    # support CNN
    predictor = dlib.shape_predictor('shape_predictor_5_face_landmarks.dat')

    for fname in sys.argv[1:]:
        img = dlib.load_rgb_image(fname)
        pilimage = Image.open(fname)
        draw = ImageDraw.Draw(pilimage)


        # Ask the detector to find the bounding boxes of each face. The 1 in the
        # second argument indicates that we should upsample the image 1 time.
        # This will make everything bigger and allow us to detect more faces.
        dets = detector(img, 1)
        for k, d in enumerate(dets, start=1):
            orig_bb = (d.left(), d.top(), d.right(), d.bottom())
            # Get the landmarks/parts for the face in box d.
            shape = predictor(img, d)
            # Get coordinats for our cardinal points
            corr = get_correction(shape)

            # Get bounding box corrected for shape (square) and head position
            corrected_bb = face_area(orig_bb, corr)

            # Crop original image to new bounding box
            imcrop = pilimage.crop(box=corrected_bb)

            # Create medalion
            immed = medalion(imcrop)

            # Save medalion
            outfname = pilimage.filename.replace('dane', 'dane5')
            outfname = os.path.splitext(outfname)[0]+f'-med-{k}.jpg'
            immed.save(outfname, 'JPEG')

if __name__ == '__main__':
    main()
