import cv2
import numpy as np
from PIL import Image
import argparse
import os


def apply_child_drawing_style(img):
    img = cv2.bilateralFilter(img, 9, 75, 75)
    edges = cv2.Canny(img, 100, 200)
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    cartoon = cv2.addWeighted(img, 0.8, edges_colored, 0.2, 0)
    return cartoon


def apply_cartoon_style(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.adaptiveThreshold(cv2.medianBlur(gray, 7), 255,
                                  cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 2)
    color = cv2.bilateralFilter(img, 9, 300, 300)
    cartoon = cv2.bitwise_and(color, color, mask=edges)
    return cartoon


def apply_sketch_style(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    inv = 255 - gray
    blur = cv2.GaussianBlur(inv, (21, 21), 0)
    sketch = cv2.divide(gray, 255 - blur, scale=256.0)
    return cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)


def apply_oil_paint_style(img):
    return cv2.xphoto.oilPainting(img, 7, 1) if hasattr(cv2, 'xphoto') else img


def apply_anime_style(img):
    img = cv2.bilateralFilter(img, 9, 75, 75)
    img = cv2.edgePreservingFilter(img, flags=1, sigma_s=64, sigma_r=0.2)
    return img


def apply_style(img, style):
    if style == 'child_drawing':
        return apply_child_drawing_style(img)
    elif style == 'cartoon':
        return apply_cartoon_style(img)
    elif style == 'sketch':
        return apply_sketch_style(img)
    elif style == 'oil_paint':
        return apply_oil_paint_style(img)
    elif style == 'anime':
        return apply_anime_style(img)
    else:
        raise ValueError(f"Unknown style: {style}")


def main():
    parser = argparse.ArgumentParser(description='Apply style transformation to an image.')
    parser.add_argument('--input', required=True, help='Path to input image')
    parser.add_argument('--style', required=True, choices=['child_drawing', 'cartoon', 'sketch', 'oil_paint', 'anime'])
    parser.add_argument('--output', default='output.jpg', help='Path to save the output image')
    args = parser.parse_args()

    img = cv2.imread(args.input)
    if img is None:
        print(f"Failed to read image: {args.input}")
        return

    styled_img = apply_style(img, args.style)
    cv2.imwrite(args.output, styled_img)
    print(f"Saved styled image to: {args.output}")


if __name__ == '__main__':
    main()