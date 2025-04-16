import cv2
import argparse

def extract_sketch(input_path, output_path='middle.png'):
    img = cv2.imread(input_path)
    if img is None:
        raise ValueError(f"Could not open image {input_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    inv = 255 - gray
    blur = cv2.GaussianBlur(inv, (21, 21), 0)
    sketch = cv2.divide(gray, 255 - blur, scale=256.0)

    cv2.imwrite(output_path, sketch)
    print(f"[✓] Sketch saved as {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='Path to AI-generated input image')
    parser.add_argument('--output', default='middle.png', help='Path to save sketch')
    args = parser.parse_args()

    extract_sketch(args.input, args.output)
