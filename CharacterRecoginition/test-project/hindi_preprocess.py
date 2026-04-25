import io

import numpy as np
from PIL import Image


def _normalize_polarity(img_array):
    """Ensure foreground strokes are bright on dark background."""
    img_array = img_array.astype(np.float32) / 255.0
    if float(img_array.mean()) > 0.5:
        img_array = 1.0 - img_array
    return img_array


def _extract_foreground_bbox(img_array, threshold=0.15):
    ys, xs = np.where(img_array > threshold)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _center_to_canvas(img_array, canvas_size=28, content_size=20):
    bbox = _extract_foreground_bbox(img_array)
    if bbox is None:
        return np.zeros((canvas_size, canvas_size), dtype=np.float32)

    x0, y0, x1, y1 = bbox
    cropped = img_array[y0:y1, x0:x1]

    cropped_pil = Image.fromarray((cropped * 255).astype(np.uint8), mode="L")
    cropped_pil.thumbnail((content_size, content_size), Image.Resampling.LANCZOS)

    out = np.zeros((canvas_size, canvas_size), dtype=np.float32)
    arr = np.asarray(cropped_pil, dtype=np.float32) / 255.0
    h, w = arr.shape
    y = (canvas_size - h) // 2
    x = (canvas_size - w) // 2
    out[y : y + h, x : x + w] = arr
    return out


def preprocess_pil_image(pil_img):
    rgba = pil_img.convert("RGBA")
    background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
    gray = Image.alpha_composite(background, rgba).convert("L")

    img_array = np.asarray(gray, dtype=np.float32)
    img_array = _normalize_polarity(img_array)
    img_array = _center_to_canvas(img_array, canvas_size=28, content_size=20)

    return img_array.reshape(1, 28, 28, 1).astype(np.float32)


def preprocess_image_bytes(image_bytes):
    pil_img = Image.open(io.BytesIO(image_bytes))
    return preprocess_pil_image(pil_img)


def preprocess_image_path(image_path):
    pil_img = Image.open(image_path)
    return preprocess_pil_image(pil_img)
