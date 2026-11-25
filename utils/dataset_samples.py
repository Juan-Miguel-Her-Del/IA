"""Hojas de inspeccion con imagenes reales; no genera contenido de entrenamiento."""
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

from utils.config import CLASSES


def font(size=15):
    for name in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def tile(root, row, width=240, height=230):
    canvas = Image.new("RGB", (width, height), "white")
    with Image.open(root / row["path"]) as source:
        image = ImageOps.contain(ImageOps.exif_transpose(source).convert("RGB"), (width - 12, height - 66))
        canvas.paste(image, ((width - image.width) // 2, 4))
    draw = ImageDraw.Draw(canvas)
    lines = [f"{row['class']} | {row['dataset']}", row["source_id"], row.get("split", row.get("status", ""))]
    for number, line in enumerate(lines):
        draw.text((6, height - 62 + number * 19), line[:34], fill="black", font=font())
    return canvas


def generate_samples(root, rows, candidates, destination, seed):
    destination.mkdir(parents=True, exist_ok=True)
    overview = Image.new("RGB", (960, 6 * 230), "#eeeeee")
    for class_index, label in enumerate(CLASSES):
        selected = sorted((r for r in rows if r["class"] == label and r.get("status") == "usable"), key=lambda r: r["path"])
        random.Random(f"{seed}:{label}:samples").shuffle(selected)
        selected = selected[:8]
        sheet = Image.new("RGB", (960, 460), "#eeeeee")
        for index, row in enumerate(selected):
            image = tile(root, row)
            sheet.paste(image, ((index % 4) * 240, (index // 4) * 230))
            if index < 4:
                overview.paste(image, (index * 240, class_index * 230))
        sheet.save(destination / f"{label}.jpg", quality=90)
    overview.save(destination / "categories.jpg", quality=90)
    by_path = {row["path"]: row for row in rows}
    review = [r for r in candidates if r["left"] in by_path and r["right"] in by_path]
    # Todas las parejas se muestran en paginas separadas, sin limitar la auditoria.
    for page in range((len(review) + 7) // 8):
        canvas = Image.new("RGB", (960, 4 * 258), "#eeeeee")
        for index, pair in enumerate(review[page * 8:(page + 1) * 8]):
            x, y = (index % 2) * 480, (index // 2) * 258
            canvas.paste(tile(root, by_path[pair["left"]]), (x, y))
            canvas.paste(tile(root, by_path[pair["right"]]), (x + 240, y))
            ImageDraw.Draw(canvas).text((x + 6, y + 233),
                f"pHash={pair['phash_distance']} dHash={pair['dhash_distance']} | revisar manualmente", fill="black", font=font(13))
        canvas.save(destination / f"visual_duplicates_{page + 1:03d}.jpg", quality=90)
