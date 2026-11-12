"""Build comparison JPEGs using the checked-in input/output PNGs (no inference)."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path,
                        default=Path(__file__).resolve().parents[1] / "examples/examples.json")
    args = parser.parse_args()
    from PIL import Image, ImageDraw, ImageFont, ImageOps

    manifest = args.manifest.resolve()
    records = json.loads(manifest.read_text(encoding="utf-8"))["examples"]
    root = manifest.parent
    destination = root / "comparisons"
    destination.mkdir(parents=True, exist_ok=True)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 18)
    except OSError:
        font = ImageFont.load_default()

    def panel(cases, path, cell=512, pairs_per_row=1):
        rows = (len(cases) + pairs_per_row - 1) // pairs_per_row
        sheet = Image.new("RGB", (2 * pairs_per_row * cell, rows * (cell + 44)), "white")
        draw = ImageDraw.Draw(sheet)
        for index, record in enumerate(cases):
            row, pair = divmod(index, pairs_per_row)
            for side, field in enumerate(("input", "reference_output")):
                with Image.open(root / record[field]) as source:
                    image = ImageOps.contain(source.convert("RGB"), (cell, cell))
                x = (pair * 2 + side) * cell
                y = row * (cell + 44)
                sheet.paste(image, (x + (cell - image.width) // 2,
                                    y + 44 + (cell - image.height) // 2))
                label = "INPUT" if side == 0 else "OUTPUT"
                draw.text((x + 8, y + 12), f"{record['id']} | {label}", fill="black", font=font)
        sheet.save(path, quality=92)

    for category in ("I", "II", "III", "IV"):
        selected = [record for record in records if record["type"] == category]
        if selected:
            panel(selected, destination / f"type_{category}.jpg")
    assets = root.parent / "assets"
    assets.mkdir(exist_ok=True)
    panel(records, assets / "examples_overview.jpg", cell=384, pairs_per_row=2)
    print(f"Built four type comparisons and {assets / 'examples_overview.jpg'}")


if __name__ == "__main__":
    main()
