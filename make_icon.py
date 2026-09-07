"""Draw HorseRacePredictor.ico (a green track with a blue finishing post).
Run once; Pillow ships with Streamlit."""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
SIZES = [16, 20, 24, 32, 40, 48, 64, 128, 256]
TOP, BOTTOM, ACCENT, POST = (16, 74, 40), (34, 139, 84), (250, 204, 21), (37, 99, 235)


def draw(size: int) -> Image.Image:
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    for y in range(size):
        t = y / max(1, size - 1)
        d.line([(0, y), (size, y)],
               fill=tuple(int(TOP[i] + (BOTTOM[i] - TOP[i]) * t) for i in range(3)) + (255,))
    mask = Image.new("L", (size, size), 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, size - 1, size - 1], radius=max(2, size // 5), fill=255)
    img.putalpha(mask)
    d = ImageDraw.Draw(img)
    # oval track
    pad = max(1, size // 6)
    w = max(1, size // 14)
    d.ellipse([pad, size // 3, size - pad, size - pad], outline=ACCENT, width=w)
    # finishing post
    x = size - pad - w
    d.line([(x, size // 6), (x, size // 3 + w)], fill=POST, width=max(1, w))
    d.rectangle([x, size // 6, min(size - 1, x + size // 5), size // 6 + size // 10], fill=POST)
    return img


def main() -> None:
    frames = [draw(s) for s in SIZES]
    out = HERE / "HorseRacePredictor.ico"
    frames[-1].save(out, format="ICO", sizes=[(s, s) for s in SIZES],
                    append_images=frames[:-1])
    print("wrote", out)


if __name__ == "__main__":
    main()
