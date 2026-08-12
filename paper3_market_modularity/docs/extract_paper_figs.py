"""Crop the paper's own figures out of the PDF so the notebook can show them
next to ours.

Renders `docs/1501.05040v3.pdf` at 4x and cuts fixed fractional boxes. The boxes
were set by eye once; re-run only if the source PDF changes.

    /opt/anaconda3/bin/python docs/extract_paper_figs.py
"""

import os

import pypdfium2 as pdfium
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(os.path.dirname(os.path.dirname(HERE)), "docs", "1501.05040v3.pdf")
OUT = os.path.join(HERE, "paper_figs")
SCALE = 4
COLOURS = 128          # plots use few colours; quantising cuts file size ~3x

#: name -> (page index, left, top, right, bottom) as fractions of the page
BOXES = {
    "fig3":    (3, 0.118, 0.512, 0.928, 0.786),   # Black Monday + 4 snapshots
    "fig4":    (5, 0.078, 0.052, 0.950, 0.298),   # modularity + path length, 25 y
    "fig5":    (5, 0.078, 0.448, 0.950, 0.643),   # dynamical / fixed / lagged Q
    "fig6_a":  (7, 0.075, 0.092, 0.955, 0.264),   # modularity
    "fig6_b":  (7, 0.075, 0.265, 0.955, 0.432),   # path length
    "fig6_c":  (7, 0.075, 0.430, 0.955, 0.599),   # assortativity
    "fig6_d":  (7, 0.075, 0.597, 0.955, 0.767),   # transitivity
    "fig7":    (8, 0.094, 0.244, 0.512, 0.548),   # PCA
    "figS3_a": (12, 0.075, 0.066, 0.955, 0.228),  # betweenness
    "figS3_b": (12, 0.075, 0.229, 0.955, 0.390),  # clique number
    "figS3_c": (12, 0.075, 0.391, 0.955, 0.556),  # rich club
    "figS3_d": (12, 0.075, 0.556, 0.955, 0.723),  # matching index
}


def main():
    os.makedirs(OUT, exist_ok=True)
    doc = pdfium.PdfDocument(SRC)
    pages = {}
    for name, (pg, l, t, r, b) in BOXES.items():
        if pg not in pages:
            pages[pg] = doc[pg].render(scale=SCALE).to_pil().convert("RGB")
        im = pages[pg]
        W, H = im.size
        crop = im.crop((int(l * W), int(t * H), int(r * W), int(b * H)))
        crop = crop.quantize(colors=COLOURS, method=Image.MEDIANCUT, dither=Image.NONE)
        path = os.path.join(OUT, name + ".png")
        crop.save(path, optimize=True)
        print(f"{name:<10} {crop.size}  {os.path.getsize(path) // 1024} KB")


if __name__ == "__main__":
    main()
