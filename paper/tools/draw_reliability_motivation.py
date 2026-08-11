#!/usr/bin/env python3
"""Draw the structure--dynamics motivation figure as PDF, SVG, and PNG."""

from pathlib import Path
import shutil
import subprocess

from reportlab.graphics import renderPDF, renderSVG
from reportlab.graphics.shapes import Circle, Drawing, Line, Polygon, Rect, String
from reportlab.lib.colors import HexColor


ROOT = Path(__file__).resolve().parents[2]
FIGURE_DIR = ROOT / "paper" / "figures"
PDF_DIR = ROOT / "output" / "pdf"
SCALE = 72.0

BLACK = HexColor("#202020")
GRAY = HexColor("#686868")
LIGHT_GRAY = HexColor("#D8D8D8")
VERY_LIGHT_GRAY = HexColor("#F4F4F4")
USER = HexColor("#79AE57")
USER_DARK = HexColor("#4D7D36")
ITEM = HexColor("#F3C43D")
ITEM_DARK = HexColor("#A97900")
BLUE = HexColor("#3978B5")
BLUE_LIGHT = HexColor("#DDEAF5")
RED = HexColor("#D94B45")
RED_LIGHT = HexColor("#F7E2E0")
ORANGE = HexColor("#E58A32")
GREEN_LIGHT = HexColor("#E4F0DC")
WHITE = HexColor("#FFFFFF")


def sx(value):
    return value * SCALE


def text(drawing, x, y, value, size=9.0, color=BLACK, anchor="middle",
         bold=False, italic=False):
    if bold and italic:
        font = "Times-BoldItalic"
    elif bold:
        font = "Times-Bold"
    elif italic:
        font = "Times-Italic"
    else:
        font = "Times-Roman"
    drawing.add(String(
        sx(x), sx(y), value, fontName=font, fontSize=size,
        fillColor=color, textAnchor=anchor,
    ))


def line(drawing, x1, y1, x2, y2, color=BLACK, width=1.2, dash=None):
    shape = Line(
        sx(x1), sx(y1), sx(x2), sx(y2),
        strokeColor=color, strokeWidth=width,
    )
    if dash:
        shape.strokeDashArray = dash
    drawing.add(shape)
    return shape


def box(drawing, x, y, w, h, color=GRAY, width=1.0, dash=None,
        fill=None):
    shape = Rect(
        sx(x), sx(y), sx(w), sx(h),
        strokeColor=color, strokeWidth=width, fillColor=fill,
    )
    if dash:
        shape.strokeDashArray = dash
    drawing.add(shape)
    return shape


def arrow(drawing, x1, y1, x2, y2, color=BLACK, width=1.2):
    line(drawing, x1, y1, x2, y2, color, width)
    head = 0.10
    drawing.add(Polygon([
        sx(x2), sx(y2),
        sx(x2 - head), sx(y2 + 0.055),
        sx(x2 - head), sx(y2 - 0.055),
    ], fillColor=color, strokeColor=color, strokeWidth=0.4))


def node(drawing, x, y, kind, label=None, radius=0.115, bold=False):
    fill = USER if kind == "user" else ITEM
    stroke = USER_DARK if kind == "user" else ITEM_DARK
    drawing.add(Circle(
        sx(x), sx(y), sx(radius), fillColor=fill,
        strokeColor=stroke, strokeWidth=1.25 if bold else 0.9,
    ))
    if label:
        text(drawing, x, y - 0.025, label, 8.0, BLACK, bold=bold)


def signal_tag(drawing, x, y, label, face, edge, width):
    box(drawing, x, y, width, 0.34, edge, 0.8, fill=face)
    text(drawing, x + width / 2, y + 0.105, label, 8.2, edge, bold=True)


def target_pair(drawing, ux, ix, y, edge_color, dashed=False,
                u_label="u", i_label="i"):
    line(
        drawing, ux + 0.12, y, ix - 0.12, y,
        edge_color, 2.0, [5, 3] if dashed else None,
    )
    node(drawing, ux, y, "user", u_label, bold=True)
    node(drawing, ix, y, "item", i_label, bold=True)


def small_loss_curve(drawing, x, y, color):
    points = [
        (x, y + 0.24),
        (x + 0.20, y + 0.13),
        (x + 0.40, y + 0.10),
        (x + 0.60, y + 0.08),
        (x + 0.80, y + 0.07),
    ]
    for first, second in zip(points, points[1:]):
        line(drawing, first[0], first[1], second[0], second[1], color, 1.6)
    for px, py in points:
        drawing.add(Circle(sx(px), sx(py), 1.7, fillColor=color,
                           strokeColor=None))


def support_context(drawing, y, strong):
    # Target edge and bilateral neighborhoods.
    target_pair(
        drawing, 6.00, 8.52, y,
        BLUE if strong else RED,
        dashed=not strong,
    )
    box(drawing, 6.45, y - 0.54, 0.78, 1.08, GRAY, 0.8, [3, 2])
    box(drawing, 7.32, y - 0.54, 0.75, 1.08, GRAY, 0.8, [3, 2])
    text(drawing, 6.84, y + 0.66, "N(u) \\ {i}", 7.8, GRAY)
    text(drawing, 7.70, y + 0.66, "N(i) \\ {u}", 7.8, GRAY)

    item_positions = [(6.84, y + 0.32), (6.84, y), (6.84, y - 0.32)]
    user_positions = [(7.70, y + 0.32), (7.70, y), (7.70, y - 0.32)]
    for px, py in item_positions:
        node(drawing, px, py, "item", radius=0.085)
        line(drawing, 6.12, y, px - 0.09, py, LIGHT_GRAY, 0.85)
    for px, py in user_positions:
        node(drawing, px, py, "user", radius=0.085)
        line(drawing, px + 0.09, py, 8.40, y, LIGHT_GRAY, 0.85)

    if strong:
        # Multiple bilateral co-occurrence supports.
        for index in range(3):
            line(
                drawing,
                item_positions[index][0] + 0.09,
                item_positions[index][1],
                user_positions[index][0] - 0.09,
                user_positions[index][1],
                BLUE,
                1.65,
            )
        line(drawing, 6.93, y + 0.32, 7.61, y, ORANGE, 1.25)
        line(drawing, 6.93, y, 7.61, y - 0.32, ORANGE, 1.25)
    else:
        # Sparse and inconsistent context.
        line(drawing, 6.93, y + 0.32, 7.61, y - 0.32, LIGHT_GRAY, 0.9)


def build_drawing():
    drawing = Drawing(sx(13.8), sx(5.15))
    drawing.add(Rect(
        0, 0, sx(13.8), sx(5.15), fillColor=WHITE, strokeColor=None,
    ))

    # Panel (a): same dynamic evidence.
    box(drawing, 0.28, 0.58, 4.58, 4.22, LIGHT_GRAY, 0.9)
    text(drawing, 0.52, 4.49, "Loss-only view", 11.5, BLACK,
         "start", bold=True)
    text(drawing, 0.52, 4.18,
         "Two interactions can exhibit similar learning difficulty",
         8.7, GRAY, "start")

    target_pair(drawing, 0.92, 2.15, 3.20, RED, dashed=True,
                u_label="u1", i_label="i1")
    small_loss_curve(drawing, 2.72, 3.05, RED)
    text(drawing, 3.12, 3.52, "momentum loss", 7.8, GRAY)
    signal_tag(drawing, 3.72, 3.03, "high", RED_LIGHT, RED, 0.70)

    target_pair(drawing, 0.92, 2.15, 1.88, BLUE,
                u_label="u2", i_label="i2")
    small_loss_curve(drawing, 2.72, 1.73, BLUE)
    text(drawing, 3.12, 2.20, "momentum loss", 7.8, GRAY)
    signal_tag(drawing, 3.72, 1.71, "high", BLUE_LIGHT, BLUE, 0.70)

    box(drawing, 0.62, 0.86, 3.90, 0.48, GRAY, 0.8, [3, 2],
        VERY_LIGHT_GRAY)
    text(drawing, 2.57, 1.02,
         "Noisy-like or hard-but-clean?  Loss alone is ambiguous.",
         8.8, BLACK, bold=True)

    arrow(drawing, 4.97, 2.69, 5.35, 2.69, BLACK, 1.3)

    # Panel (b): complementary bilateral structure.
    box(drawing, 5.45, 0.58, 8.07, 4.22, LIGHT_GRAY, 0.9)
    text(drawing, 5.69, 4.49, "Structure--dynamics view", 11.5, BLACK,
         "start", bold=True)
    text(drawing, 5.69, 4.18,
         "Bilateral context separates distinct meanings of high loss",
         8.7, GRAY, "start")

    support_context(drawing, 3.32, strong=False)
    text(drawing, 8.90, 3.70, "Noisy-like edge", 9.6, RED, "start", bold=True)
    signal_tag(drawing, 8.90, 3.25, "high momentum risk", RED_LIGHT, RED, 1.50)
    signal_tag(drawing, 10.50, 3.25, "low user-side", VERY_LIGHT_GRAY,
               GRAY, 1.22)
    signal_tag(drawing, 11.82, 3.25, "low item-side", VERY_LIGHT_GRAY,
               GRAY, 1.22)
    arrow(drawing, 10.05, 2.98, 11.06, 2.98, RED, 1.2)
    signal_tag(drawing, 11.16, 2.81, "refine", RED_LIGHT, RED, 0.88)

    line(drawing, 5.74, 2.59, 13.22, 2.59, LIGHT_GRAY, 0.8)

    support_context(drawing, 1.77, strong=True)
    text(drawing, 8.90, 2.15, "Hard-but-clean edge", 9.6, BLUE,
         "start", bold=True)
    signal_tag(drawing, 8.90, 1.70, "high momentum risk", BLUE_LIGHT, BLUE,
               1.50)
    signal_tag(drawing, 10.50, 1.70, "high user-side", GREEN_LIGHT,
               USER_DARK, 1.22)
    signal_tag(drawing, 11.82, 1.70, "high item-side", GREEN_LIGHT,
               USER_DARK, 1.22)
    arrow(drawing, 10.05, 1.43, 11.06, 1.43, BLUE, 1.2)
    signal_tag(drawing, 11.16, 1.26, "retain", BLUE_LIGHT, BLUE, 0.88)

    # Compact legend, matching the restrained graph style of NT-SSM.
    node(drawing, 9.50, 4.48, "user", radius=0.075)
    text(drawing, 9.65, 4.43, "user", 7.7, GRAY, "start")
    node(drawing, 10.32, 4.48, "item", radius=0.075)
    text(drawing, 10.47, 4.43, "item", 7.7, GRAY, "start")
    line(drawing, 11.10, 4.48, 11.48, 4.48, BLUE, 1.7)
    text(drawing, 11.57, 4.43, "structural support", 7.7, GRAY, "start")
    line(drawing, 12.55, 4.48, 12.91, 4.48, RED, 1.7, [4, 2])
    text(drawing, 13.00, 4.43, "uncertain", 7.7, GRAY, "start")

    text(drawing, 2.57, 0.27, "(a) Loss-only ambiguity", 9.5, BLACK,
         bold=True)
    text(drawing, 9.49, 0.27, "(b) Bilateral structural evidence", 9.5,
         BLACK, bold=True)
    return drawing


def draw():
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    stem = "structure_dynamics_motivation"
    svg_path = FIGURE_DIR / (stem + ".svg")
    paper_pdf_path = FIGURE_DIR / (stem + ".pdf")
    pdf_path = PDF_DIR / (stem + ".pdf")
    png_path = FIGURE_DIR / (stem + ".png")

    drawing = build_drawing()
    renderSVG.drawToFile(drawing, str(svg_path))
    renderPDF.drawToFile(drawing, str(paper_pdf_path))
    shutil.copyfile(paper_pdf_path, pdf_path)

    renderer = shutil.which("pdftoppm")
    if renderer is None:
        bundled = Path(
            "/Users/chenyijun/.cache/codex-runtimes/"
            "codex-primary-runtime/dependencies/bin/override/pdftoppm"
        )
        renderer = str(bundled) if bundled.exists() else None
    if renderer is None:
        raise RuntimeError("pdftoppm is required to render the PNG preview")
    subprocess.run([
        renderer, "-png", "-r", "220", "-singlefile",
        str(pdf_path), str(png_path.with_suffix("")),
    ], check=True)
    print(svg_path)
    print(paper_pdf_path)
    print(png_path)


if __name__ == "__main__":
    draw()
