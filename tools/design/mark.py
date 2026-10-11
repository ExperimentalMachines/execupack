"""
The ExecuPack mark, the Lidded Box, drawn once and written everywhere it appears:

    docs     docs/brand: marks, lockups, icon, social preview (and their PNGs)

    python3 tools/design/mark.py           # SVG outputs
    python3 tools/design/mark.py --png     # also PNGs, rendered by Chrome

A box seen from above with its lid on: the ember lid overhangs the box, so its rim reads as
a stripe at any size. The faces are parted by cuts of constant width, the way ExecuServe's
Block is (execuserve/tools/design/mark.py): the two marks share the projection, the cut, the
palette and the wordmark, and nothing else. The cuts are real gaps, not lines painted in a
background colour, so the mark sits on any ground.

Everything is straight lines in a 100-unit box, so moving the mark between formats is
arithmetic on points: no masks, filters or arcs. The lockup needs uharfbuzz and fontTools
(pip install uharfbuzz fonttools) and the vendored Red Hat Display in fonts/.
"""

import math
import pathlib
import shutil
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
FONT = ROOT / "tools/design/fonts/RedHatDisplay-Bold.ttf"
INK, PAPER, EMBER = "#262626", "#F3F4F7", "#EE4C2C"

# ExecuServe's isometric axes: one unit along u or v moves 1.8 across and 1.0 down. The
# vertical is drawn at the same true length, so the box is a cube, not the Block's slab.
UP = math.hypot(1.8, 1.0)

# The box: a 20-unit cube. The lid is the top 4 units and overhangs each side by 1.2.
SIDE, HEIGHT, RIM, OVERHANG = 20.0, 20.0, 4.0, 1.2

# Full drawing for 40 px and up; the small one (wider cuts, same shape) for favicons and
# anything under 40 px, where a 2.6 cut falls below a pixel.
FULL = dict(cut=2.6)
SMALL = dict(cut=3.4)


# ---- geometry ----------------------------------------------------------------


def iso(u, v, w):
    return 1.8 * (u - v), (u + v) - w * UP


def _line_cross(p1, d1, p2, d2):
    """Intersection of two lines given as point + direction."""
    det = d1[0] * d2[1] - d1[1] * d2[0]
    t = ((p2[0] - p1[0]) * d2[1] - (p2[1] - p1[1]) * d2[0]) / det
    return p1[0] + t * d1[0], p1[1] + t * d1[1]


def inset(poly, moves):
    """Move the polygon's edges inward, edge i by moves.get(i, 0), and keep the others. Two
    faces sharing an edge each give up half the cut, so the gap is the full cut wide along
    its whole length (ExecuServe's mark.py does the same)."""
    n = len(poly)
    cx = sum(p[0] for p in poly) / n
    cy = sum(p[1] for p in poly) / n
    lines = []
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        dx, dy = b[0] - a[0], b[1] - a[1]
        length = math.hypot(dx, dy)
        nx, ny = -dy / length, dx / length
        if (cx - a[0]) * nx + (cy - a[1]) * ny < 0:  # point the normal inward
            nx, ny = -nx, -ny
        e = moves.get(i, 0.0)
        lines.append(((a[0] + nx * e, a[1] + ny * e), (dx, dy)))
    return [_line_cross(*lines[i - 1], *lines[i]) for i in range(n)]


def box(spec):
    """Polygons of the mark: the box's two faces, then the lid (top and two rim faces)."""
    e = spec["cut"] / 2
    a, b, low = -OVERHANG, SIDE + OVERHANG, HEIGHT - RIM
    top = [iso(a, a, HEIGHT), iso(b, a, HEIGHT), iso(b, b, HEIGHT), iso(a, b, HEIGHT)]
    rim_left = [iso(a, b, HEIGHT), iso(b, b, HEIGHT), iso(b, b, low), iso(a, b, low)]
    rim_right = [iso(b, a, HEIGHT), iso(b, b, HEIGHT), iso(b, b, low), iso(b, a, low)]
    lid = [inset(top, {1: e, 2: e}), inset(rim_left, {0: e, 1: e}), inset(rim_right, {0: e, 1: e})]
    # The box's faces show only below the lid: their top edge is the rim's lower edge, moved
    # down by the whole cut (the lid is in front, so the gap is all the box's), and the two
    # faces part at the front corner like the lid's.
    corner = iso(SIDE, SIDE, 0)
    left = [
        _line_cross(iso(0, SIDE, 0), (0, 1), iso(a, b, low), (1.8, 1.0)),
        _line_cross(corner, (0, 1), iso(a, b, low), (1.8, 1.0)),
        corner,
        iso(0, SIDE, 0),
    ]
    right = [
        _line_cross(corner, (0, 1), iso(b, a, low), (-1.8, 1.0)),
        _line_cross(iso(SIDE, 0, 0), (0, 1), iso(b, a, low), (-1.8, 1.0)),
        iso(SIDE, 0, 0),
        corner,
    ]
    body = [inset(left, {0: 2 * e, 1: e}), inset(right, {0: 2 * e, 3: e})]
    return body, lid


def bounds(polys):
    xs = [p[0] for poly in polys for p in poly]
    ys = [p[1] for poly in polys for p in poly]
    return min(xs), min(ys), max(xs), max(ys)


def fit(body, lid, size, box_size):
    """Scale and centre the mark so its bounding box fits `box_size` inside a `size` square."""
    x0, y0, x1, y1 = bounds(body + lid)
    s = box_size / max(x1 - x0, y1 - y0)
    ox, oy = size / 2 - s * (x0 + x1) / 2, size / 2 - s * (y0 + y1) / 2
    move = lambda poly: [(ox + s * x, oy + s * y) for x, y in poly]  # noqa: E731
    return [move(p) for p in body], [move(p) for p in lid]


def fit_circle(body, lid, size, radius):
    """Scale and centre so the farthest point sits on a circle, as ExecuServe's launcher
    icon does: avatars and app tiles are cropped by masks of any shape."""
    polys = body + lid
    x0, y0, x1, y1 = bounds(polys)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    far = max(math.hypot(x - cx, y - cy) for poly in polys for x, y in poly)
    s = radius / far
    move = lambda poly: [(size / 2 + s * (x - cx), size / 2 + s * (y - cy)) for x, y in poly]  # noqa: E731
    return [move(p) for p in body], [move(p) for p in lid]


def num(v):
    return f"{v:.2f}".rstrip("0").rstrip(".")


def d(polys):
    return "".join("M" + "L".join(f"{num(x)} {num(y)}" for x, y in poly) + "Z" for poly in polys)


# ---- brand kit -----------------------------------------------------------------

GENERATED = "Generated by tools/design/mark.py; edit the numbers there, not this file."


def write(rel, text):
    path = ROOT / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    print("wrote", rel)


def text_path(text, font_file, size, tracking=0.0):
    """HarfBuzz shaping (so kerning applies) and fontTools outlines; baseline at y = 0."""
    import uharfbuzz as hb
    from fontTools.pens.svgPathPen import SVGPathPen
    from fontTools.pens.transformPen import TransformPen
    from fontTools.ttLib import TTFont

    face = hb.Face(hb.Blob.from_file_path(str(font_file)))
    buf = hb.Buffer()
    buf.add_str(text)
    buf.guess_segment_properties()
    hb.shape(hb.Font(face), buf, {"kern": True, "liga": True})
    tt = TTFont(str(font_file))
    glyphs, order = tt.getGlyphSet(), tt.getGlyphOrder()
    scale = size / face.upem
    pen = SVGPathPen(glyphs)
    x = 0.0
    for info, pos in zip(buf.glyph_infos, buf.glyph_positions, strict=True):
        glyphs[order[info.codepoint]].draw(TransformPen(pen, (scale, 0, 0, -scale, x + pos.x_offset * scale, 0)))
        x += pos.x_advance * scale + tracking * size
    return pen.getCommands(), x - tracking * size


def mark_svg(body_colour, lid_colour=EMBER, spec=FULL):
    body, lid = fit(*box(spec), 100, 96.0)
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><path fill="{body_colour}" d="{d(body)}"/>'
        f'<path fill="{lid_colour}" d="{d(lid)}"/></svg>\n'
    )


def lockup_svg(ink):
    """Mark and wordmark: Red Hat Display Bold, the mark as tall as 2.15 cap heights, the
    proportions of ExecuServe's lockup."""
    size = 64.0
    word, advance = text_path("ExecuPack", FONT, size, -0.012)
    cap = 0.7 * size
    height = 2.15 * cap
    body, lid = fit(*box(FULL), height, height)
    x0, _, x1, _ = bounds(body + lid)
    move = lambda poly: [(x - x0, y) for x, y in poly]  # noqa: E731
    tx = (x1 - x0) + 0.3 * size
    width = tx + advance
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {num(width)} {num(height)}">'
        f'<path fill="{ink}" d="{d([move(p) for p in body])}"/><path fill="{EMBER}" d="{d([move(p) for p in lid])}"/>'
        f'<path fill="{ink}" transform="translate({num(tx)} {num(height / 2 + cap / 2)})" d="{word}"/></svg>\n'
    )


def icon_svg():
    """Square icon for avatars (GitHub, Hugging Face): opaque ink, paper box, ember lid, the
    farthest point 30 from the centre of a 108 canvas, inside the circle any crop keeps."""
    body, lid = fit_circle(*box(FULL), 108, 30.0)
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="18 18 72 72"><rect x="18" y="18" width="72" height="72" '
        f'fill="{INK}"/><path fill="{PAPER}" d="{d(body)}"/><path fill="{EMBER}" d="{d(lid)}"/></svg>\n'
    )


def social_preview_svg():
    """1280 x 640 GitHub social preview: the lockup in paper on ink, nothing else."""
    lock = lockup_svg(PAPER)
    w, h = (float(v) for v in lock.split('viewBox="0 0 ')[1].split('"')[0].split())
    s = 760 / w
    inner = lock[lock.index(">") + 1 : lock.rindex("</svg>")]
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1280 640"><rect width="1280" height="640" fill="{INK}"/>'
        f'<g transform="translate({num(640 - w * s / 2)} {num(320 - h * s / 2)}) scale({num(s)})">{inner}</g></svg>\n'
    )


def png(name, w, h, bg):
    chrome = shutil.which("google-chrome") or "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
    page = ROOT / "build" / f"{name}.html"
    page.parent.mkdir(exist_ok=True)
    page.write_text(
        f"<html><body style='margin:0'><img src='{ROOT}/docs/brand/{name}.svg' "
        f"style='display:block;width:{w}px;height:{h}px'></body></html>"
    )
    subprocess.run(
        [chrome, "--headless=new", "--disable-gpu", "--hide-scrollbars", "--force-device-scale-factor=1",
         f"--default-background-color={bg}", f"--window-size={w},{h}",
         f"--screenshot={ROOT}/docs/brand/{name}.png", page.as_uri()],
        capture_output=True, check=True,
    )  # fmt: skip
    page.unlink()
    print("wrote", f"docs/brand/{name}.png")


def brand_kit(with_png):
    write("docs/brand/mark.svg", mark_svg(INK))
    write("docs/brand/mark-dark.svg", mark_svg(PAPER))
    write("docs/brand/mark-small.svg", mark_svg(INK, spec=SMALL))
    write("docs/brand/mark-small-dark.svg", mark_svg(PAPER, spec=SMALL))
    write("docs/brand/mark-mono.svg", mark_svg(INK, INK))
    write("docs/brand/lockup.svg", lockup_svg(INK))
    write("docs/brand/lockup-dark.svg", lockup_svg(PAPER))
    write("docs/brand/icon.svg", icon_svg())
    write("docs/brand/social-preview.svg", social_preview_svg())
    if with_png:
        png("mark", 1024, 1024, "00000000")
        png("icon", 512, 512, "ff262626")
        png("social-preview", 1280, 640, "ff262626")


if __name__ == "__main__":
    brand_kit("--png" in sys.argv)
