# ExecuPack brand kit

The mark is the **Lidded Box**: a box seen from above with its lid on. The lid is ember and
overhangs the box, so its rim reads as a stripe at any size. The faces are parted by open cuts.
Every file here is written by `tools/design/mark.py`. Change the numbers there and rerun it
(`--png` also renders the PNGs).

| File | Use |
|---|---|
| `mark.svg`, `mark-dark.svg` | The mark on light and on dark grounds |
| `mark-small.svg`, `mark-small-dark.svg` | Below 40 px: the same shape with wider cuts (3.4 for 2.6) |
| `mark-mono.svg` | One colour; the cut keeps the lid apart from the box |
| `lockup.svg`, `lockup-dark.svg` | Mark and wordmark (Red Hat Display Bold, outlined) |
| `icon.svg` / `.png` | 512 px square icon for the GitHub and Hugging Face avatars, opaque ink |
| `social-preview.svg` / `.png` | 1280 × 640 GitHub social preview |
| `mark.png` | 1024 px mark on transparency |

**Family.** ExecuPack packs and exports models; ExecuServe serves them on the phone. The two marks
share ExecuServe's isometric angle (1.8 across for 1 down), its cut width, its palette and its
lockup, and nothing else: ExecuServe's Block is a chip on legs with a small ember die, ExecuPack's
is a box with a whole ember lid and no legs. Don't borrow the Block's parts for this mark, or this
mark's for the Block.

**Colour.** Ink `#262626` or paper `#F3F4F7` for the box, ember `#EE4C2C` for the lid and nowhere
else in the mark.

**Space.** Keep clear space of one rim depth on every side. In the icon, the farthest point sits 30
from the centre of a 108 canvas, the circle every avatar crop keeps.

**Don't** recolour the box's faces separately, close the cut under the lid, add outlines or
shadows, or rotate the box. Don't place PyTorch's or ExecuTorch's logo inside or beside the mark as
one unit; credit ExecuTorch in text.

The wordmark's font, Red Hat Display, is vendored in `tools/design/fonts/` under the SIL Open Font
License (`OFL.txt`).
