import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools/codecheck"))
import numpy as np
from ptecodes import load_pte

a_pte, a = load_pte(Path(sys.argv[1]))
b_pte, b = load_pte(Path(sys.argv[2]))
print("named keys", len(a), len(b), "same key set:", set(a) == set(b))
diff = [k for k in a if k in b and a[k] != b[k]]
print("differing entries:", len(diff), "of", len(a))
for k in diff[:12]:
    x, y = a[k], b[k]
    note = ""
    if len(x) == len(y) and len(x) % 4 == 0:
        fx, fy = np.frombuffer(x, np.float32), np.frombuffer(y, np.float32)
        if np.isfinite(fx).all() and np.isfinite(fy).all():
            note = f"as f32: max|d|={np.abs(fx - fy).max():.3g} max|x|={np.abs(fx).max():.3g}"
    nbytes = sum(1 for i in range(min(len(x), len(y))) if x[i] != y[i]) if len(x) < 5_000_000 else "big"
    print(" ", k[:70], len(x), len(y), "bytes differing:", nbytes, note)
# constant buffer in program
ca = [bytes(t.storage) for t in a_pte.program.constant_buffer] if a_pte.program.constant_buffer else []
cb = [bytes(t.storage) for t in b_pte.program.constant_buffer] if b_pte.program.constant_buffer else []
print("program constant_buffer entries", len(ca), len(cb), "equal:", ca == cb)
seg_a = getattr(a_pte, "constant_data", None)
for i, (x, y) in enumerate(zip(ca, cb)):
    if x != y:
        note = ""
        if len(x) == len(y) and len(x) % 4 == 0:
            fx, fy = np.frombuffer(x, np.float32), np.frombuffer(y, np.float32)
            note = f"f32 max|d|={np.nanmax(np.abs(fx - fy)):.3g} max|x|={np.nanmax(np.abs(fx)):.3g}"
            ix, iy = np.frombuffer(x, np.int8), np.frombuffer(y, np.int8)
            note += f" int8 differing={int((ix != iy).sum())}/{len(ix)}"
        print(" const", i, len(x), len(y), note)
