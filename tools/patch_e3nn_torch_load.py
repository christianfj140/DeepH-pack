"""Patch e3nn 0.4.4 for PyTorch >= 2.6.

DeepH pins old e3nn versions. PyTorch 2.6 changed torch.load's default
weights_only value, which breaks e3nn's bundled constants.pt load.
"""

from pathlib import Path
import importlib.util


spec = importlib.util.find_spec("e3nn")
if spec is None or spec.origin is None:
    raise SystemExit("e3nn is not installed in this Python environment")

wigner_path = Path(spec.origin).parent / "o3" / "_wigner.py"
old = "_Jd, _W3j_flat, _W3j_indices = torch.load(os.path.join(os.path.dirname(__file__), 'constants.pt'))"
new = "_Jd, _W3j_flat, _W3j_indices = torch.load(os.path.join(os.path.dirname(__file__), 'constants.pt'), weights_only=False)"

text = wigner_path.read_text()
if new in text:
    print(f"Already patched: {wigner_path}")
elif old in text:
    wigner_path.write_text(text.replace(old, new))
    print(f"Patched: {wigner_path}")
else:
    raise SystemExit(f"Expected torch.load line not found in {wigner_path}")
