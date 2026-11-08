"""Check that ecg_unet.py is identical to the architecture in YOUR original file.

    python verify_architecture.py path/to/your_original_file.py

Works with a notebook export too: only the six model classes are read, the other lines are ignored.
It compares (1) the code of every class (docstrings, comments and type hints are ignored) and
(2) the behaviour: same parameter names/shapes and the same output for the same weights and input.
"""
import ast
import difflib
import math
import sys
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import ecg_unet

NAMES = ["Attention1d", "ODConv1d", "ODConvBlock", "ConvBlock", "UpBlock", "ECGUNet"]


class _Strip(ast.NodeTransformer):
    """Remove docstrings and type hints so only the real code is compared."""

    def _doc(self, node):
        b = node.body
        if b and isinstance(b[0], ast.Expr) and isinstance(getattr(b[0], "value", None), ast.Constant) \
                and isinstance(b[0].value.value, str):
            node.body = b[1:] or [ast.Pass()]

    def visit_ClassDef(self, node):
        self._doc(node)
        self.generic_visit(node)
        return node

    def visit_FunctionDef(self, node):
        self._doc(node)
        node.returns = None
        a = node.args
        for arg in a.posonlyargs + a.args + a.kwonlyargs + [x for x in (a.vararg, a.kwarg) if x]:
            arg.annotation = None
        self.generic_visit(node)
        return node

    def visit_AnnAssign(self, node):
        self.generic_visit(node)
        return None if node.value is None else ast.Assign(targets=[node.target], value=node.value, lineno=0)


def normalised(node):
    import copy
    return ast.unparse(ast.fix_missing_locations(_Strip().visit(copy.deepcopy(node))))


def main(path):
    tree = ast.parse(open(path, encoding="utf-8").read())
    orig = {}
    for n in tree.body:
        if isinstance(n, ast.ClassDef) and n.name in NAMES:
            orig[n.name] = n                              # the last definition in the file wins
    ok = True
    mine_cls = {c.name: c for c in ast.parse(open(ecg_unet.__file__, encoding="utf-8").read()).body
                if isinstance(c, ast.ClassDef)}
    for name in NAMES:
        if name not in orig:
            print(f"[MISSING] {name} not found in {path}")
            ok = False
            continue
        a, b = normalised(orig[name]), normalised(mine_cls[name])
        if a == b:
            print(f"[same code]  {name}")
        else:
            ok = False
            print(f"[DIFFERENT]  {name}")
            print("\n".join(difflib.unified_diff(a.splitlines(), b.splitlines(), "your file", "ecg_unet.py", lineterm="", n=1)))
    if not all(n in orig for n in NAMES):
        return False

    ns = {"torch": torch, "nn": nn, "F": F, "Callable": Callable, "math": math, "np": np}
    exec(compile(ast.fix_missing_locations(ast.Module([orig[n] for n in NAMES], [])), path, "exec"), ns)
    for bf, ks, kn in [(8, 7, 1), (8, 9, 4)]:
        torch.manual_seed(0)
        mine = ecg_unet.ECGUNet(1, 1, bf, ks, kn).eval()
        theirs = ns["ECGUNet"](1, 1, bf, ks, kn).eval()
        sm, st = mine.state_dict(), theirs.state_dict()
        same_keys = list(sm) == list(st) and all(sm[k].shape == st[k].shape for k in sm)
        theirs.load_state_dict(sm)
        x = torch.randn(3, 1, 1000)
        with torch.no_grad():
            diff = float((mine(x) - theirs(x)).abs().max())
        good = same_keys and diff == 0.0
        ok &= good
        print(f"[{'same behaviour' if good else 'DIFFERENT'}]  base_filters={bf} kernel_size={ks} kernel_num={kn}: "
              f"parameter names/shapes {'match' if same_keys else 'DIFFER'}, max output difference {diff:.1e}")
    print("\nRESULT:", "the architecture is IDENTICAL" if ok else "DIFFERENCES FOUND")
    return ok


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    sys.exit(0 if main(sys.argv[1]) else 1)
