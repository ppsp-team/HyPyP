import json, re, sys, glob, os
os.chdir("/home/gdumas/Github/HyPyP")
for nb in sorted(glob.glob("tutorial/*.ipynb")):
    d = json.load(open(nb))
    code = "\n".join("".join(c["source"]) for c in d["cells"] if c["cell_type"] == "code")
    calls = sorted(set(re.findall(r"\b((?:analyses|prep|stats|utils|viz|io|datasets|hypyp(?:\.\w+)*|XDFImport|Study|Dyad|Recording|ComplexMorletWavelet|SyntheticSignal|PairSignals|DataBrowser)\.\w+|XDFImport|Study|Dyad|Recording|ComplexMorletWavelet|SyntheticSignal|PairSignals|DataBrowser)\s*\(", code)))
    imports = sorted(set(l.strip() for l in code.splitlines() if re.match(r"\s*(import|from)\s", l)))
    print("=====", nb, "cells:", len(d["cells"]))
    print("IMPORTS:", imports)
    print("CALLS:", calls)
