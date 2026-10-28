import ast, json, glob, inspect, os, re, sys, warnings
warnings.filterwarnings("ignore")
os.chdir("/home/gdumas/Github/HyPyP")
import hypyp, hypyp.fnirs as fnirs, hypyp.wavelet as wv, hypyp.xdf as xdf, hypyp.signal as sig, mne
from hypyp import analyses, prep, stats, viz, utils, datasets
NS = {"analyses": analyses, "prep": prep, "stats": stats, "viz": viz, "utils": utils, "datasets": datasets,
      "fnirs": fnirs, "hypyp": hypyp, "mne": mne, "io": mne.io}
for n in ["ComplexMorletWavelet","ComplexGaussianWavelet","BaseWavelet","CWT","CoherenceDataFrame","PairSignals"]:
    NS[n] = getattr(wv, n, None)
NS["XDFImport"] = xdf.XDFImport; NS["SyntheticSignal"] = sig.SyntheticSignal
NS["Task"] = utils.Task; NS["Study"]=fnirs.Study; NS["Dyad"]=fnirs.Dyad; NS["Recording"]=fnirs.Recording
NS["DataBrowser"]=getattr(fnirs,"DataBrowser",None)

def resolve(node):
    if isinstance(node, ast.Name): return node.id
    if isinstance(node, ast.Attribute):
        b = resolve(node.value); return f"{b}.{node.attr}" if b else None
    return None

def lookup(dotted):
    parts = dotted.split("."); obj = NS.get(parts[0])
    if obj is None: return None
    for p in parts[1:]:
        obj = getattr(obj, p, None)
        if obj is None: return None
    return obj

for nb in sorted(glob.glob("tutorial/*.ipynb")):
    d = json.load(open(nb)); problems = []
    for c in d["cells"]:
        if c["cell_type"] != "code": continue
        src = "\n".join(l for l in "".join(c["source"]).splitlines() if not l.lstrip().startswith(("%","!")))
        try: tree = ast.parse(src)
        except SyntaxError as e: problems.append(f"SYNTAX {e}"); continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call): continue
            name = resolve(node.func)
            if not name or name.split(".")[0] not in NS: continue
            if name.split(".")[0] in ("mne","io"): continue
            obj = lookup(name)
            if obj is None:
                problems.append(f"MISSING {name}"); continue
            try: sigt = inspect.signature(obj)
            except (ValueError, TypeError): continue
            params = sigt.parameters
            has_var_kw = any(p.kind == p.VAR_KEYWORD for p in params.values())
            kws = [k.arg for k in node.keywords if k.arg]
            bad = [k for k in kws if k not in params and not has_var_kw]
            if bad: problems.append(f"BAD KWARG {name}({', '.join(bad)}) ; sig={list(params)}")
            npos = len(node.args); maxpos = sum(1 for p in params.values() if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)) 
            if inspect.isclass(obj) or not any(p.kind==p.VAR_POSITIONAL for p in params.values()):
                if npos > maxpos: problems.append(f"TOO MANY POSITIONAL {name}: {npos} > {maxpos}")
    print("=====", nb)
    for p in sorted(set(problems)): print("  ", p)
    if not problems: print("   (no mismatch found)")
