import re, tomllib, collections
lock = tomllib.load(open("/home/gdumas/Github/HyPyP/uv.lock","rb"))
rows=[]
for p in lock["package"]:
    wheels = p.get("wheels", [])
    if not wheels: 
        rows.append((p["name"], p["version"], "sdist-only")); continue
    tags = [w["url"].rsplit("/",1)[-1] for w in wheels]
    if any("py3-none-any" in t or "abi3" in t or "py2.py3" in t for t in tags): continue
    cps = sorted({m for t in tags for m in re.findall(r"cp3\d+", t)})
    if cps and "cp314" not in cps:
        rows.append((p["name"], p["version"], "no cp314 wheel; has "+",".join(cps)))
for r in rows: print(r)
