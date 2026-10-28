"""Function lengths, near-duplicate pairs (difflib on code without docstrings), commented-out code, colormaps."""
import ast, difflib, re
src = open("hypyp/viz.py").read()
lines = src.splitlines()
tree = ast.parse(src)
funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
print(f"total lines {len(lines)}, top-level functions {len(funcs)}")

def body_code(f):
    start = f.body[0].end_lineno if isinstance(f.body[0], ast.Expr) and isinstance(getattr(f.body[0], 'value', None), ast.Constant) else f.lineno
    return [l.strip() for l in lines[start:f.end_lineno] if l.strip() and not l.strip().startswith("#")]

info = {}
tot_doc = tot_code = 0
for f in funcs:
    total = f.end_lineno - f.lineno + 1
    doc = (f.body[0].end_lineno - f.body[0].lineno + 1) if isinstance(f.body[0], ast.Expr) else 0
    code = body_code(f)
    info[f.name] = code
    tot_doc += doc; tot_code += len(code)
    flag = "  >100 lines" if total > 100 else ""
    print(f"  {f.name:26s} L{f.lineno}-{f.end_lineno}: total={total:4d} docstring={doc:4d} code(non-blank)={len(code):3d}{flag}")
print(f"docstring lines {tot_doc} ({100*tot_doc/len(lines):.0f}%), code lines {tot_code}")

pairs = [("plot_sensors_2d_inter", "plot_sensors_2d_intra"), ("plot_sensors_3d_inter", "plot_sensors_3d_intra"),
         ("plot_sensors_2d_inter", "plot_sensors_3d_inter"), ("plot_links_2d_inter", "plot_links_3d_inter"),
         ("plot_links_2d_intra", "plot_links_3d_intra"), ("plot_links_2d_inter", "plot_links_2d_intra"),
         ("plot_2d_topomap_inter", "plot_2d_topomap_intra"), ("get_3d_heads_inter", "get_3d_heads_intra"),
         ("viz_2D_topomap_inter", "viz_3D_inter"), ("viz_2D_topomap_intra", "viz_3D_intra"), ("viz_2D_topomap_inter", "viz_2D_headmodel_inter")]
print("\nnear-duplicate pairs (difflib ratio on stripped code lines, identical lines):")
for a, b in pairs:
    sm = difflib.SequenceMatcher(None, info[a], info[b])
    same = sum(bl.size for bl in sm.get_matching_blocks())
    print(f"  {a:24s} vs {b:24s}: ratio={sm.ratio():.2f}, identical lines {same}/{len(info[a])},{len(info[b])}")

# commented-out code heuristic
cc = [i + 1 for i, l in enumerate(lines) if re.match(r"^\s*#\s*(plt\.|ax\.|np\.|[a-z_]+\s*=|for |if |return |import |def )", l)]
print(f"\ncommented-out code lines (heuristic): {len(cc)} {cc}")
print("TODO/FIXME:", len(re.findall(r"TODO|FIXME", src)))
print("colormaps:", sorted(set(re.findall(r'colormaps\[\"([A-Za-z_]+)\"\]', src))))
print("type hints 'ax: str':", len(re.findall(r"ax: str", src)))
print("plt.show() calls in library code:", sum(1 for l in lines if l.strip() == "plt.show()"))
print("plt.close calls:", src.count("plt.close"))
print("Basehead.obj refs:", src.count("Basehead.obj"))
print("unused imports check:", [name for name in ["Path", "copy", "Union", "ExitStack", "atexit", "math", "glm", "Mesh", "Camera", "xwt", "meshio", "importlib_resources"] if len(re.findall(r"\b" + name + r"\b", src)) <= 1])
