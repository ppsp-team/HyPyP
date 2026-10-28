# Issue draft: `hypyp.viz` link plots: unchecked shapes, diagonal spikes, negative line widths, mismatched child heads

Found by the 2026-10-13 review of 0.6.2 (F20 and the `viz` section; `scripts/viz/s2_index_mapping.py`, `s3_*.py`, `s5_*.py`, `s8_*.py`, PNGs alongside). What is right and should stay: both heads are proper rotations (not reflections), `C[i, j]` joins participant 1 sensor i to participant 2 sensor j in 2D and 3D, no public function mutates the caller's Epochs or montage, signed values keep their sign, no `jet`.

## Silent misattribution on an unsliced matrix (F20)

`plot_links_2d_inter:1023-1027` and `plot_links_3d_inter:1397-1401` loop `e1 in range(n1)`, `e2 in range(n2)` with no check that `C.shape == (n1, n2)`. `compute_sync` returns `(2n, 2n)`; passed unsliced, the inter quadrant is never read and participant 1's intra block is drawn as inter-brain links: an entry at `[0, 30]` (P1 intra) draws Fp1 of P1 to O2 of P2, an entry in the true inter quadrant draws nothing, no error. Fix: `assert C.shape == (len(loc1), len(loc2))` with a message, and `C1.shape == (n1, n1)` for the intra plots.

## Diagonal self-loops (`plot_links_2d_intra:1207-1232`, `plot_links_3d_intra:1596-1628`)

`e1 == e2` passes the threshold; with `steps >= 3` the bezier with `p0 == p1` degenerates to a spike pointing away from the head centre, 0.75× the sensor-to-centre distance. The `'auto'` threshold (`:1193, :1583`) includes the diagonal: with a PLV-style matrix only self-loops are drawn and no real link (558 segments on the sample data). The tutorial zeroes the diagonal by hand. Fix: `if e1 == e2: continue`, and exclude the diagonal from the auto threshold.

## Negative links get negative line widths (`:1035-1037, 1217, 1231, 1411-1413, 1608, 1625`)

The width of a negative value divides by `nanmax(C) - threshold`, the positive range; whenever `|min| > max` (typical for ImCoh with one dominant sign) widths go negative, Agg draws `|lw|`, and width is non-monotonic in |C|. Fix: width from `|C|` against `max(|C|)`.

## `children` / `child_head` inconsistent (`plot_2d_topomap_inter:1718-1741`, `plot_sensors_2d_inter:389-409`, `get_3d_heads_inter:2068-2073`, `plot_sensors_3d_inter:664-684`)

In 2D with `child_head=True` (the README example) the outline shrinks the left head and the sensors shrink the right one; in 3D with `child_head=False` the meshes are equal and the sensors are not. Fix: pass `child_head` through to the outline and mesh functions and use one convention.

## Smaller

- 2D sensor and link functions draw on `plt.gca()` while the outline takes `ax` (`:429-455, 562-588, 961-970, 1142-1151`): in a multi-panel figure they land on different axes
- `plt.show()` in five wrappers, no `plt.close` anywhere; wrappers return `ax`, not the figure
- NaN in the matrix gives a NaN `'auto'` threshold and nothing drawn, no warning (`:1010, 1385`); `np.min(C[:])` against `np.nanmax` paints negative links black
- `transform()` mutates its argument and corrupts Z for any `rotZ` other than ±π/2 (`:127-145`)
- both triangles and both directions are drawn, so a directed matrix loses its asymmetry silently
- `plot_significant_sensors` projects positions orthographically (`:2253-2257`): occipital sensors fall outside the outline; pass `epochs.info` to `plot_topomap`
- `plot_xwt` crashes on every input and is called nowhere (see the `xwt` issue)
- about 550 to 650 of the 1125 code lines are near-duplicates (inter/intra, 2D/3D); `viz.py` has no test at all
