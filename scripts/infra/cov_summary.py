import json
d = json.load(open("/home/gdumas/.claude/jobs/9fe7d438/tmp/cov.json"))
t = d["totals"]
print("TOTAL percent_covered=%.1f  statements=%d missing=%d  files=%d" % (t["percent_covered"], t["num_statements"], t["missing_lines"], len(d["files"])))
rows = [(f, v["summary"]["percent_covered"], v["summary"]["num_statements"]) for f, v in d["files"].items() if v["summary"]["num_statements"] > 0]
rows.sort(key=lambda r: r[1])
print("Lowest 10:")
for f, p, n in rows[:10]: print("  %-72s %5.1f%%  (%d stmts)" % (f, p, n))
print("Fully covered files:", sorted(f for f, p, n in rows if p == 100.0))
# subpackage aggregates
import collections
agg = collections.defaultdict(lambda: [0,0])
for f, v in d["files"].items():
    key = "/".join(f.split("/")[:2]) if f.count("/")>=2 else f
    agg[key][0] += v["summary"]["covered_lines"]; agg[key][1] += v["summary"]["num_statements"]
print("Aggregates:")
for k,(c,n) in sorted(agg.items(), key=lambda kv: kv[1][0]/max(kv[1][1],1)):
    print("  %-30s %5.1f%% (%d/%d)" % (k, 100*c/max(n,1), c, n))
