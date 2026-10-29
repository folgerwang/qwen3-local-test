# Runs qwen.exe --eval for each backend / prompt / split and compares top-10 logits to the fp32 reference.
import os, subprocess, sys
exe = os.path.abspath("build/Release/qwen.exe")
backends = sys.argv[1].split(",") if len(sys.argv) > 1 else ["cpu", "vulkan"]
cases = [("short", "tests/eval_ids.txt", "tests/ref_short_top10.txt", [0, 5]),
         ("long", "tests/eval_long_ids.txt", "tests/ref_long_top10.txt", [0, 3])]
ok = True
for be in backends:
    for name, ids_f, ref_f, splits in cases:
        if not os.path.exists(ref_f) or os.path.getsize(ref_f) == 0: continue
        ref = [(int(a), float(b)) for a, b in (l.split() for l in open(ref_f) if l.strip())]
        ids = open(ids_f).read().strip()
        for sp in splits:
            r = subprocess.run([exe, "--backend", be, "--eval", ids, "--eval-split", str(sp), "--ctx", "2048"],
                               capture_output=True, text=True)
            got = [(int(a), float(b)) for a, b in (l.split() for l in r.stdout.splitlines() if l.strip())]
            if len(got) != 10:
                print(f"{be:7s} {name:5s} split={sp}: FAILED\n{r.stderr[-2000:]}"); ok = False; continue
            refd = dict(ref)
            diff = max(abs(v - refd[i]) if i in refd else 99 for i, v in got[:5])
            top1 = got[0][0] == ref[0][0]
            status = "OK " if top1 and diff < 0.05 else "BAD"
            ok &= status == "OK "
            print(f"{status} {be:7s} {name:5s} split={sp}: top1 {got[0][0]} (ref {ref[0][0]}), max|dlogit| top5 = {diff:.4f}")
print("ALL OK" if ok else "SOME FAILED")
