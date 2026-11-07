# Runs qwen.exe --eval for each backend / prompt / split and compares top-10 logits to the fp32 reference.
# usage: python tests/check.py [cpu,vulkan] [35b|flash]
import os, subprocess, sys
exe = os.path.abspath("build/Release/qwen.exe")
backends = sys.argv[1].split(",") if len(sys.argv) > 1 else ["cpu", "vulkan"]
model = sys.argv[2] if len(sys.argv) > 2 else "35b"
if model == "flash":
    # the long prompt (3000 tokens) is past the indexer budget, so it exercises the sparse attention selection
    extra = ["--model", "models/Qwen3.8-Flash-Next-FP8", "--ctx", "4096"]
    cases = [("short", "tests/eval_ids.txt", "tests/ref_flash_short_top10.txt", [0, 5]),
             ("long", "tests/eval_flash_long_ids.txt", "tests/ref_flash_long_top10.txt", [0, 3])]
else:
    extra = ["--ctx", "2048"]
    cases = [("short", "tests/eval_ids.txt", "tests/ref_short_top10.txt", [0, 5]),
             ("long", "tests/eval_long_ids.txt", "tests/ref_long_top10.txt", [0, 3])]
    # image prompt (vulkan only; oracle: tests/ref_logits_vision.py). Split 3 decodes the tokens after the
    # image one by one, exercising the shifted rotary positions on the decode path.
    if "vulkan" in backends:
        cases.append(("image", "tests/eval_vision_ids.txt", "tests/ref_vision_top10.txt", [0, 3]))
ok = True
for be in backends:
    for name, ids_f, ref_f, splits in cases:
        if not os.path.exists(ref_f) or os.path.getsize(ref_f) == 0: continue
        if name == "image" and be != "vulkan": continue
        ref = [(int(a), float(b)) for a, b in (l.split() for l in open(ref_f) if l.strip())]
        ids = open(ids_f).read().strip()
        img = ["--image", "tests/vision_test.png"] if name == "image" else []
        for sp in splits:
            r = subprocess.run([exe, "--backend", be, "--eval", ids, "--eval-split", str(sp)] + img + extra,
                               capture_output=True, text=True)
            got = [(int(a), float(b)) for a, b in (l.split() for l in r.stdout.splitlines() if l.strip())]
            if len(got) != 10:
                print(f"{be:7s} {name:5s} split={sp}: FAILED\n{r.stderr[-2000:]}"); ok = False; continue
            refd = dict(ref)
            diff = max(abs(v - refd[i]) if i in refd else 99 for i, v in got[:5])
            top1 = got[0][0] == ref[0][0]
            # the image prompt carries ~0.02 rms numeric noise over the vocab (a rows/columns position swap is ~0.08)
            status = "OK " if top1 and diff < (0.15 if name == "image" else 0.05) else "BAD"
            ok &= status == "OK "
            print(f"{status} {be:7s} {name:5s} split={sp}: top1 {got[0][0]} (ref {ref[0][0]}), max|dlogit| top5 = {diff:.4f}")
print("ALL OK" if ok else "SOME FAILED")
