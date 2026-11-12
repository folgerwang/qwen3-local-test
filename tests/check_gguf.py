# Checks the qwen4exp (GGUF) backends.
#   python tests/check_gguf.py tiny [backends]     tiny random model (make_tiny_gguf.py) against HF transformers
#   python tests/check_gguf.py MODEL [backends]    a real GGUF (dir or file): Vulkan against the CPU backend
# On the real model the backends agree to ~1e-4 until a router near-tie (10th vs 11th expert within ~1e-5) is
# resolved differently by their summation orders; from there the logits drift by ~0.1, hence the loose bound.
# backends: comma separated, default cpu,vulkan. The tiny model exercises every codec, QSA selection (from 17
# tokens on), the PLE window reset at EOS, prefill + decode splits and the tiled GEMM path.
import os, random, subprocess, sys

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
exe = os.path.join(root, "build", "Release", "qwen.exe")
if not os.path.exists(exe): exe = os.path.join(root, "build", "qwen")
target = sys.argv[1] if len(sys.argv) > 1 else "tiny"
backends = sys.argv[2].split(",") if len(sys.argv) > 2 else ["cpu", "vulkan"]

def top10(out):
    return [(int(a), float(b)) for a, b in (l.split() for l in out.splitlines() if len(l.split()) == 2 and l[0].isdigit())]

def run(model, be, ids, split, ctx):
    r = subprocess.run([exe, "--model", model, "--backend", be, "--eval", ids, "--eval-split", str(split), "--ctx", str(ctx),
                        "--no-stats"], capture_output=True, text=True)
    got = top10(r.stdout)
    if len(got) != 10: print(f"{be} split={split}: FAILED\n{r.stderr[-2000:]}")
    return got

def compare(name, got, ref, tol):
    if len(got) != 10: return False
    refd = dict(ref)
    diff = max(abs(v - refd[i]) if i in refd else 99 for i, v in got[:5])
    ok = got[0][0] == ref[0][0] and diff < tol
    print(f"{'OK ' if ok else 'BAD'} {name}: top1 {got[0][0]} (ref {ref[0][0]}), max|dlogit| top5 = {diff:.4f}")
    return ok

ok = True
if target == "tiny":
    model = os.path.join(root, "build", "tiny-qwen4exp.gguf")
    subprocess.check_call([sys.executable, os.path.join(root, "tests", "make_tiny_gguf.py"), model], stdout=subprocess.DEVNULL)
    rnd = random.Random(7)
    for n in (18, 90, 300):
        ids = [rnd.randrange(0, 512) for _ in range(n)]
        if n > 60: ids[20] = ids[21] = ids[55] = 256  # EOS inside the PLE n-gram window
        ids = ",".join(map(str, ids))
        r = subprocess.run([sys.executable, os.path.join(root, "tests", "ref_hf_gguf.py"), model, ids], capture_output=True, text=True)
        ref = top10(r.stdout)
        if len(ref) != 10: sys.exit("reference failed:\n" + r.stderr[-3000:])
        for be in backends:
            for sp in (0, 3, n // 2):
                ok &= compare(f"{be:7s} tiny n={n:3d} split={sp:3d}", run(model, be, ids, sp, 512), ref, 0.002)
else:
    cases = [("short", "tests/eval_ids.txt", [0, 5]), ("long", "tests/eval_long_ids.txt", [0, 3])]
    for name, ids_f, splits in cases:
        ids = open(os.path.join(root, ids_f)).read().strip()
        ref = run(target, "cpu", ids, 0, 2048)
        if len(ref) != 10: sys.exit("cpu reference failed")
        print(f"cpu {name}: top1 {ref[0][0]} {ref[0][1]:.4f}")
        for be in backends:
            for sp in splits:
                if be == "cpu" and sp == 0: continue
                ok &= compare(f"{be:7s} {name:5s} split={sp}", run(target, be, ids, sp, 2048), ref, 0.5)
print("ALL OK" if ok else "SOME FAILED")
