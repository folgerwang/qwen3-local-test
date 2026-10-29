# Compares qwen.exe tokenization against HF `tokenizers` (test oracle only).
import json, subprocess, sys, random
from tokenizers import Tokenizer

import os
model, exe = sys.argv[1], os.path.abspath(sys.argv[2])
ref = Tokenizer.from_file(model + "/tokenizer.json")
cases = [
    "Hello world", "Hello, world! How's it going? I'm fine, you'll see. THEY'RE HERE'S",
    "  leading spaces\n\n\ttabs\r\n  trailing   ", "numbers 1234567 3.14159 1e10 ٣٤٥ ⅷ",
    "中文测试：你好，世界！こんにちは、世界。안녕하세요 세계", "emoji 😀👍🏽 family 👨‍👩‍👧 flags 🇨🇳🇺🇸",
    "code: def f(x):\n    return x**2  # comment\n\nint main() { return 0; }",
    "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
    "Combining: e\u0301 a\u0308 \u212b \u1e0b\u0323 Hangul \u1100\u1161\u11a8 decomposed",
    "Arabic: مرحبا بالعالم. Hindi: नमस्ते दुनिया. Thai: สวัสดีชาวโลก",
    "Mixed\u00a0nbsp and\u3000ideographic space and \u2028 line sep",
    "symbols !!! ??? ... --- ___ *** ### $$$ @@@ ~~~ ``` '''",
    "'s 'S 'Re 'LL 'd x's", "a\n\n\nb \n \n c   \n", "trailing newline\n", " ", "", "\n\n",
    "URL https://example.com/path?q=1&b=2#frag e-mail: foo.bar@baz.org",
]
rnd = random.Random(0)
alphabet = "abcXYZ019 \n\t.,'!?中文é\u0301😀ـ-_<>|"
for _ in range(300):
    cases.append("".join(rnd.choice(alphabet) for _ in range(rnd.randint(1, 40))))
with open("tests/tok_cases.json", "w", encoding="utf-8") as f:
    json.dump(cases, f, ensure_ascii=False)
out = subprocess.run([exe, "--model", model, "--tokenize-file", "tests/tok_cases.json"],
                     capture_output=True, text=True, encoding="utf-8").stdout.splitlines()
bad = 0
for c, line in zip(cases, out):
    mine = [int(x) for x in line.split()]
    theirs = ref.encode(c, add_special_tokens=False).ids
    if mine != theirs:
        bad += 1
        if bad <= 10: print("MISMATCH", repr(c), "\n  mine  ", mine, "\n  theirs", theirs)
print(f"{len(cases) - bad}/{len(cases)} match" + ("" if len(out) == len(cases) else f" (got {len(out)} lines)"))
