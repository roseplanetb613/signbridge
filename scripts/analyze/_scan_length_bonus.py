"""临时扫描：长度偏置 bonus 扫描（骨架模型 dev 集）。"""
import json
import subprocess
import sys

out = []
for b in (0.0, 0.1, 0.2, 0.5, 1.0, 2.0, 3.0):
    r = subprocess.run(
        [sys.executable, "scripts/analyze/wer_buckets.py",
         "--checkpoint", "checkpoints/best.pt", "--splits", "dev",
         "--beam-width", "5", "--device", "cpu",
         "--length-bonus", str(b), "--show-examples", "0"],
        capture_output=True, text=True, encoding="utf-8", errors="replace")
    j = json.load(open("reports/wer_buckets/results.json", encoding="utf-8"))
    res = j["dev"]
    del_j = sum(1 for s in res["per_sample"] for op in s["ops"]
                if op.startswith("del:"))
    hyp_len = sum(len(s["hyp"]) for s in res["per_sample"]) / len(res["per_sample"])
    print(f"bonus={b}: WER {res['overall_wer']:.4f} seg_acc {res['seg_acc']:.4f} "
          f"del {del_j} hyp_avg {hyp_len:.2f}", flush=True)
