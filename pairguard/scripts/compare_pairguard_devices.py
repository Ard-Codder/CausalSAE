"""Compare two matching PairGuard bench JSONs; left/right order controls ratios."""
import argparse
import json
import statistics
from pathlib import Path


def compare(left, right, n):
    ignored = {"device", "output"}
    config = lambda x: {k: v for k, v in x["config"].items() if k not in ignored}
    if any(x["mode"] != "bench" for x in (left, right)):
        raise ValueError("Supply two bench-mode JSONs")
    if config(left) != config(right):
        # Paths to copies of the identical prompt dataset have no effect.
        a, b = config(left), config(right)
        a.pop("data")
        b.pop("data")
        if a != b:
            raise ValueError("Configs differ")
    for k in ("model", "toy", "model_commit", "prompt_hash", "transformers"):
        if left[k] != right[k]:
            raise ValueError(f"Mismatch: {k}")
    a, b = (x["timing"] for x in (left, right))
    for k in ("input_ids_sha256", "batch", "padded_prompt_width", "new_tokens_per_prompt"):
        if a[k] != b[k]:
            raise ValueError(f"Timing mismatch: {k}")
    med = lambda x, k: statistics.median(x[k])
    key = "prefill_plus_fixed_decode_s"
    ta, tb = med(a, key), med(b, key)
    batches = (n + a["batch"] - 1) // a["batch"]
    return {
        "left_device": left["device"], "right_device": right["device"], "toy": left["toy"],
        "left_torch": left["torch"], "right_torch": right["torch"],
        "left_prefill_s": med(a, "prefill_s"), "right_prefill_s": med(b, "prefill_s"),
        "left_prefill_decode_s": ta, "right_prefill_decode_s": tb,
        "runtime_ratio_left_div_right": ta / tb,
        "outputs": n, "left_extrapolated_minutes": ta * batches / 60,
        "right_extrapolated_minutes": tb * batches / 60,
        "warning": "Linear extrapolation for this fixed batch/length, unhooked, excluding loading.",
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("left", type=Path)
    p.add_argument("right", type=Path)
    p.add_argument("--outputs", type=int, default=1000)
    args = p.parse_args()
    if args.outputs <= 0:
        p.error("--outputs must be positive")
    print(json.dumps(compare(json.loads(args.left.read_text()), json.loads(args.right.read_text()),
                             args.outputs), indent=2))
