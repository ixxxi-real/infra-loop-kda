"""Profile one exact chunk_kda case with NVTX warmup/capture markers."""

import argparse
import json

from common import load_kernel, load_workloads, sha256, source_info
from runtime import Inputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--workloads", required=True)
    parser.add_argument("--id", required=True)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=5)
    args = parser.parse_args()
    document = load_workloads(args.workloads)
    matches = [case for case in document["cases"] if case["id"] == args.id]
    if len(matches) != 1 or matches[0].get("continuation_splits"):
        parser.error("id must select one non-continuation case")
    if args.iters < 1 or args.warmup < 1:
        parser.error("iters and warmup must be positive")
    provenance = source_info(args.source_root)
    torch, kernel, origins = load_kernel(args.source_root)
    with torch.inference_mode():
        inputs = Inputs(torch, document["model_profile"], matches[0])
        with torch.cuda.nvtx.range("kda_warmup"):
            for _ in range(args.warmup):
                inputs.reset()
                result = inputs.invoke(kernel)
                torch.cuda.synchronize()
                del result
        for _ in range(args.iters):
            inputs.reset()
            torch.cuda.synchronize()
            # Each range encloses one call; no mutable-state replay without reset.
            with torch.cuda.nvtx.range("kda_capture"):
                result = inputs.invoke(kernel)
            torch.cuda.synchronize()
            del result
    print(json.dumps({"source": provenance, "runtime": origins, "workloads_sha256": sha256(args.workloads),
                      "id": args.id, "iters": args.iters, "inputs": inputs.describe()}, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
