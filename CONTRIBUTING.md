# Contributing

Open a change around one task or one toolchain update at a time. A task change
should include its contract, source trace, validation command, and evidence
manifest updates. A submodule update should identify the reviewed upstream
commit and explain any compatibility impact. Keep the repository usable from a
fresh clone: public examples must not depend on a developer's filesystem,
credentials or GPU allocation.

Before opening a pull request, run:

```bash
make validate
make test
make task-test
make workload-example
bash -n scripts/doctor.sh scripts/install-skills.sh
make verify
make package
```

Do not commit checkpoints, gateway details, credentials, GPU UUIDs, profiler
databases, generated build output, or local Humanize/KDA state. Keep raw
artifacts in the external evidence store and commit only redacted manifests,
checksums and conclusions.

The GitHub Actions CPU matrix is the minimum merge gate. It intentionally does
not claim GPU performance. A kernel result needs a pinned source commit,
frozen workload hash, correctness and precision reports, paired baseline/
candidate timings, and the hardware/runtime identity used to collect them.

For GPU results, state the hardware, CUDA/PyTorch/SGLang source commits,
workload identity, correctness gate, timing protocol, and known limits. Keep
optimization acceptance separate from serving integration.
