# KDA prefill optimization knowledge

This directory is a small, evidence-backed memory for future optimization
agents. Each entry states when a transformation applies, what resource it
changes, and why it was accepted or rejected. A negative result is useful: it
prevents the next campaign from repeating a change that already failed the
full workload.

Sources are the experiment's `candidates.jsonl` and immutable evidence paths.
Do not promote an entry from a microbenchmark alone. Link a diagnosis/profile
and the full correctness, precision and paired benchmark evidence.

Files:

- `optimization-skeletons.jsonl`: reusable structural patterns and risks.
- `negative-evidence.jsonl`: rejected KDA prefill candidates and their limits.
