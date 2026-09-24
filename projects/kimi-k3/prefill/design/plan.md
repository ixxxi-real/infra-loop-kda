# Kimi K3 prefill KDA plan

Completed optimization work:

- [x] Freeze model facts and the 62-case resolved workload.
- [x] Freeze the source-only patch and checksum.
- [x] Run the correctness gate, advisory precision diagnostic and paired timing measurements.
- [x] Record the public evidence summary and scope limits.

Remaining integration work:

- [ ] Make source commit `8eea3c25a3eaa3c850dbdb0ede2bba7de8f03e93` reachable for a fresh clone.
- [ ] Apply the patch in an isolated integration worktree and run `git apply --check`.
- [ ] Verify imports, the real Kimi K3 caller, backend dispatch and TP8 serving behavior.
- [ ] Run a separate serving-level acceptance and update `integration_status`.
