# Adding a task

Each optimization is a self-contained package. Use `task-create`; it derives the
package from the Kernel Design Agents basic-flow contract and produces a fresh,
deliberately unstartable scaffold.

> **Do not copy `projects/kimi-k3/prefill`.** That package is *accepted*: it carries
> `status: accepted`, `optimization_status: accepted`, `patch_file`,
> `patch_sha256` and an `evidence_manifest` pointing at recorded acceptance
> evidence. Copying it would make a brand-new task claim an acceptance it has
> not earned, against a base commit it was never measured on. `task-create`
> refuses to scaffold any of those fields.

## Create the package

```bash
k3ctl task-create \
  --id decode-kda \
  --kind decode \
  --base-commit 0123456789abcdef0123456789abcdef01234567 \
  --objective "Reduce decode-phase attention latency without changing gate semantics." \
  --validation-command "python3 bench/correctness.py --source-root <root> --workloads <workloads> --report <report>" \
  --performance-command "python3 bench/benchmark.py --baseline-root <baseline> --candidate-root <candidate> --workloads <workloads> --out <out>"
```

`--base-commit` must be a complete 40-character SHA. Everything is validated
*before* anything is written, so a rejected invocation leaves no partial task
behind. Add `--dry-run` to see the file list without writing, and `--register`
to append the task to a **local** configuration — registering into the tracked
example config is refused.

What you get:

| File | Purpose |
| --- | --- |
| `task.json` | Canonical task state. Nothing else overrides it. |
| `contract.md` | Objective, correctness requirements, commands, promotion rules. |
| `source-trace.md` | The exact base commit and how to verify it. |
| `plan-input.md` | Human-authored input to planning. Not the executable plan. |
| `prompt.md` | Implementation prompt derived from the KDA basic flow. |
| `workloads.placeholder.json` | Replace with a resolved workload. |
| `model-profile.json` | Model facts this task depends on. |

The KDA prompt is *referenced and hashed*, not copied: `task.json` records
`workflow.kda_prompt`, its SHA-256 and the KDA commit, so attribution stays
accurate and the upstream licence is respected.

## Why a fresh scaffold cannot start

By design. `k3ctl agent start` refuses it, and `agent plan` lists why:

- `status` is `scaffolded`, not `ready`;
- `workload_status` is `placeholder`, and only an explicit `resolved` passes;
- no workspace has been prepared.

## Making it startable

```bash
# 1. Resolve the workload, then set workload_status to "resolved" in task.json.
# 2. Write plan-input.md into a real plan: goal, acceptance criteria, steps.
# 3. Materialise the exact base commit into an isolated workspace.
k3ctl workspace prepare --task decode-kda

# 4. Set "status" to "ready" in task.json.
# 5. Confirm the loop would start, without spawning anything.
k3ctl agent plan --task decode-kda
```

`workspace prepare` fails if the exact base commit is not reachable from the
configured local source repository. It never substitutes the public submodule
pin — that is a different source base. Point `--source-repo` at a clone that
contains the commit, or publish the ref.

## A decode task

Decode work differs from prefill mainly in its workload and gates, not in its
structure:

- **Workload:** decode-phase shapes — single-token steps, KV-cache growth,
  continuation state, ragged batches across steps.
- **Correctness:** state carried across steps must stay bit-consistent with the
  baseline where the contract requires it. Define the case count explicitly; a
  gate that executes zero cases is never a pass.
- **Timing:** per-step latency at a fixed cache occupancy, not whole-sequence
  throughput. Record occupancy alongside every measurement, or the numbers are
  not comparable.
- **Allowed files:** list them in `task.json`. Keep the change scoped.

```bash
k3ctl task-create --id decode-kda --kind decode --base-commit <sha> \
  --objective "…" --validation-command "…" --performance-command "…"
```

## An integration task

Integration validates that an accepted kernel works in a serving deployment. It
is a separate task because acceptance and serving validation are separate
states — a kernel can be accepted while `integration_status` is `not_applied`.

```bash
k3ctl task-create --id integration-kda --kind integration --base-commit <sha> \
  --objective "Validate the accepted prefill kernel in the real serving path." \
  --validation-command "…"
```

An integration task:

- references the accepted patch by path and **exact SHA-256** rather than
  copying acceptance state into itself;
- validates the real caller and serving path, not the kernel in isolation;
- records `serving_validation` separately from `optimization_status`;
- applies the patch in its own prepared workspace, never in `external/sglang`.

Read [`docs/source-compatibility.md`](source-compatibility.md) first: an
accepted patch is not automatically portable to a different source base.

## During implementation

```bash
make init                                     # pinned submodules
make skills-check                             # report only; writes nothing
make skills-install SKILLS_ARGS="--apply"     # link the skills
```

Use KDA's KernelWiki for prior-art lookup, the NCU skill for profile diagnosis,
and Humanize for plan/review iteration — see
[`docs/agent-loop.md`](agent-loop.md). Keep tool outputs in the task's external
evidence location; only the contract, manifest, interpretation and checksums
belong in Git.

## Registering and validating

If you did not pass `--register`, add the entry to your local config yourself:

```json
{ "id": "decode-kda", "kind": "decode", "path": "projects/kimi-k3/decode/decode-kda" }
```

Then:

```bash
make validate CONFIG=config/project.local.json
make task-list CONFIG=config/project.local.json
```

Do not create a second top-level project per task. The project config registers
tasks; the task package owns its state and reports; the evidence store owns
large run outputs.
