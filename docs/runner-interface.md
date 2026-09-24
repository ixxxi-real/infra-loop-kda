# Runner interface

The public repository intentionally does not encode one team's gateway or GPU
launcher. `k3ctl run` freezes a complete, reproducible invocation and then
delegates it to a configured adapter; the adapter owns the environment.

## The two adapters

| Adapter | Purpose |
| --- | --- |
| `local-command` | Runs the frozen argv on this machine under a status supervisor. Intended for offline work, mock testing, and as the reference a site wrapper can copy. |
| `site-wrapper` | Forwards every operation to a configured argv. Owns SSH, container routing, GPU scheduling and GPU/UUID resource locking. |

Nothing is passed through a shell. `runner.commands.*` and `runner.wrapper` must
be **argv lists**, not strings: a string would require shell parsing, and a
workload value could then become a command.

```json
{
  "runner": {
    "adapter": "site-wrapper",
    "wrapper": ["ssh", "gpu-host", "/opt/site/k3-run"],
    "environment": {},
    "commands": {}
  }
}
```

With `adapter: "site-wrapper"` and no `wrapper`, every operation fails with an
explicit statement of what is missing. The CLI never simulates a remote run.

## A local-command configuration, in full

`local-command` means "run the frozen argv on this machine". It does **not** mean
the work is CPU-only: the configuration below invokes the real measurement
harness, and `bench/_worker.py` imports torch, sets
`torch.backends.cuda.matmul.allow_tf32` and runs under `torch.inference_mode()`.
**These commands require a GPU.** `local-command` simply means the GPU is this
machine rather than a remote one.

For the genuinely CPU-only operation, see [Running without a GPU](#running-without-a-gpu)
below.

Commands are argv lists; the tokens in braces are substituted by the runner:

```json
{
  "runner": {
    "adapter": "local-command",
    "environment": { "PYTHONUNBUFFERED": "1" },
    "commands": {
      "python": "python3",
      "correctness": [
        "{python}", "{task_dir}/bench/correctness.py",
        "--source-root", "{candidate_root}",
        "--workloads", "{task_dir}/workloads.resolved.json",
        "--report", "{output_dir}/correctness.json"
      ],
      "benchmark": [
        "{python}", "{task_dir}/bench/benchmark.py",
        "--baseline-root", "{baseline_root}",
        "--candidate-root", "{candidate_root}",
        "--workloads", "{task_dir}/workloads.resolved.json",
        "--out", "{output_dir}/benchmark",
        "--trials", "5", "--warmup", "5", "--samples", "30"
      ]
    }
  }
}
```

| Token | Substituted with |
| --- | --- |
| `{baseline_root}` | The immutable baseline tree |
| `{candidate_root}` | The writable candidate tree (also `{source_root}`) |
| `{output_dir}` | `…/runs/<run-id>/artifacts` |
| `{scratch_dir}` | `…/runs/<run-id>/scratch` |
| `{task_dir}` | The task package directory |
| `{python}` | `runner.commands.python`, default `python3` |

Omitting `commands` entirely is fine: for a `prefill` task the runner derives
these same invocations from the harness already in the task package. Define one
only to override it. An unresolved `{token}` blocks the plan rather than being
passed through literally.

```bash
k3ctl run plan  --task <id> --operation correctness   # freezes argv, starts nothing
k3ctl run start --task <id> --operation correctness   # requires a GPU
k3ctl run status --task <id>
```

## Running without a GPU

`run plan` is always safe: it freezes the argv, records identity, and starts
nothing. Use it to inspect exactly what would run on a GPU host.

The one operation that genuinely executes without a GPU is `preflight-static`.
`bench/preflight.py` imports only the standard library, and `--static-check`
skips the worker subprocess entirely while still validating the workload
document and allowing unresolved checkpoint metadata:

```bash
k3ctl run plan  --task <id> --operation preflight-static
k3ctl run start --task <id> --operation preflight-static
k3ctl run status --task <id>
```

Note that `preflight-static` is a named operation, not a pipeline stage, so its
plan reports `in_pipeline: false`.

### What the test suite actually exercises

The adapter, supervisor, identity recording, cancellation and artifact checking
are tested against **fixture** harnesses written for that purpose — not against
`bench/correctness.py` or `bench/benchmark.py`. Artifact completeness is
additionally tested against the real `bench/benchmark.py` *output schema*, with
its worker subprocess mocked.

So the machinery is covered; the GPU commands themselves are not. No GPU
measurement has been performed.

## One stage per run, not a campaign

Each `run start` executes exactly **one named stage**. There is no command that
runs a whole acceptance campaign, and that is deliberate: a single "run
everything" verb would invite reading one green result as acceptance.

The plan records the pipeline and this run's position in it, for orientation
only:

```json
{
  "operation": "benchmark",
  "pipeline": {
    "pipeline": ["preflight", "correctness", "precision", "benchmark"],
    "position": 4,
    "total": 4
  }
}
```

Named stages for a `prefill` task: `preflight`, `preflight-static`,
`correctness`, `correctness-only`, `precision`, `benchmark`, `profile`.

Sequencing across stages is the operator's job, and gating is the harness's.

To be precise about what is and is not checked: there is **no cross-run
stage-evidence prerequisite check**. Finding an earlier `correctness.json`
somewhere says nothing about whether that run passed, whether it targeted this
candidate tree, or whether it shares this snapshot, so its presence is never
treated as a satisfied precondition.

Other preconditions *are* enforced, and they block the plan:

| Enforced precondition | Effect |
| --- | --- |
| A prepared workspace exists, with distinct baseline and candidate roots | Blocks |
| The baseline tree still matches the digest recorded at preparation | Blocks |
| The workspace record's task id and base commit match this task | Blocks |
| The command has no unresolved `{token}` | Blocks |
| A `site-wrapper` adapter has a configured `wrapper` | Blocks |

What is *not* enforced is stage ordering.
`benchmark.py` re-runs the correctness gate itself unless handed a verified
`--correctness-report` from the same frozen snapshot — that verified-reuse path
is the only sound way to skip it, and it is opted into explicitly through
`runner.commands`.

A `stage_history` field lists earlier artifacts purely as orientation, and says
in its own payload that it is not a prerequisite check.

## What the adapter must implement

The wrapper is invoked with a subcommand and `--run-id`:

| Invocation | Must do |
| --- | --- |
| `<wrapper> start --run-id <id> --cwd <dir> -- <argv…>` | Start the job. Print a stable site job id on the first stdout line. |
| `<wrapper> status --run-id <site_run_id>` | Print exactly one of `running`, `exited`, `signalled`, `cancelled`, `unknown`. |
| `<wrapper> cancel --run-id <site_run_id>` | Cancel that job and its workers. |
| `<wrapper> fetch --run-id <site_run_id> --destination <dir>` | Place the run's artifacts in `<dir>`. |

The id passed to `status`, `cancel` and `fetch` is the **`site_run_id`** the
wrapper printed on its first `start` stdout line, recorded in the run manifest.
Only `start` receives the local `--run-id`. If the wrapper prints nothing, the
local run id is reused as a fallback — so a wrapper that tracks jobs under its
own scheduler id must print that id, or later operations will address the wrong
job.

A `status` value outside that set is reported as `unknown` rather than guessed
at. A non-zero wrapper exit is an error, never an empty result.

### Resource locking belongs to the adapter

The generic CLI does not claim, hold or release any GPU or UUID lock, and says
so in every run plan. A configured GPU adapter is required to do it. Two
concurrent runs pointed at one device are the adapter's problem to prevent.

### Disconnect and resume

Pass a stable `--run-id` to `run start` to keep one identity across a
disconnect:

```bash
# Preview first if you want to: with no --run-id this reserves a throwaway id
# and starts nothing.
k3ctl run plan  --task <id> --operation benchmark

# Start under a stable id. `start` freezes its own plan internally.
k3ctl run start --task <id> --operation benchmark --run-id nightly-01

# Reconnect later, as often as you like. This never spawns anything.
k3ctl run status --task <id> --run-id nightly-01
```

Do **not** pass the same `--run-id` to `run plan` and then to `run start`:
`start` freezes its own plan under that id, and a run directory is never
reused, so the second command would fail. Reserve the id with `start`.

Reusing an existing id always fails, deliberately: a second run under one id
would inherit the first run's recorded identity and artifacts. Because `status`
spawns nothing, a reconnect cannot create a duplicate job.

A run is managed by the adapter it *started with*, recorded in its manifest. If
the configured adapter changes afterwards, `status`/`cancel`/`fetch` still use
the original and report the drift, so a local adapter never inspects a pid that
belongs to a remote job.

## What a run plan freezes

`run plan` starts nothing and writes `run.json` containing:

- the complete argv, cwd and environment, with `shell: false`;
- the **baseline** and **candidate** roots, and proof they are distinct;
- `identity.run_sha256` over argv, operation, base commit, workload hash,
  toolchain hash and `inputs_sha256`;
- `inputs_sha256` over the candidate tree, baseline tree, resolved workload and
  harness sources — so a measurement taken before a candidate edit and one taken
  after can never share an identity;
- `baseline_integrity`: the workspace record's task id, its base commit versus
  the task's current one, the prepared paths, and the baseline tree digest;
- the declared contract gates, as metadata;
- every blocker, if it is not ready.

### Baseline and candidate must differ

A/B harnesses take `--baseline-root` and `--candidate-root`. Pointing both at
one tree always reports zero difference, so identical roots are refused. The
baseline is materialised by `workspace prepare` and must stay byte-identical to
what was recorded; a modified baseline blocks the plan, because the run would
otherwise claim the task's base commit as provenance for a tree that is no
longer it.

## Execution state is not evidence, and evidence is not a gate result

`run status` reports three separate things and never collapses them:

```json
{
  "execution": { "state": "exited", "exit_code": 0, "ran_to_completion": true },
  "evidence":  { "complete": true },
  "gates_evaluated": false,
  "accepted": false
}
```

- **`execution`** — did the command run, and how did it end. A detached run is
  supervised so the real exit status survives: a finished run is distinguishable
  from a killed one, and a cancelled run is never a completion. If the supervisor
  dies without recording a terminal status, the state is `unknown` — not success.
- **`evidence`** — are the declared artifacts present, non-empty, and internally
  consistent. For a report directory this means the harness's own
  `summary.json` records `status: passed`, a non-zero case count, and every
  sidecar the schema requires. `correctness.json` is required for the benchmark
  schema, and the per-trial worker reports are derived from the summary's own
  `execution_order`, so a directory containing only `summary.json` is not
  complete. Hashes recorded in `reports_sha256` are verified, not just existence.
  Absolute paths such as `reused_correctness_report` are treated as provenance
  and are never existence-checked, so a fetched bundle is not falsely incomplete.
- **`gates_evaluated` / `accepted`** — always `false` here. The task harness
  evaluates the contract gates against the evidence, and a human reviews the
  result.

`exit_code == 0` means the command ran to completion. Nothing more.

## Cancellation reaches the whole tree

The measurement harnesses spawn their own workers (`benchmark.py` and
`precision/diagnose.py` both run `_worker.py` subprocesses). The supervisor
starts the child in its own process group and signals the **group**, so a cancel
does not leave workers running and still holding a GPU. It escalates to
`SIGKILL` only if the tree ignores `SIGTERM`, and records whether descendants
were actually cleared.

## Evidence manifests

For every run, the adapter or the operator should produce a manifest based on
`evidence/manifests/run.example.json` containing:

- task ID and exact source commit;
- redacted command and environment/image identity;
- workload and measurement parameters;
- URIs, sizes and SHA-256 checksums for raw artifacts;
- collection time and retention owner.

The runner may update an external task database, but the reviewed result must be
copied back into the task report and manifest so a GitHub reviewer can audit it
without access to the cluster.

## Scope of testing

The `local-command` adapter is exercised end to end, including cancellation of a
real process tree with a grandchild worker, and artifact completeness against the
genuine `bench/benchmark.py` output schema with its worker subprocess mocked.

The `site-wrapper` adapter has **no configured implementation here**. Its
delegation, status-value validation and refusal paths are tested against a stub;
no SSH, container or GPU path has been exercised. Treat a site adapter as
unvalidated until you have run it in your own environment.
