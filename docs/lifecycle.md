# Task lifecycle

Use one canonical state in `projects/<project>/<stage>/task.json`. The optimization state and serving integration state are independent.

| State | Exit condition | Loop may start? |
| --- | --- | --- |
| `scaffolded` | Task directory and contract exist; inputs are still unresolved. | No |
| `prepared` | Source commit, model facts, workload and acceptance thresholds are frozen. | **Yes** |
| `candidate_running` | A run manifest exists and points to the exact command and environment. | **Yes** (resume) |
| `candidate_verified` | Correctness and timing evidence meet the contract. | No — awaiting review |
| `accepted` | The patch and evidence are reviewable and reproducible. | No — immutable |
| `delivered` | The accepted artifact is copied or applied to the target source tree. | No — immutable |
| `closed` | Delivery and any requested serving validation are recorded. | No — immutable |
| `rejected` | The candidate was rejected, with the reason recorded. | No — immutable |
| `paused` / `blocked` | Progress is stopped with a reason, owner and next action. | No — resolve first |

`ready` is accepted as a synonym for `prepared`, and `in_progress` for
`candidate_running`.

Integration has its own state: `not_applied`, `applied`, `serving_verified`. An accepted kernel may remain `not_applied` until a separate integration run is completed.

Every transition should update the task state, add or update a report, and link the evidence manifest. Do not infer acceptance from a directory name or an old log.

## The start gate is enforced, not advisory

`k3ctl agent start` refuses to spawn anything unless the lifecycle permits it.
`k3ctl agent plan` reports the same decision without spawning, so you can see
exactly why a task is held back.

Two conditions are checked beyond the state itself:

- **`workload_status` must be exactly `resolved`.** This is an allow-list: a
  `placeholder`, `unresolved`, or misspelled value blocks the start, so a fresh
  scaffold can never drive a loop by accident.
- **`optimization_status` must not be `accepted`.** An accepted result is frozen
  and is never reopened implicitly.

An unrecognised status is refused rather than guessed at, and the error lists the
states that are known. A typo cannot be treated as startable.

## Reopening finished work

A terminal task is immutable here. To work with one:

| Intent | Command |
| --- | --- |
| Reproduce or inspect its exact base | `k3ctl workspace prepare --task <id>` |
| Do further optimization | `k3ctl task-create --id <new-id> …` |

`workspace prepare` materialises the task's exact base commit without starting a
loop, so reproduction never risks mutating an accepted result. There is no flag
that reopens a terminal task in place: the new work gets its own task, its own
contract, and its own evidence.
