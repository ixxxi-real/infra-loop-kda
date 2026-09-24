# Runtime layout

The tracked tree contains current kernel deliverables and the evidence needed to review them. Runtime state is deliberately lower priority:

- .infra/ stores migrated history, logs, runs, sessions and caches that are not part of the public package.
- projects/<project>/<stage>/runtime/ is an ignored compatibility area used by the control-plane adapters when they need task-local state.
- projects/<project>/<stage>/archive/ stores concise historical reports; raw profiler files and transcripts stay external.

Do not commit a raw benchmark log, NCU database, agent transcript, checkpoint,
GPU identifier or private host path. Promote only a manifest, checksum and
short interpretation into evidence/manifests/ or the stage archive.
