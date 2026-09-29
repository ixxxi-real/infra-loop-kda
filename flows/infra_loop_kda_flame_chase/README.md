# Protected Infra Loop-KDA Flame Chase

This flow loads the pinned `external/flowverse` Flame Chase implementation and
adds one control-plane invariant: `.kda-task` is copied to temporary storage
before each cleaner epoch and restored after it. The cleaner can still remove
scratch output in the candidate tree, but candidate ledgers, benchmark and
failure records, and profiler originals are preserved byte-for-byte.

The adapter intentionally does not duplicate the upstream loop. Fresh coding
sessions, alternating chasers, budgets, resumability and history behavior come
from the pinned Flowverse checkout.
