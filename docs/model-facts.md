# Kimi K3 model facts

This file is the public, stable description of the model inputs needed to design a task. It must contain facts that another engineer can review without revealing a private checkpoint.

Record:

- model identifier and configuration schema version;
- attention, head, KV-head, hidden-size and sequence-length assumptions used by the kernel;
- dtype, quantization and mask/layout assumptions;
- the source of each fact and the date it was checked;
- a redacted checkpoint hash or a private-artifact reference when publication is not allowed.

Do not put local filesystem paths, gateway addresses, node names, credentials or unreviewed measurements here. Those belong in the ignored local configuration or in an external evidence store.

The migrated Kimi K3 profile resolves to 96 heads, 12 local heads at attention TP8, K/V head dimensions 128/128, BF16 activation and state, gate lower bound `-5`, full-rank gate, short-convolution size 4 and 64-token kernel chunks. The checkpoint metadata hash is `9710e121a58d03ac92c8d6da287a19541994319afbbe6d6202af001ffd379213`.

These facts are recorded in the sanitized task input [`projects/kimi-k3/prefill/model-profile.json`](../projects/kimi-k3/prefill/model-profile.json). The live serving backend and traffic distribution remain unresolved by this task.
