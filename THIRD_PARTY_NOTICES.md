# Third-party notices

This repository is a control plane that links to independently maintained
projects. The submodules are separate works and keep their own licenses and
notices:

| Component | Pinned source | License reference |
| --- | --- | --- |
| SGLang | `external/sglang` at `9ac2710bd37622f38edb078cc753244a3c38c334` | [`external/sglang/LICENSE`](external/sglang/LICENSE) |
| Kernel Design Agents | `external/kda` at `ef6ce617693ef0782b3ecb9f37e39bbf10226a90` | [`external/kda/LICENSE`](external/kda/LICENSE) and [`external/kda/THIRD_PARTY_NOTICES.md`](external/kda/THIRD_PARTY_NOTICES.md) |
| Humanize | `external/humanize` at `4eeb392cd2450c3df99cb898bd6b24839967bbc3` | MIT, as stated by the upstream [`README.md`](external/humanize/README.md) |

The first-party control-plane files are distributed under the Apache License
2.0 in [`LICENSE`](LICENSE). A submodule's license governs that submodule's
contents; updating a gitlink may change its applicable notices and must be
reviewed as part of the update.
