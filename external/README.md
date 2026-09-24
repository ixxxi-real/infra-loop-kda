# External dependencies

The repository keeps large or independently versioned projects as Git
submodules. They are never copied into the first-party source tree.

| Path | Upstream | Why it is here | Update policy |
| --- | --- | --- | --- |
| `external/sglang` | [Infrawaves/sglang](https://github.com/Infrawaves/sglang) | Kimi K3 runtime source and patch target | Pin the commit recorded by each task |
| `external/kda` | [NVlabs/kda](https://github.com/NVlabs/kda) | Agent workflow plus recursive KernelWiki/NCU skills | Pin and review explicitly |
| `external/humanize` | [PolyArch/humanize](https://github.com/PolyArch/humanize) | Humanize/RLCR planning and Codex review integration | Pin and review explicitly |

Initialize everything from a fresh clone:

```bash
git submodule sync --recursive
git submodule update --init --recursive
```

The root `.gitmodules` uses HTTPS URLs so a public clone does not require a
preconfigured SSH key. The nested KDA modules are also pinned by KDA's own
`.gitmodules` file.

Before running a task, verify the exact SGLang commit in
`config/project.local.json` and the task's `task.json`. A branch name alone is
not reproducible. Do not commit build output, generated extensions, or local
changes inside a submodule.
