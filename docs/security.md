# Publication and secret handling

This repository is designed to be public. Before pushing it to GitHub, check that no file contains:

- access tokens, passwords, private keys or signed URLs;
- private model paths or customer data;
- internal gateway/node addresses or GPU UUIDs;
- raw traces containing request data;
- private container registry coordinates.

Use `config/project.local.json` for machine-specific values and an external artifact store for raw evidence. If a secret is accidentally committed, remove it from Git history and rotate it; deleting the current file is not sufficient.
