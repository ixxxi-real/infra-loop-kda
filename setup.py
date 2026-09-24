"""Setuptools bridge for source distributions and wheel builds.

The project is configured in ``pyproject.toml``. This small bridge keeps source
archives usable with the setuptools versions bundled by common Python 3.x
virtual environments.
"""

from setuptools import setup


setup(
    name="infra-loop-kda",
    version="0.1.0",
    description="Reproducible control plane for automated GPU kernel optimization",
    # Both packages are required: tools.k3ctl is the entrypoint, and every
    # command implementation lives in tools.k3. Omitting tools.k3 would install
    # a console script that cannot import its own modules.
    packages=["tools", "tools.k3"],
    python_requires=">=3.9",
    entry_points={"console_scripts": ["k3ctl=tools.k3ctl:main"]},
)
