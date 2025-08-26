#!/usr/bin/env python
from pathlib import Path
from setuptools import find_packages, setup

# Load the long description from the README
this_dir = Path(__file__).resolve().parent
long_description = (this_dir / "README.md").read_text(encoding="utf-8")

setup(
    # ---- core metadata ----
    name="clipseg-debris",
    version="0.1.0",
    description="Debris segmentation using post-hurricane aerial imagery",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Kooshan Amini, Yuhao Liu, Jamie Padgett, Guha Balakrishnan, Ashok Veeraraghavan",
    author_email="jamie.padgett@rice.edu",
    url="https://github.com/Way-Yuhao/CLIPSeg-debris",
    license="MIT",
    python_requires=">=3.10",

    # ---- package discovery ----
    package_dir={"": "src"},
    packages=find_packages(where="src"),

    # ---- dependencies ----
    # install_requires=[
    #     "torch>=2.3",
    #     "lightning>=2.4",
    #     "hydra-core>=1.3,<1.4",
    #     "transformers>=4.41",
    #     "openai-clip",
    #     "numpy",
    #     "pandas",
    #     "scikit-learn",
    #     "matplotlib",
    #     "rich",
    # ],
    extras_require={
        "dev": ["pytest", "ipython", "pre-commit"]
    },

    # ---- CLI entry points ----
    entry_points={
        "console_scripts": [
            "train_clipseg = src.train:main",
            "eval_clipseg  = src.eval:main",
        ],
    },

    # ---- nice extras for PyPI ----
    project_urls={
        "Paper": "https://onlinelibrary.wiley.com/doi/10.1111/mice.70033",
        "Dataset": "https://designsafe.org/projects/CLIPSeg-debris",
        "Bug Tracker": "https://github.com/Way-Yuhao/CLIPSeg-debris/issues",
    },
)