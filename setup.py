from setuptools import setup

requirements = [
    "numpy>=1.16",
    "scipy>=1.3",
    "torch>=2.1.1",
    "gpytorch>=1.11",
    "botorch<0.10",
    "wandb>=0.16",
    "matplotlib>=3.7",
    "tqdm>=4.0",
    "notebook>=6.0",
    "ipywidgets>=8.1.1",
    "scikit-learn>=1.1",
    "pandas>=2.2",
    "openml>=0.14.2"
]

setup(
    name="pandora_automl",
    version="1.0",
    description="Cost-aware Stopping for Bayesian Optimization",
    author="Qian Xie and Linda Cai",
    python_requires='>=3.9',
    packages=["pandora_automl"],
    install_requires=requirements
)
