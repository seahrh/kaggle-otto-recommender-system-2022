from setuptools import find_packages, setup

__version__ = "1.0"
setup(
    name="otto2022",
    version=__version__,
    python_requires="~=3.7",
    install_requires=[
        "optuna==3.0.3",
        "pandas==1.3.5",
        "pyarrow==8.0.0",
        "scikit-learn==1.0.2",
        "sentencepiece==0.1.97",
        "pytorch-lightning==1.7.7",
        "transformers==4.20.1",
        "torch==1.13.1+cu117",
        "torchaudio==0.13.1+cu117",
        "torchvision==0.14.1+cu117",
        "tqdm==4.64.0",
    ],
    extras_require={
        "lint": [
            "black==22.3.0",
            "isort==5.10.1",
            "pre-commit==2.19.0",
            "flake8==4.0.1",
            "mypy==0.961",
        ],
        "tests": [
            "pytest==7.1.2",
            "pytest-cov==3.0.0",
        ],
        "notebook": ["jupyterlab==3.4.3", "ipywidgets==7.7.1", "seaborn==0.11.2"],
    },
    packages=find_packages("src", exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
    package_dir={"": "src"},
    include_package_data=True,
    description="OTTO Multi-Objective Recommender System (Kaggle Competition 2022)",
    license="MIT",
    author="seahrh",
    author_email="seahrh@gmail.com",
    url="https://github.com/seahrh/kaggle-otto-recommender-system-2022",
)
