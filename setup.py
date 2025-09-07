from setuptools import setup, find_packages

setup(
    name="agency_evaluations",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "argh",
        "pandas",
        "numpy",
        "scikit-learn",
        "plotly",
        "pyyaml",
    ],
    entry_points={
        "console_scripts": [
            "agency_pipeline=pipeline:pipeline",
        ],
    },
    author="Benjamin Sturgeon",
    author_email="bwm.sturgeon@gmail.com",
    description="A pipeline for agency evaluations",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/bensturgeon/agency_evaluations",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
)