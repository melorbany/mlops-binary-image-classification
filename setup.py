from setuptools import find_packages, setup

setup(
    name="cats-vs-dogs",
    version="0.1.0",
    description="Binary image classification MLOps pipeline",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
)
