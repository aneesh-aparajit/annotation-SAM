# setup.py
from pathlib import Path
from setuptools import find_namespace_packages, setup


BASE_DIR = Path(__file__).parent
with open(Path(BASE_DIR, "requirements.txt"), "r") as file:
    required_packages = [ln.strip() for ln in file.readlines()]

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="annotation-SAM",
    version="1.0.0",
    description="A package to annotate images effeciently with SAM",
    author="Aneesh Aparajit G",
    author_email="aneeshaparajit.g2002@gmail.com",
    url="https://github.com/aneesh-aparajit/annotation-SAM",
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=[required_packages],
    packages=find_namespace_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
    ],
    include_package_data=True,
)
