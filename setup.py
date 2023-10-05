from os import path

from setuptools import find_packages, setup

here = path.abspath(path.dirname(__file__))

with open(path.join(here, "README.md"), encoding="utf-8") as readme_file:
    readme = readme_file.read()

with open(path.join(here, "requirements.txt")) as requirements_file:
    # Parse requirements.txt, ignoring any commented-out lines.
    requirements = [
        line
        for line in requirements_file.read().splitlines()
        if not line.startswith("#")
    ]

setup(
    name="saxswaxs_workflows",
    version="1.0",
    description="Data analysis workflows for SAXS/WAXS",
    long_description=readme,
    author="Collaboration between DESY and ALS",
    author_email="wkoepp@lbl.gov",
    url="https://github.com/als-computing/SAXSWAXS_workflows",
    packages=find_packages(),
    install_requires=requirements,
)
