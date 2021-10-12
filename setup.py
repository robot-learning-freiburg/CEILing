"""Install script for setuptools."""

import setuptools
from os import path

# read the contents of your README file
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setuptools.setup(
    name="ceiling",
    version="0.0.1",
    install_requires=[
        "PyRep @ git+https://github.com/stepjam/PyRep.git@603d283f02faf4c9874ddbd931bf7bafb3ea50ad",
        "RLBench @ git+https://github.com/stepjam/RLBench.git@ce2e87b4b363da2678ba1ab7980248585a1eb67b"
    ],
    packages=setuptools.find_packages(),
    python_requires=">=3.7",
)
