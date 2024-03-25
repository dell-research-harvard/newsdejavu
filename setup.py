from setuptools import setup, find_packages
import pathlib

# The directory containing this file
here = pathlib.Path(__file__).parent.resolve()

# The text of the README file
long_description = (here / "README.md").read_text(encoding="utf-8")

# This call to setup() does all the work
setup(
    name="newsdejavu",  # Required: the name of the package
    version="0.0.3",  # Required: the version of the package
    package_dir={"": "src"},  # Optional: tells setuptools that packages are under src
    packages=find_packages(where="src", exclude=["test*"]),  # Required: the packages to include
    python_requires=">=3.9, <4",  # Required: the Python version compatibility
    install_requires=[  # Optional: a list of dependencies needed to use the package
        "numpy",
        "pandas",
        "nltk",
        "opencv-python",
        "datasets",
        "torch",
        "requests",
        "Pillow>=10.0.1,<=15.0",
        "transformers",
        "sentence_transformers",
    ],
    long_description=long_description,  # Optional: the long description of the package
    long_description_content_type="text/markdown",  # Optional: the format of the long description
)
