import os

import setuptools

# Change directory to allow installation from anywhere
script_folder = os.path.dirname(os.path.realpath(__file__))
os.chdir(script_folder)

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

# Installs
setuptools.setup(
    name="clover",
    version="1.1.0",
    author="Clover Contributors",
    author_email="opensource@example.com",
    description="Clover: inference-only NAVSIM-v1 planning release",
    url="https://github.com/your-org/clover",
    python_requires=">=3.8",
    packages=setuptools.find_packages(script_folder),
    package_dir={"": "."},
    classifiers=[
        "Programming Language :: Python :: 3.9",
        "Operating System :: OS Independent",
        "License :: OSI Approved :: Apache Software License",
    ],
    license="apache-2.0",
    install_requires=requirements,
)
