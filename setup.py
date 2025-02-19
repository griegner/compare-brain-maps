from pathlib import Path

from setuptools import find_packages, setup

requirements = Path("requirements.txt").read_text().splitlines()

setup(name="compare_brain_maps", packages=find_packages(), install_requires=requirements)
