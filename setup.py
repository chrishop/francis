from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="francis",
    version="0.0.1",
    packages=find_packages(),
    include_package_data=True,
    install_requires=requirements,
    entry_points={"console_scripts": ["francis = francis.__main__:main"]},
)
