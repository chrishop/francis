from setuptools import setup, find_packages

setup(
    name="francis",
    version="0.0.1",
    packages=find_packages(),
    include_package_data=True,
    install_requires=["xeno-canto"],
    entry_points={"console_script": ["francis = francis.__main__:main"]},
)
