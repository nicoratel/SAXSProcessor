from setuptools import setup, find_packages

setup(
    name="saxsprocessor",
    packages=find_packages("src"),
    package_dir={"": "src"},
)