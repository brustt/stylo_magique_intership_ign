from setuptools import find_packages, setup

setup(
    packages=find_packages(where="src"),
    name="magic_cd",
    package_dir={"": "src"},
    version="0.1.0",
    description="change detection segmnentation with user prompt",
    author="martin dzr",
    license="Apache License 2.0",
    install_requires=["python-dotenv"],
)
