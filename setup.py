from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf8") as fh:
    long_description = fh.read()

setup(
    install_requires=[
        "pydantic==1.9.1",
        "jsonschema==4.17.1",
        "setuptools~=65.5.1",
    ],
    name="marqo-commons",
    version="1.0.0",
    author="marqo org",
    author_email="org@marqo.io",
    description="Commons for marqo projects",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(where="src", exclude=("tests*",)),
    keywords="search python marqo opensearch neural tensor semantic vector embedding",
    platform="any",
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    include_package_data=True,
    python_requires=">=3",
    package_dir={"": "src"},
)