from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf8") as fh:
    long_description = fh.read()

setup(
    install_requires=[
        "pydantic>=2.7.4",
        "jsonschema==4.17.1",
        "setuptools~=65.5.1",
    ],
    name="marqo-commons",
    version="1.0.1",
    author="Marqo",
    author_email="support@marqo.ai",
    description="Commons for Marqo projects",
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
