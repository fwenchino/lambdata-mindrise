import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="lambdata_mindrise",
    version="0.0.6",
    author="Andre Savkin",
    author_email="mindrise@live.com",
    description="A package of helper functions for data analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/andre-sav/lambdata-mindrise",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)