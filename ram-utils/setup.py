from setuptools import setup
with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name='ram-utils',
    version='0.0.1',
    author='Sigdel Shree Ram',
    author_email='shreeramsigdel77@gmail.com',
    description='Utilities packages',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/shreeramsigdel77/ram-utils.git",
    packages=setup.find_packages(),
    classfifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',

    
)