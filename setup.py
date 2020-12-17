from setuptools import setup, find_packages

packages = find_packages(
    where='.',
    include=['ram_utils*']
)
with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name='ram_utils',
    version='0.0.1',
    author='Sigdel Shree Ram',
    author_email='shreeramsigdel77@gmail.com',
    description='Utilities packages',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/shreeramsigdel77/ram_utils.git",
    packages=packages,
    classfifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'numpy>=1.18.0',
        'opencv-python>=4.2.0.0',
        'pylint>=2.4.2',
    ],
    python_requires='>=3.7',

    
)