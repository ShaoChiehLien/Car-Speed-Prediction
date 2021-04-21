from setuptools import setup, find_packages
import codecs
import os

VERSION = '0.7.5'
DESCRIPTION = 'Speed detection library for automobile'

with open('requirements.txt') as f:
    required = f.read().splitlines()

with open("README.md") as f:
    long_description = f.read()

# Setting up
setup(
    name='car_speed_detection',
    version=VERSION,
    author='Shao-chieh Lien',
    author_email='shaochiehlien@gmail.com',
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    long_description=long_description,
    packages=find_packages(),
    # Referred to: https://setuptools.readthedocs.io/en/latest/userguide/datafiles.html
    # package_data = {'car_speed': ['*/**/*', '*/**/**/*', '*/**/**/**/*', '*/**/**/**/**/*']},
    license='MIT',
    install_requires=required,
    keywords=['python', 'car speed detection', 'software-based speedometer', 'dashboard camera', 'optical flow', 'machine learning', 'keras'],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ]
)
