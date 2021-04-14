from setuptools import setup, find_packages
import codecs
import os

VERSION = '0.7.5'
DESCRIPTION = 'Speed detection library for automobile'
LONG_DESCRIPTION = 'A package that could serve as a speedometer for autonomous car using dashboard camera'

with open('requirements.txt') as f:
    required = f.read().splitlines()

# Setting up
setup(
    name='car_speed_detection',
    version=VERSION,
    author='Shao-chieh Lien',
    author_email='shaochiehlien@gmail.com',
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    long_description=LONG_DESCRIPTION,
    packages=find_packages(),
    license='MIT',
    install_requires=required,
    keywords=['python', 'car speed detection', 'software-based speedometer', 'dashboard camera', 'optical flow', 'machine learning', 'keras'],
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ]
)
