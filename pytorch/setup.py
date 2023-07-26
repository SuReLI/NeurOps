""" standard setup.py file for neurops"""

from setuptools import setup

with open("README.md", 'r') as f:
    long_description = f.read()

REQUIREMENTS = [i.strip() for i in open("requirements.txt").readlines()]

setup(
    name='neurops',
    version='0.1',
    description='NeurOps',
    author='Kaitlin Maile',
    author_email='kaitlinmaile@gmail.com',
    packages=['neurops'],
    install_requires=REQUIREMENTS,
    long_description=long_description
)
