import os
from setuptools import setup, find_packages


# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name = "divergence-estimation",
    version = "0.0.1",
    author = "Brandon Oselio",
    author_email = "boselio@umich.edu",
    description = ("Divergence Estimation Package"),
    license = "Apache 2.0",
    keywords = "bayes error, divergence, henze-penrose",
    packages = find_packages(),
    long_description=read('README.md'),
)
