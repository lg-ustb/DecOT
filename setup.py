from setuptools import setup

NAME = 'DecOT'
VERSION = '1.0.1'
PACKAGES = ['DecOT']

setup(name=NAME,
      version=VERSION,
      packages=PACKAGES,
      python_requires=">=3.6.0",

      url="https://github.com/lg-ustb/DecOT",
      author="lg_ustb",
      author_email="lg_ustb@163.com",

      install_requires=["pandas", "numpy", 'scipy', 'POT', 'rpy2', 'multiprocess', 'matlab'],
      
      )
